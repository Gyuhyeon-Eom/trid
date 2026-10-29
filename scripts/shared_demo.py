"""Temporary multi-visitor demo. Only accepts requests from the authenticated edge proxy."""
import argparse
from dataclasses import dataclass, field
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import html
import json
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit, unquote

import torch
from trid.extractive import load_reader, read as read_extractive, passage_spans
from trid.prompts import rag_prompt
from trid.pdf_documents import Document, DocumentError, make_chunks, answer_document

ROOT = Path(__file__).resolve().parents[1]
TTL = 15 * 60
MAX_BODY = 2 * 1024 * 1024
MAX_SESSIONS = 32

def read_chunk(model, _presence, tok, text, question, device):
    """Adapt prompt-relative extractive spans to the document reader's chunk offsets."""
    prompt = rag_prompt([('D1', text)], question)
    result = read_extractive(model, tok, prompt, device)
    abstains = result['presence'] != 0
    span = None
    answer = result['answer']
    if not abstains:
        sid, lo, hi = passage_spans(prompt)[0]
        start, end = result['span']
        if result['source'] != sid or not lo <= start < end <= hi:
            raise DocumentError('모델의 원문 위치 검증에 실패했습니다.')
        span = [start - lo, end - lo]
        answer = result['span_text']
        if text[span[0]:span[1]] != answer:
            raise DocumentError('모델의 원문 위치 검증에 실패했습니다.')
    return {**result, 'answer': answer, 'abstains': abstains, 'source_character_span': span}



@dataclass
class Session:
    csrf: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    touched: float = field(default_factory=time.monotonic)
    document: object = None
    document_id: str | None = None
    calls: list = field(default_factory=list)


class State:
    def __init__(self, run):
        metadata = json.loads((Path(run) / 'training.json').read_text())
        if metadata.get('status') != 'complete':
            raise ValueError('학습 완료 모델만 공유할 수 있습니다.')
        self.model, self.tok, self.device = load_reader(run, device='cpu')
        self.sessions = {}
        self.lock = threading.RLock()
        self.work = threading.Lock()
        self.run = Path(run).name

    def prune(self):
        with self.lock:
            now = time.monotonic()
            for sid in list(self.sessions):
                if now - self.sessions[sid].touched > TTL:
                    del self.sessions[sid]


def parse_pdf(body, name, tok):
    # Parsing untrusted PDFs happens in a separate process with a wall-clock timeout.
    try:
        process = subprocess.Popen(
            [sys.executable, str(ROOT / 'scripts/shared_pdf_extract.py')],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            cwd=ROOT,
        )
        output = []
        def communicate():
            output.append(process.communicate(body)[0])
        communication = threading.Thread(target=communicate, daemon=True)
        communication.start()
        deadline = time.monotonic() + 15
        try:
            while process.poll() is None:
                if time.monotonic() > deadline:
                    raise DocumentError('PDF 처리 시간이 초과됐습니다. 더 작은 문서를 넣어 주세요.')
                if sys.platform == 'darwin':
                    memory = subprocess.run(['/bin/ps', '-o', 'rss=', '-p', str(process.pid)], capture_output=True, timeout=2).stdout.strip()
                    if memory and int(memory) > 512 * 1024:
                        raise DocumentError('PDF 처리 메모리 한도를 넘었습니다. 더 작은 문서를 넣어 주세요.')
                communication.join(.1)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=3)
            communication.join(3)
        if process.returncode or not output:
            raise DocumentError('PDF를 읽지 못했습니다. 작은 텍스트 PDF로 다시 시도해 주세요.')
        result = json.loads(output[0])
    except (subprocess.SubprocessError, ValueError):
        raise DocumentError('PDF를 읽지 못했습니다. 텍스트가 있는 작은 PDF로 다시 시도해 주세요.')
    if 'error' in result:
        raise DocumentError(result['error'])
    name = name.replace('\\', '/').rsplit('/', 1)[-1][:180] or 'document.pdf'
    return Document(name, result['pages'], make_chunks(result['pages'], tok), result['warnings'])


class BoundedServer(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, *args):
        self.slots = threading.BoundedSemaphore(8)
        super().__init__(*args)

    def process_request(self, request, address):
        if not self.slots.acquire(False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, address)
        except Exception:
            self.slots.release()
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self.slots.release()


def make_server(state, secret, port=0, hostname='test.tridatum.co'):
    if len(secret) < 32:
        raise ValueError('Proxy secret must contain at least 32 characters')
    assets = ROOT / 'trid/shared_ui'
    origin = 'https://' + hostname

    class Handler(BaseHTTPRequestHandler):
        server_version = 'trid'
        sys_version = ''

        def setup(self):
            super().setup()
            self.connection.settimeout(20)

        def log_message(self, *args):
            pass

        def authenticated(self):
            supplied = self.headers.get('X-Trid-Proxy', '')
            return (supplied.isascii() and hmac.compare_digest(supplied, secret)
                    and self.headers.get('X-Trid-Host') == hostname)

        def respond(self, status, value, mime='application/json; charset=utf-8', cookie=None):
            body = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            for key, val in {
                'Content-Type': mime, 'Content-Length': str(len(body)), 'Cache-Control': 'no-store',
                'X-Content-Type-Options': 'nosniff', 'X-Frame-Options': 'DENY',
                'Referrer-Policy': 'no-referrer', 'Connection': 'close',
                'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'; object-src 'none'; base-uri 'none'; form-action 'self'",
            }.items():
                self.send_header(key, val)
            if cookie:
                self.send_header('Set-Cookie', f'__Host-trid={cookie}; Path=/; Secure; HttpOnly; SameSite=Strict; Max-Age={TTL}')
            self.end_headers()
            self.wfile.write(body)
            self.close_connection = True

        def session(self, create=False):
            jar = SimpleCookie()
            try:
                jar.load(self.headers.get('Cookie', ''))
                sid = jar['__Host-trid'].value if '__Host-trid' in jar else ''
            except Exception:
                sid = ''
            with state.lock:
                state.prune()
                session = state.sessions.get(sid)
                if session is None and create and len(state.sessions) < MAX_SESSIONS:
                    sid = secrets.token_urlsafe(32)
                    session = Session()
                    state.sessions[sid] = session
                if session:
                    session.touched = time.monotonic()
                return sid, session

        def do_GET(self):
            if not self.authenticated():
                self.respond(403, {'error': '허용되지 않은 연결입니다.'}); return
            path = urlsplit(self.path).path
            if path == '/healthz':
                self.respond(200, {'ok': True, 'model': state.run}); return
            if path == '/app.js':
                self.respond(200, (assets / 'app.js').read_bytes(), 'text/javascript; charset=utf-8'); return
            if path != '/':
                self.respond(404, {'error': '없는 페이지입니다.'}); return
            sid, session = self.session(create=True)
            if session is None:
                self.respond(503, {'error': '현재 테스트 인원이 많습니다. 잠시 후 다시 접속해 주세요.'}); return
            page = ((assets / 'index.html').read_text()
                    .replace('TOKEN_PLACEHOLDER', session.csrf)
                    .replace('MODEL_PLACEHOLDER', html.escape(state.run)))
            self.respond(200, page.encode(), 'text/html; charset=utf-8', cookie=sid)

        def do_POST(self):
            if not self.authenticated() or self.headers.get('Origin') != origin:
                self.respond(403, {'error': '허용되지 않은 요청입니다.'}); return
            sid, session = self.session()
            token = self.headers.get('X-Trid-Token', '')
            if session is None or not token.isascii() or not hmac.compare_digest(token, session.csrf):
                self.respond(403, {'error': '세션이 만료됐습니다. 페이지를 새로고침해 주세요.'}); return
            with state.lock:
                now = time.monotonic()
                session.calls = [t for t in session.calls if now - t < 60]
                if len(session.calls) >= 20:
                    self.respond(429, {'error': '요청이 많습니다. 1분 뒤 다시 시도해 주세요.'}); return
                session.calls.append(now)
            try:
                size = int(self.headers.get('Content-Length', '-1'))
                if self.headers.get('Transfer-Encoding') or not 0 <= size <= MAX_BODY:
                    self.respond(413, {'error': '입력은 2MB 이하로 넣어 주세요.'}); return
                body = self.rfile.read(size)
                if len(body) != size:
                    raise DocumentError('입력 전송이 완료되지 않았습니다.')
                path = urlsplit(self.path).path
                if not state.work.acquire(timeout=2):
                    self.respond(503, {'error': '다른 질문을 처리 중입니다. 잠시 후 다시 시도해 주세요.'}); return
                try:
                    if path in {'/api/load', '/api/sample', '/api/text'}:
                        if path == '/api/load':
                            doc = parse_pdf(body, unquote(self.headers.get('X-File-Name', 'document.pdf')), state.tok)
                        elif path == '/api/sample':
                            content = (ROOT / 'data/reading-demo-v3/travel-named.txt').read_text()
                            doc = Document('가상 출장 규정 · 샘플', [content], make_chunks([content], state.tok), ['시연용 가상 자료입니다. 실제 기관 규정이 아닙니다.'])
                        else:
                            request = json.loads(body)
                            content = request.get('text') if isinstance(request, dict) else None
                            if not isinstance(content, str) or not 10 <= len(content.strip()) <= 5000:
                                raise DocumentError('문서는 10~5,000자로 입력해 주세요.')
                            import unicodedata
                            content = unicodedata.normalize('NFC', content.strip())
                            doc = Document('직접 입력한 문서', [content], make_chunks([content], state.tok), [])
                        with state.lock:
                            session.document = doc
                            session.document_id = secrets.token_urlsafe(24)
                        result = {'document_id': session.document_id, 'name': doc.name, 'pages': doc.pages, 'warnings': doc.warnings}
                    elif path == '/api/clear':
                        with state.lock:
                            session.document = session.document_id = None
                        result = {'cleared': True}
                    elif path == '/api/ask':
                        request = json.loads(body)
                        if not isinstance(request, dict):
                            raise DocumentError('질문 형식을 확인해 주세요.')
                        if session.document is None or request.get('document_id') != session.document_id:
                            self.respond(409, {'error': '이 브라우저에서 문서를 먼저 넣어 주세요.'}); return
                        question = request.get('question')
                        if not isinstance(question, str) or len(question) > 500:
                            raise DocumentError('질문은 500자 이하로 입력해 주세요.')
                        started = time.monotonic()
                        result = answer_document(session.document, question, state.model, None, state.tok, state.device,
                                                 reader=read_chunk)
                        if any(item['prediction']['presence'] == 2 for item in result['checked']):
                            result.update(status='review', answer='모델이 근거 충돌로 판단한 구간이 있습니다. /source로 원문을 확인해 주세요.')
                        result['model'] = state.run
                        result['elapsed_ms'] = round((time.monotonic() - started) * 1000)
                    else:
                        self.respond(404, {'error': '없는 기능입니다.'}); return
                finally:
                    state.work.release()
                self.respond(200, result, cookie=sid)
            except DocumentError as error:
                self.respond(422, {'error': str(error)})
            except (ValueError, UnicodeError):
                self.respond(422, {'error': '입력 형식을 확인해 주세요.'})
            except (ConnectionError, TimeoutError):
                self.close_connection = True
            except Exception:
                self.respond(500, {'error': '처리에 실패했습니다. 작은 문서로 다시 시도해 주세요.'})

    return BoundedServer(('127.0.0.1', port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default=str(ROOT / 'runs/reader-factorial-01-cpt-v1-v2-v4'))
    parser.add_argument('--secret-file', required=True)
    parser.add_argument('--port', type=int, default=51873)
    args = parser.parse_args()
    torch.set_num_threads(4)
    state = State(args.run)
    server = make_server(state, Path(args.secret_file).read_text().strip(), args.port)
    stop = threading.Event()
    def prune():
        while not stop.wait(30):
            state.prune()
    threading.Thread(target=prune, daemon=True).start()
    print(f'trid temporary demo ready on 127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()
        state.sessions.clear()


if __name__ == '__main__':
    main()
