"""Temporary multi-visitor demo. Only accepts requests from the authenticated edge proxy."""
import argparse
from dataclasses import dataclass, field
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import html
import json
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit, unquote

import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_external_chat import GRAMMAR, MAX_TOKENS, SYSTEM, complete, digest, evidence_grammar, request as llama_request
from trid.extractive import load_reader, read as read_extractive, passage_spans
from trid.prompts import rag_prompt
from trid.pdf_documents import Document, DocumentError, make_chunks, answer_document, retrieve

ROOT = Path(__file__).resolve().parents[1]
TTL = 15 * 60
MAX_BODY = 2 * 1024 * 1024
MAX_SESSIONS = 32
# Source-first answers, one or two pairs like the value-first GRAMMAR: "[D1] 값" or "[D1] 값, [D2] 값".
SOURCE_FIRST_ANSWER = re.compile(r'\[(D[1-9])\]\s*([^\[\],\n]+?)(?:\s*,\s*\[(D[1-9])\]\s*([^\[\],\n]+?))?\.?')

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

def answer_document_chat(document, question, base, family, context, constrained=False):
    """One completion over the retrieved chunks (as D1..Dn), with the evaluation's `grammar_instructed` system prompt
    and optionally its `evidence_constrained` grammar. `value [Dn]` and source-first `[Dn] value` are accepted (the
    evaluation's content_* reading); an answer is shown only when every value is found verbatim in the chunk it
    cites, otherwise the reply is a non-answer and the raw output is not displayed as an answer."""
    question = question.strip()
    if not question:
        raise DocumentError('질문을 입력해 주세요.')
    selected = [(chunk, score) for chunk, score in retrieve(document, question) if '[D' not in chunk.text]
    result = {'file': document.name, 'candidates': [], 'checked_chunks': len(selected),
              'total_chunks': len(document.chunks), 'warnings': document.warnings, 'raw_output': None,
              'scope': '관련도 상위 최대 6개 구간을 확인한 결과입니다. 미검색 구간까지 정보가 없다고 보장하지 않습니다.'}
    if not selected:
        return {**result, 'status': 'not_found', 'answer': '질문과 관련된 문서 구간을 찾지 못했습니다. 질문을 바꾸거나 추출 내용을 확인해 주세요.'}
    prompt = rag_prompt([(f'D{i}', chunk.text) for i, (chunk, _) in enumerate(selected, 1)], question)
    messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}]
    r = complete(base, messages, family, MAX_TOKENS, context, evidence_grammar(prompt) if constrained else None)
    if r.get('skipped_context'):
        raise DocumentError('검색된 문서 구간이 모델 문맥 한도를 넘었습니다. 질문을 더 구체적으로 입력해 주세요.')
    output = r['generated']
    result['raw_output'] = output
    if output.removesuffix('.') == '확인 불가':
        return {**result, 'status': 'not_found', 'answer': '확인한 문서 구간에서 답을 찾지 못했습니다(모델 보류).'}
    if output.removesuffix('.') == '근거 충돌':
        return {**result, 'status': 'review', 'answer': '모델이 근거 충돌로 판단했습니다. /source로 원문을 확인해 주세요.'}
    match = GRAMMAR.fullmatch(output)
    source_first = None if match else SOURCE_FIRST_ANSWER.fullmatch(output)
    pairs = ([(match[1], match[2]), (match[3], match[4])] if match
             else [(source_first[2], source_first[1]), (source_first[4], source_first[3])] if source_first else [])
    candidates = []
    for value, sid in pairs:
        if value is None:
            continue
        value, index = value.strip(), int(sid[1:]) - 1
        chunk = selected[index][0] if index < len(selected) else None
        start = chunk.text.find(value) if chunk and value else -1
        if start < 0:
            return {**result, 'status': 'review', 'answer': '모델 출력이 원문 위치 검증을 통과하지 못해 답으로 표시하지 않습니다.'}
        candidates.append({'page': chunk.page, 'text': chunk.text, 'page_text_start': chunk.start,
                           'retrieval_score': round(selected[index][1], 3),
                           'page_character_span': [chunk.start + start, chunk.start + start + len(value)],
                           'prediction': {'answer': value, 'source': sid, 'abstains': False, 'presence': 0,
                                          'source_character_span': [start, start + len(value)]}})
    if not candidates:
        return {**result, 'status': 'review', 'answer': '모델 출력이 형식 검증을 통과하지 못해 답으로 표시하지 않습니다.'}
    return {**result, 'status': 'answer', 'candidates': candidates,
            'answer': ', '.join(f"{c['prediction']['answer']} [{c['prediction']['source']}]" for c in candidates)}


@dataclass
class Session:
    csrf: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    touched: float = field(default_factory=time.monotonic)
    document: object = None
    document_id: str | None = None
    calls: list = field(default_factory=list)


PUBLIC_FIELDS = ('id', 'label', 'kind', 'base_model', 'license', 'version', 'status', 'evaluation', 'limits')


class State:
    """Serves the versions listed in a versions file. The first `extractive` entry also supplies the chunking
    tokenizer; `llama` entries talk to a loopback llama.cpp server whose model file hash must match the file."""
    def __init__(self, versions):
        self.versions, self.children = {}, []
        if not versions or versions[0]['kind'] != 'extractive':
            raise ValueError('첫 버전은 추출 독해기여야 합니다(문서 구간 분할 토크나이저).')
        try:
            for meta in versions:
                if meta['kind'] == 'extractive':
                    metadata = json.loads((ROOT / meta['run'] / 'training.json').read_text())
                    if metadata.get('status') != 'complete':
                        raise ValueError('학습 완료 모델만 공유할 수 있습니다.')
                    model, tok, device = load_reader(ROOT / meta['run'], device='cpu')
                    self.versions[meta['id']] = dict(meta, model=model, tok=tok, device=device)
                elif meta['kind'] == 'llama':
                    self.versions[meta['id']] = dict(meta, **self.start_llama(meta))
                else:
                    raise ValueError(f"unknown version kind {meta['kind']}")
        except BaseException:
            self.close()
            raise
        self.tok = self.versions[versions[0]['id']]['tok']
        self.default = versions[0]['id']
        self.run = self.versions[self.default]['label']
        self.sessions = {}
        self.lock = threading.RLock()
        self.work = threading.Lock()

    def start_llama(self, meta):
        model_file = ROOT / meta['model_file']
        if digest(model_file) != meta['model_sha256']:
            raise ValueError(f"{meta['id']}: model file hash mismatch")
        port = meta['port']
        try:
            llama_request(f'http://127.0.0.1:{port}', '/health')
            raise RuntimeError(f"{meta['id']}: port {port} is already serving; refusing to adopt an unmanaged server")
        except OSError:
            pass
        log = open(ROOT / '.shared-demo' / f"llama-{meta['id']}.log", 'wb') if (ROOT / '.shared-demo').is_dir() else subprocess.DEVNULL
        child = subprocess.Popen([str(ROOT / meta['llama_server']), '-m', str(model_file), '--host', '127.0.0.1', '--port', str(port),
                                  '-c', str(meta['context']), '-np', '1', '-t', '4', '-tb', '4', '--jinja'],
                                 stdout=log, stderr=log)
        self.children.append(child)
        base = f'http://127.0.0.1:{port}'
        for _ in range(240):
            if child.poll() is not None:
                raise RuntimeError(f"{meta['id']}: llama-server exited")
            try:
                props = llama_request(base, '/props')
                if Path(props['model_path']).resolve() == model_file.resolve():
                    return {'base': base, 'context': props['default_generation_settings']['n_ctx']}
                raise RuntimeError(f"{meta['id']}: another model is served on port {port}")
            except OSError:
                time.sleep(.5)
        raise RuntimeError(f"{meta['id']}: llama-server readiness timeout")

    def public_versions(self):
        return [{k: v[k] for k in PUBLIC_FIELDS} for v in self.versions.values()]

    def close(self):
        for child in self.children:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()

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
    public_sources = ROOT / 'data/public-document-sft-source'
    public_docs = {
        doc['id']: doc for doc in (
            json.loads(line) for line in (public_sources / 'documents.jsonl').read_text().splitlines()
            if line.strip()
        ) if doc['license'] == 'KOGL-Type1'
    }
    public_questions = {}
    for line in (public_sources / 'annotations.jsonl').read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            public_questions.setdefault(row['source_id'], []).append(row['question'])
    public_options = ''.join(
        # Source titles carry HTML entities (e.g. &middot;); decode once, then escape for the page.
        f'<option value="{html.escape(key, quote=True)}">{html.escape(doc["date"] + " · " + html.unescape(doc["title"]))}</option>'
        for key, doc in public_docs.items()
    )
    version_options = ''.join(
        f'<option value="{html.escape(v["id"], quote=True)}">{html.escape(v["label"])}</option>'
        for v in state.public_versions()
    )
    version_cards = ''.join(
        f'<pre class="muted version-card" data-version="{html.escape(v["id"], quote=True)}"'
        f'{"" if v["id"] == state.default else " hidden"}>'
        + html.escape(f'{v["label"]} · {v["version"]} · {v["status"]}\n기반: {v["base_model"]} · 라이선스: {v["license"]}\n'
                      f'실측: {v["evaluation"]}\n한계: {v["limits"]}') + '</pre>'
        for v in state.public_versions()
    )

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
                self.respond(200, {'ok': True, 'model': state.run, 'versions': list(state.versions)}); return
            if path == '/app.js':
                self.respond(200, (assets / 'app.js').read_bytes(), 'text/javascript; charset=utf-8'); return
            if path != '/':
                self.respond(404, {'error': '없는 페이지입니다.'}); return
            sid, session = self.session(create=True)
            if session is None:
                self.respond(503, {'error': '현재 테스트 인원이 많습니다. 잠시 후 다시 접속해 주세요.'}); return
            page = ((assets / 'index.html').read_text()
                    .replace('TOKEN_PLACEHOLDER', session.csrf)
                    .replace('MODEL_PLACEHOLDER', html.escape(state.run))
                    .replace('VERSION_OPTIONS_PLACEHOLDER', version_options)
                    .replace('VERSION_CARDS_PLACEHOLDER', version_cards)
                    .replace('PUBLIC_OPTIONS_PLACEHOLDER', public_options))
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
                            request = json.loads(body)
                            if not isinstance(request, dict):
                                raise DocumentError('문서 선택 형식을 확인해 주세요.')
                            if 'source_id' in request:
                                key = request['source_id']
                                if not isinstance(key, str) or key not in public_docs:
                                    raise DocumentError('목록에서 공개 문서를 선택해 주세요.')
                                source = public_docs[key]
                                content = source['text']
                                warnings = [
                                    f"발표일: {source['date']} · 당시 자료이며 현재 적용 여부는 확인하지 않았습니다.",
                                    f"원문: {source['source']}",
                                    source['attribution'] + ' · 공공누리 제1유형, 텍스트만 이용',
                                    '학습 데이터로 선정된 문서입니다. 독립 성능 평가가 아닙니다.',
                                ]
                                doc = Document(html.unescape(source['title']), [content], make_chunks([content], state.tok), warnings)
                            else:
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
                        if path == '/api/sample' and 'source_id' in request:
                            result['suggested_questions'] = public_questions.get(request['source_id'], [])
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
                        version_id = request.get('version', state.default)
                        if not isinstance(version_id, str) or version_id not in state.versions:
                            raise DocumentError('목록에 있는 모델 버전을 선택해 주세요.')
                        version = state.versions[version_id]
                        started = time.monotonic()
                        if version['kind'] == 'extractive':
                            result = answer_document(session.document, question, version['model'], None, version['tok'],
                                                     version['device'], reader=read_chunk)
                            if any(item['prediction']['presence'] == 2 for item in result['checked']):
                                result.update(status='review', answer='모델이 근거 충돌로 판단한 구간이 있습니다. /source로 원문을 확인해 주세요.')
                        else:
                            result = answer_document_chat(session.document, question, version['base'],
                                                          version['family'], version['context'],
                                                          version['decoding'] == 'evidence_constrained')
                        result.pop('checked', None)
                        result['model'] = version['label']
                        result['version'] = version_id
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
    parser.add_argument('--versions', type=Path, default=ROOT / 'deploy/shared-demo/versions.json',
                        help='JSON list of served versions; the first is the default and must be extractive')
    parser.add_argument('--secret-file', required=True)
    parser.add_argument('--port', type=int, default=51873)
    args = parser.parse_args()
    torch.set_num_threads(4)
    def stop_signal(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop_signal)
    state = None
    try:
        state = State(json.loads(args.versions.read_text()))
        server = make_server(state, Path(args.secret_file).read_text().strip(), args.port)
        stop = threading.Event()
        def prune():
            while not stop.wait(30):
                state.prune()
        threading.Thread(target=prune, daemon=True).start()
        print(f'trid temporary demo ready on 127.0.0.1:{server.server_port} versions={list(state.versions)}', flush=True)
        try:
            server.serve_forever()
        finally:
            stop.set()
            server.server_close()
            state.sessions.clear()
    except KeyboardInterrupt:
        pass
    finally:
        if state is not None:
            state.close()


if __name__ == '__main__':
    main()
