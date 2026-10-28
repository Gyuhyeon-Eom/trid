"""Loopback-only, in-memory PDF question interface for the local experimental model."""
import argparse,hmac,json,secrets,threading,webbrowser
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit,parse_qs,unquote
import torch
from trid.short_reader import load_pipeline
from trid.pdf_documents import load_pdf,answer_document,DocumentError,MAX_BYTES


class State:
    def __init__(self,run):
        self.span,self.presence,self.tok,self.device=load_pipeline(run)
        self.document=None;self.document_id=None;self.lock=threading.Lock()


def make_server(state,port=0,token=None):
    token=token or secrets.token_urlsafe(32);assets=Path(__file__).resolve().parents[1]/'trid/pdf_ui'
    def token_ok(value):return isinstance(value,str) and value.isascii() and hmac.compare_digest(value,token)
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup();self.connection.settimeout(30)
        def log_message(self,*args):pass  # No filenames, questions or token-bearing URLs in request logs.
        def host_ok(self):return self.headers.get('Host') in {f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
        def respond(self,status,body,content_type='application/json; charset=utf-8'):
            if not isinstance(body,bytes):body=json.dumps(body,ensure_ascii=False).encode()
            self.send_response(status);self.send_header('Content-Type',content_type);self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('X-Frame-Options','DENY')
            self.send_header('Referrer-Policy','no-referrer');self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'; object-src 'none'")
            self.end_headers();self.wfile.write(body)
        def do_GET(self):
            if not self.host_ok():self.respond(403,{'error':'허용되지 않은 호스트입니다.'});return
            url=urlsplit(self.path)
            if url.path=='/app.js':self.respond(200,(assets/'app.js').read_bytes(),'text/javascript; charset=utf-8');return
            supplied=parse_qs(url.query).get('token',[''])[0]
            if url.path!='/' or not token_ok(supplied):
                self.respond(403,{'error':'실행 터미널에서 제공한 앱 주소로 열어 주세요.'});return
            page=(assets/'index.html').read_text().replace('TOKEN_PLACEHOLDER',token)
            self.respond(200,page.encode(),'text/html; charset=utf-8')
        def do_POST(self):
            expected={f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'}
            if not self.host_ok() or not token_ok(self.headers.get('X-Trid-Token','')) or self.headers.get('Origin') not in expected|{None}:
                self.respond(403,{'error':'허용되지 않은 요청입니다.'});return
            try:
                size=int(self.headers.get('Content-Length','-1'))
                if not 0<=size<=MAX_BYTES:raise DocumentError('입력 크기는 20MB 이하로 제한됩니다.')
                if self.headers.get('Transfer-Encoding'):raise DocumentError('지원하지 않는 전송 형식입니다.')
                body=self.rfile.read(size);path=urlsplit(self.path).path
                with state.lock:
                    if path=='/api/load':
                        doc=load_pdf(body,unquote(self.headers.get('X-File-Name','document.pdf')),state.tok)
                        state.document=doc;state.document_id=secrets.token_urlsafe(16)
                        result={'document_id':state.document_id,'name':doc.name,'pages':doc.pages,'warnings':doc.warnings,'chunks':len(doc.chunks)}
                    elif path=='/api/clear':
                        state.document=None;state.document_id=None;result={'cleared':True}
                    elif path=='/api/ask':
                        request=json.loads(body)
                        if not state.document or request.get('document_id')!=state.document_id:
                            self.respond(409,{'error':'문서가 변경됐습니다. PDF를 다시 확인해 주세요.'});return
                        q=request.get('question')
                        if not isinstance(q,str):raise DocumentError('질문은 텍스트로 입력해 주세요.')
                        result=answer_document(state.document,q,state.span,state.presence,state.tok,state.device)
                    else:self.respond(404,{'error':'없는 기능입니다.'});return
                self.respond(200,result)
            except (DocumentError,ValueError,UnicodeError):
                import sys
                message=str(sys.exc_info()[1]);self.respond(422,{'error':message if isinstance(sys.exc_info()[1],DocumentError) else '입력 형식을 확인해 주세요.'})
            except Exception:
                self.respond(500,{'error':'처리하지 못했습니다. 문서 크기나 텍스트 추출 상태를 확인해 주세요.'})
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler);server.daemon_threads=True
    return server,f'http://127.0.0.1:{server.server_port}/?token={token}'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',default='runs/trid-korean-short-v2');p.add_argument('--port',type=int,default=0);p.add_argument('--open',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);state=State(a.run);server,url=make_server(state,a.port)
    print('trid 기관 PDF · 로컬 전용 · 종료 Ctrl+C',flush=True);print(url,flush=True)
    if a.open:webbrowser.open(url)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close();state.document=None


if __name__=='__main__':main()
