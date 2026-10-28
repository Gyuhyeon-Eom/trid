import http.client,json,threading,unittest
from pathlib import Path
from types import SimpleNamespace
from scripts.institution_pdf_app import make_server
from test_pdf_documents import CharTokenizer


class PdfAppTests(unittest.TestCase):
    def setUp(self):
        self.state=SimpleNamespace(document=None,document_id=None,lock=threading.Lock(),tok=CharTokenizer())
        self.server,self.url=make_server(self.state,token='test-secret')
        self.thread=threading.Thread(target=lambda:self.server.serve_forever(poll_interval=.01),daemon=True);self.thread.start()
        self.addCleanup(self.close)
    def close(self):
        self.server.shutdown();self.server.server_close();self.thread.join(1)
    def request(self,method,path,body=b'',headers=None):
        connection=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=5)
        connection.request(method,path,body=body,headers=headers or {});r=connection.getresponse();result=(r.status,r.read(),dict(r.getheaders()));connection.close();return result
    def test_local_host_token_origin_and_no_cache(self):
        self.assertEqual(self.server.server_address[0],'127.0.0.1')
        status,body,headers=self.request('GET','/?token=test-secret')
        self.assertEqual(status,200);self.assertEqual(headers['Cache-Control'],'no-store')
        self.assertEqual(self.request('GET','/')[0],403)
        self.assertEqual(self.request('GET','/?token=test-secret',headers={'Host':'evil.example'})[0],403)
        self.assertEqual(self.request('POST','/api/clear',b'{}')[0],403)
        self.assertEqual(self.request('POST','/api/clear',b'{}',{'X-Trid-Token':'test-secret','Origin':'https://evil.example'})[0],403)
    def test_upload_clear_and_stale_document(self):
        path=Path(__file__).resolve().parents[1]/'output/pdf/trid-pdf-sample.pdf'
        if not path.exists():self.skipTest('PDF smoke fixture unavailable')
        headers={'X-Trid-Token':'test-secret','X-File-Name':'../../sample.pdf'}
        status,body,_=self.request('POST','/api/load',path.read_bytes(),headers)
        self.assertEqual(status,200);loaded=json.loads(body);self.assertEqual(loaded['name'],'sample.pdf');self.assertEqual(len(loaded['pages']),2)
        self.assertIsNotNone(self.state.document)
        query=json.dumps({'document_id':'wrong','question':'식비'}).encode()
        self.assertEqual(self.request('POST','/api/ask',query,headers)[0],409)
        self.assertEqual(self.request('POST','/api/clear',b'{}',headers)[0],200)
        self.assertIsNone(self.state.document);self.assertIsNone(self.state.document_id)
