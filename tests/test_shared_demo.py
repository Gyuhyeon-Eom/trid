import http.client
import json
import re
import threading
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from scripts.shared_demo import make_server, State, TTL, MAX_BODY
from test_pdf_documents import CharTokenizer


class SharedDemoTests(unittest.TestCase):
    def setUp(self):
        self.secret = 'x' * 40
        self.state = SimpleNamespace(sessions={}, lock=threading.RLock(), work=threading.Lock(), tok=CharTokenizer(), run='test')
        self.state.prune = lambda: State.prune(self.state)
        self.server = make_server(self.state, self.secret)
        self.thread = threading.Thread(target=lambda: self.server.serve_forever(poll_interval=.01), daemon=True)
        self.thread.start()
        self.addCleanup(self.close)

    def close(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(1)

    def request(self, method, path, data=None, headers=None, trusted=True):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        h = {'X-Trid-Proxy': self.secret, 'X-Trid-Host': 'test.tridatum.co'} if trusted else {}
        h.update(headers or {})
        body = json.dumps(data).encode() if data is not None else None
        connection.request(method, path, body, h)
        response = connection.getresponse()
        result = response.status, response.read(), dict(response.getheaders())
        connection.close()
        return result

    def visitor(self):
        status, body, headers = self.request('GET', '/')
        self.assertEqual(status, 200)
        self.assertIn('HttpOnly', headers['Set-Cookie'])
        self.assertIn('Secure', headers['Set-Cookie'])
        self.assertEqual(headers['Cache-Control'], 'no-store')
        token = re.search(rb'name="trid-token" content="([^"]+)"', body).group(1).decode()
        return {'Cookie': headers['Set-Cookie'].split(';')[0], 'X-Trid-Token': token, 'Origin': 'https://test.tridatum.co'}

    def test_proxy_csrf_origin_and_private_files(self):
        self.assertEqual(self.request('GET', '/', trusted=False)[0], 403)
        h = self.visitor()
        self.assertEqual(self.request('GET', '/runs/model.safetensors')[0], 404)
        self.assertEqual(self.request('POST', '/api/sample', {}, {**h, 'Origin': 'https://evil.test'})[0], 403)
        self.assertEqual(self.request('POST', '/api/sample', {}, {**h, 'X-Trid-Token': 'wrong'})[0], 403)
        self.assertEqual(self.request('POST', '/api/sample', {}, {**h, 'Content-Length': str(MAX_BODY + 1)})[0], 413)

    def test_documents_are_isolated_and_clear_only_own(self):
        a, b = self.visitor(), self.visitor()
        loaded = json.loads(self.request('POST', '/api/sample', {}, a)[1])
        question = {'document_id': loaded['document_id'], 'question': '해외 식비는?'}
        self.assertEqual(self.request('POST', '/api/ask', question, b)[0], 409)
        self.assertEqual(self.request('POST', '/api/sample', {}, {**b, 'X-Trid-Token': a['X-Trid-Token']})[0], 403)
        self.request('POST', '/api/sample', {}, b)
        self.request('POST', '/api/clear', {}, a)
        sessions = list(self.state.sessions.values())
        self.assertIsNone(sessions[0].document)
        self.assertIsNotNone(sessions[1].document)
        self.assertEqual(self.request('POST', '/api/ask', question, a)[0], 409)

    def test_expired_session_and_invalid_input(self):
        h = self.visitor()
        self.assertEqual(self.request('POST', '/api/ask', [], h)[0], 422)
        self.assertEqual(self.request('POST', '/api/text', {'text': 'x'}, h)[0], 422)
        session = next(iter(self.state.sessions.values()))
        session.touched = time.monotonic() - TTL - 1
        self.state.prune()
        self.assertFalse(self.state.sessions)
        self.assertEqual(self.request('POST', '/api/sample', {}, h)[0], 403)


if __name__ == '__main__':
    unittest.main()
