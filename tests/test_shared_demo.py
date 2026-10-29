import http.client
import json
import re
import threading
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from scripts.shared_demo import make_server, State, TTL, MAX_BODY, read_chunk
from trid.pdf_documents import DocumentError
from trid.prompts import rag_prompt
from trid.extractive import passage_spans
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


class ExtractiveChunkTests(unittest.TestCase):
    def prediction(self, text, question, occurrence, presence=0):
        prompt = rag_prompt([('D1', text)], question)
        _, lo, _ = passage_spans(prompt)[0]
        return dict(presence=presence, answer='운영지원팀 [D1]', span_text='운영지원팀',
                    source='D1', span=[lo + occurrence, lo + occurrence + len('운영지원팀')])

    def test_repeated_answer_uses_predicted_offset_not_first_match(self):
        text = '이전 담당: 운영지원팀. 현재 담당: 운영지원팀.'
        start = text.rindex('운영지원팀')
        with patch('scripts.shared_demo.read_extractive', return_value=self.prediction(text, '현재 담당은?', start)):
            result = read_chunk(None, None, None, text, '현재 담당은?', 'cpu')
        self.assertEqual(result['source_character_span'], [start, start + len('운영지원팀')])
        self.assertEqual(result['answer'], '운영지원팀')

    def test_abstention_does_not_expose_latent_span(self):
        text = '담당은 운영지원팀입니다.'
        prediction = self.prediction(text, '기한은?', text.index('운영지원팀'), presence=1)
        prediction['answer'] = '확인 불가'
        with patch('scripts.shared_demo.read_extractive', return_value=prediction):
            result = read_chunk(None, None, None, text, '기한은?', 'cpu')
        self.assertTrue(result['abstains'])
        self.assertIsNone(result['source_character_span'])
        self.assertEqual(result['answer'], '확인 불가')

    def test_span_outside_passage_is_rejected(self):
        prediction = self.prediction('담당은 운영지원팀입니다.', '담당은?', 4)
        prediction['span'] = [0, 5]
        with patch('scripts.shared_demo.read_extractive', return_value=prediction):
            with self.assertRaises(DocumentError):
                read_chunk(None, None, None, '담당은 운영지원팀입니다.', '담당은?', 'cpu')


if __name__ == '__main__':
    unittest.main()
