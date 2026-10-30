import http.client
import json
import re
import threading
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from scripts.shared_demo import make_server, State, TTL, MAX_BODY, read_chunk, answer_document_chat, answer_document_case
from trid.pdf_documents import Document, make_chunks
from trid.pdf_documents import DocumentError
from trid.prompts import rag_prompt
from trid.extractive import passage_spans
from test_pdf_documents import CharTokenizer


class SharedDemoTests(unittest.TestCase):
    def setUp(self):
        self.secret = 'x' * 40
        meta = dict(id='v', label='test', kind='extractive', base_model='b', license='l', version='0', status='s',
                    evaluation='e', limits='<script>')
        self.state = SimpleNamespace(sessions={}, lock=threading.RLock(), work=threading.Lock(), tok=CharTokenizer(),
                                     run='test', default='v', versions={'v': meta})
        self.state.prune = lambda: State.prune(self.state)
        self.state.public_versions = lambda: State.public_versions(self.state)
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

    def test_version_cards_are_escaped_and_unknown_version_is_rejected(self):
        status, body, _ = self.request('GET', '/')
        self.assertNotIn(b'<script>', body)
        h = self.visitor()
        loaded = json.loads(self.request('POST', '/api/sample', {}, h)[1])
        ask = {'document_id': loaded['document_id'], 'question': '해외 식비는?', 'version': 'missing'}
        self.assertEqual(self.request('POST', '/api/ask', ask, h)[0], 422)


class ChatBackendTests(unittest.TestCase):
    def ask(self, generated):
        text = '출장 담당은 운영지원팀입니다. 식비는 하루 3만 원입니다.'
        document = Document('d', [text], make_chunks([text], CharTokenizer()), [])
        with patch('scripts.shared_demo.complete', return_value={'generated': generated}):
            return answer_document_chat(document, '식비는 얼마?', 'http://127.0.0.1:1', 'qwen3', 4096), text

    def test_verified_value_is_answered_with_its_exact_span(self):
        for generated in ('3만 원 [D1]', '[D1] 3만 원', '[D1] 3만 원.'):
            result, text = self.ask(generated)
            self.assertEqual(result['status'], 'answer', generated)
            start, end = result['candidates'][0]['page_character_span']
            self.assertEqual(text[start:end], '3만 원')

    def test_two_source_first_pairs_are_each_verified(self):
        result, text = self.ask('[D1] 3만 원, [D1] 운영지원팀')
        self.assertEqual(result['status'], 'answer')
        self.assertEqual([text[a:b] for a, b in (c['page_character_span'] for c in result['candidates'])], ['3만 원', '운영지원팀'])
        result, _ = self.ask('[D1] 3만 원, [D1] 9만 원')  # one unverifiable value withholds the whole answer
        self.assertNotEqual(result['status'], 'answer')

    def test_value_absent_from_cited_passage_is_not_an_answer(self):
        for generated in ('5만 원 [D1]', '3만 원 [D2]', '3만 원', '[D1] 식비는 하루 3만 원이고 담당은 운영지원팀입니다, 확인'):
            result, _ = self.ask(generated)
            self.assertNotEqual(result['status'], 'answer', generated)
            self.assertEqual(result['candidates'], [])

    def test_abstention_and_conflict_are_not_answers(self):
        self.assertEqual(self.ask('확인 불가')[0]['status'], 'not_found')
        self.assertEqual(self.ask('근거 충돌')[0]['status'], 'review')

    def case(self, generated):
        text = '출장 담당은 운영지원팀입니다. 식비는 하루 3만 원입니다.'
        document = Document('d', [text], make_chunks([text], CharTokenizer()), [])
        with patch('scripts.shared_demo.complete', return_value={'generated': generated, 'truncated_output': False}):
            return answer_document_case(document, '식비는 얼마?', 'http://127.0.0.1:1', 'qwen3', 4096), text

    def test_case_mode_highlights_only_verbatim_sentences_and_never_verifies_commentary(self):
        result, text = self.case('[근거] [D1] 식비는 하루 3만 원입니다.\n따라서 하루 3만 원을 받습니다.')
        self.assertEqual(result['status'], 'quoted')
        self.assertFalse(result['commentary_verified'])
        self.assertEqual([text[a:b] for a, b in (c['page_character_span'] for c in result['candidates'])],
                         ['식비는 하루 3만 원입니다.'])

    def test_case_mode_withholds_commentary_sentences_with_invented_numbers(self):
        result, _ = self.case('식비는 하루 5만 원이며 담당자 번호는 02-3456-7890입니다.\n담당은 운영지원팀입니다.')
        self.assertEqual(result['status'], 'quoted')  # the second sentence is verbatim source text
        self.assertEqual(result['commentary'], '담당은 운영지원팀입니다.')
        self.assertNotIn('3456', result['commentary'])
        self.assertIn('3456', result['unsupported_numbers'])
        self.assertTrue(any('1개를 숨겼습니다' in n for n in result['commentary_notes']))

    def test_case_mode_collapses_repeated_sentences(self):
        result, _ = self.case('식비는 하루 3만 원입니다. 식비는 하루 3만 원입니다. 식비는 하루 3만 원입니다.')
        self.assertEqual(result['commentary'], '식비는 하루 3만 원입니다.')
        self.assertTrue(any('반복 2회' in n for n in result['commentary_notes']))

    def test_case_mode_reports_question_echo_as_failure(self):
        result, _ = self.case('질문: 식비는 얼마?\n식비는 하루 3만 원입니다.')
        self.assertEqual(result['status'], 'model_failure')
        self.assertEqual(result['candidates'], [])

    def test_qa_question_part_is_not_evidence(self):
        text = '질문: 강의료 100만원을 줘도 되나요?\n답변: 상한액은 직급별로 40만원입니다.'
        document = Document('d', [text], make_chunks([text], CharTokenizer()), [])
        for generated, status in (('[D1] 100만원', 'review'), ('[D1] 40만원', 'answer')):
            with patch('scripts.shared_demo.complete', return_value={'generated': generated}):
                result = answer_document_chat(document, '상한은?', 'http://127.0.0.1:1', 'qwen3', 4096)
            self.assertEqual(result['status'], status, generated)
        with patch('scripts.shared_demo.complete', return_value={'generated': '강의료는 100만원까지 줄 수 있습니다.', 'truncated_output': False}):
            result = answer_document_case(document, '상한은?', 'http://127.0.0.1:1', 'qwen3', 4096)
        self.assertNotIn('100만원', result['commentary'] or '')

    def test_chat_reply_withholds_unsourced_rule_facts(self):
        from scripts.shared_demo import chat_reply
        with patch('scripts.shared_demo.complete', return_value={'generated': '안녕하세요. 출장 식비는 하루 3만 원입니다.\n무엇을 도와드릴까요?'}):
            r = chat_reply('http://127.0.0.1:1', 'qwen3', 4096, [], '안녕')
        self.assertEqual(r['status'], 'chat')
        self.assertNotIn('3만', r['answer'])
        self.assertIn('무엇을 도와드릴까요?', r['answer'])
        self.assertTrue(any('1개를 숨겼습니다' in n for n in r['commentary_notes']))

    def test_value_hold_passage_numbers_are_neither_quotes_nor_support(self):
        from scripts.shared_demo import PassageChunk
        text = '식비는 하루 3만 원입니다. 담당은 운영지원팀입니다.'
        held = [(PassageChunk(text, 1, 0, 'd', ('d',), False, True), 1.0)]
        document = Document('d', [text], [c for c, _ in held], [])
        with patch('scripts.shared_demo.complete', return_value={'generated': '식비는 하루 3만 원입니다.', 'truncated_output': False}):
            r = answer_document_case(document, '식비는 얼마?', 'http://127.0.0.1:1', 'qwen3', 4096, selected=held)
        self.assertEqual((r['candidates'], r['commentary']), ([], None))
        self.assertTrue(any('원문(쪽)에서 직접 확인' in n for n in r['commentary_notes']))

    def test_long_workspace_passage_is_windowed_for_the_reader(self):
        from scripts.shared_demo import PassageChunk, reader_windows
        source = '가' * 50 + ''.join(f'제{i}조 출장 식비는 하루 {i}만원으로 한다. ' for i in range(1, 40))
        chunk = PassageChunk(source[50:], 1, 50, 'd', ('d',), False, True)
        windows = reader_windows([(chunk, 0.9)], CharTokenizer())
        self.assertGreater(len(windows), 1)
        for w, score in windows:
            self.assertLessEqual(len(w.text), 128)  # CharTokenizer: one token per character
            self.assertEqual(source[w.start:w.start + len(w.text)], w.text)
            self.assertEqual((w.doc_id, w.value_hold, score), ('d', True, 0.9))

    def test_chat_keeps_numbers_the_visitor_gave(self):
        from scripts.shared_demo import chat_reply
        history = [{'role': 'user', 'content': '이번 달 연차가 2일 남았어요.'}, {'role': 'assistant', 'content': '알겠습니다.'}]
        with patch('scripts.shared_demo.complete', return_value={'generated': '연차가 2일 남으셨어요. 연가는 연 21일입니다.'}):
            r = chat_reply('http://127.0.0.1:1', 'qwen3', 4096, history, '제 연차 며칠 남았죠?')
        self.assertIn('2일 남으셨어요', r['answer'])
        self.assertNotIn('21일', r['answer'])

    def test_chat_keeps_arithmetic_of_visitor_figures_but_not_other_numbers(self):
        from scripts.shared_demo import chat_reply
        history = [{'role': 'user', 'content': '숙박비 8만원, 식비 2만원이에요.'}, {'role': 'assistant', 'content': '네.'}]
        reply = '합계는 10만원입니다. 하루 일비는 2만5천원입니다.'
        with patch('scripts.shared_demo.complete', return_value={'generated': reply}):
            r = chat_reply('http://127.0.0.1:1', 'qwen3', 4096, history, '둘을 더하면 얼마예요?')
        self.assertIn('합계는 10만원입니다.', r['answer'])
        self.assertNotIn('일비', r['answer'])
        self.assertTrue(any('계산 결과는 직접' in n for n in r['commentary_notes']))

    def test_request_type_markers(self):
        from scripts.shared_demo import request_type
        self.assertEqual(request_type('회의는 오후 3시, 예산은 20만원이에요. 한 줄로 정리해 주세요.'), 'organize')
        self.assertEqual(request_type('직원들이 실제로 쓴 평균 식비는?'), 'statistic')
        self.assertEqual(request_type('여비와 출장비는 무슨 차이야?'), 'explain')
        self.assertIsNone(request_type('국내 출장 식비 지급 기준은?'))
        self.assertEqual(request_type('아까 제가 말한 출장비 예산이 얼마였죠?'), 'recall')




class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from trid.service.index import Index
        from trid.service.structure import parse_document
        from trid.service.types import Document as ServiceDocument
        from scripts.shared_demo import Workspace
        texts = {'a': ('가기관 출장 규정', '제1조(식비) 국외 출장 식비는 하루 8만원으로 한다.\n제2조(숙박) 국외 출장 숙박비는 1박 25만원으로 한다.'),
                 'b': ('나기관 출장 규정', '제1조(식비) 국외 출장 식비는 하루 6만원으로 한다.\n제2조(숙박) 숙박비는 실비로 한다.')}
        docs, meta = [], {}
        for key, (title, text) in texts.items():
            d = ServiceDocument(key, title, key, {'public'}, text)
            d.passages, d.refs = parse_document(text, key, title)
            docs.append(d)
            meta[key] = {'title': title, 'kind': '행정규칙', 'institution': title[:3], 'date': '2026-01-01', 'date_type': '발령일자',
                         'effective_date': None, 'currency': None, 'source': f'https://example.invalid/{key}', 'license': 'test'}
        self.dir = tempfile.TemporaryDirectory(); self.addCleanup(self.dir.cleanup)
        Index.build(docs).save(self.dir.name)
        Path = __import__('pathlib').Path
        (Path(self.dir.name) / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False))
        (Path(self.dir.name) / 'manifest.json').write_text(json.dumps({'documents': 2, 'by_kind': {'행정규칙': 2}, 'date_range': ['2026-01-01', '2026-01-01']}))
        self.ws = Workspace(self.dir.name)

    def test_ambiguous_institution_is_asked_back_instead_of_picked(self):
        r = self.ws.select('국외 출장 식비는 하루 얼마인가요?')
        self.assertEqual(r['status'], 'clarify')
        self.assertEqual(r['chunks'], [])
        self.assertEqual(set(r['institutions']), {'가기관', '나기관'})

    def test_named_institution_restricts_passages_to_its_document(self):
        r = self.ws.select('나기관 국외 출장 식비는 하루 얼마인가요?')
        self.assertEqual(r['status'], 'answer')
        self.assertEqual(r['applied'], 'b')
        self.assertEqual({c.doc_id for c, _ in r['chunks']}, {'b'})

    def test_institution_missing_from_corpus_does_not_fall_back(self):
        r = self.ws.select('다라공단 국외 출장 식비는 하루 얼마인가요?')
        self.assertEqual(r['status'], 'unknown_institution')
        self.assertEqual(r['chunks'], [])

    def test_pinned_document_is_the_only_source(self):
        r = self.ws.select('국외 출장 식비는 하루 얼마인가요?', pinned='b')
        self.assertEqual(r['applied'], 'b')
        self.assertEqual({c.doc_id for c, _ in r['chunks']}, {'b'})

    def test_follow_up_without_names_continues_previous_document(self):
        r = self.ws.select('그럼 숙박비는요?', context_doc='b')
        self.assertEqual(r['status'], 'answer')
        self.assertTrue(r['continued'])
        self.assertEqual({c.doc_id for c, _ in r['chunks']}, {'b'})
        self.assertTrue(self.ws.continues('그럼 숙박비는요?', 'b'))
        self.assertFalse(self.ws.continues('그럼 숙박비는요?', None))

    def test_named_institution_overrides_previous_document(self):
        r = self.ws.select('가기관 숙박비는 얼마인가요?', context_doc='b')
        self.assertEqual(r['applied'], 'a')
        self.assertNotIn('continued', r)
        self.assertFalse(self.ws.continues('가기관 숙박비는 얼마인가요?', 'b'))

    def test_previous_document_without_relevant_text_is_not_continued(self):
        r = self.ws.select('국외 출장 숙박비 1박', context_doc='b')
        self.assertNotIn('continued', r)
        self.assertEqual(r['applied'], 'a')
        self.assertFalse(self.ws.continues('국외 출장 숙박비 1박', 'b'))

    def test_official_alias_without_documents_is_named_not_substituted(self):
        from scripts.shared_demo import workspace_reply
        r = self.ws.select('부동산원 국외 출장 식비는 하루 얼마인가요?')
        self.assertEqual(r['status'], 'unknown_institution')
        self.assertEqual(r['unknown_institutions'], ['한국부동산원'])
        self.assertIn('한국부동산원(공기업', workspace_reply(self.ws, r)['answer'])

    def test_question_about_absent_topic_lists_no_unrelated_documents(self):
        from scripts.shared_demo import workspace_reply
        r = self.ws.select('국외 출장 항공권 좌석 등급은?')
        self.assertEqual((r['status'], r.get('reason')), ('not_found', 'irrelevant'))
        self.assertEqual((r['chunks'], r['others']), ([], []))
        self.assertNotIn('기관 출장 규정', workspace_reply(self.ws, r)['answer'])

    def test_statistic_question_uses_only_structured_values(self):
        from scripts.shared_demo import statistic_reply
        self.assertEqual(statistic_reply(self.ws, '직원 평균 출장 식비는?')['status'], 'not_found')
        self.ws.facts = [{'metric': '출장 식비 실지출 평균', 'value': v, 'unit': '원', 'basis': 'actual_spending', 'source_id': f's{v}',
                          'subject': {'id': None, 'name': n}, 'period': {'from': '2025-01', 'to': '2025-12'}}
                         for n, v in (('가기관', 20000), ('나기관', 30000))]
        self.ws.embedder = lambda text: __import__('numpy').ones(4) / 2  # every metric matches
        r = statistic_reply(self.ws, '직원 평균 출장 식비는?')
        self.assertEqual(r['status'], 'answer_values')
        self.assertIn('25,000원', r['answer'])

    def test_short_form_resolves_to_official_institution_instead_of_no_data(self):
        from scripts.shared_demo import workspace_reply
        r = self.ws.select('건보공단 직원 출장 식비는?')
        self.assertIn('국민건강보험공단', r['unknown_institutions'])
        self.assertIn('국민건강보험공단', workspace_reply(self.ws, r)['answer'])

    def test_exact_short_rule_name_is_recognised(self):
        r = self.ws.select('나기관 출장 규정에서 숙박비는?')
        self.assertTrue(r['title_matched'])
        self.assertEqual(r['applied'], 'b')

    def test_privacy_review_passages_are_never_retrieved(self):
        from scripts.shared_demo import Workspace
        review = __import__('pathlib').Path(self.dir.name) / 'review.jsonl'
        review.write_text(''.join(json.dumps({'passage_id': p.id, 'value_review': {'privacy_review': True}}) + '\n'
                                  for p in self.ws.index.passages if p.doc_id == 'b'))
        ws = Workspace(self.dir.name, review=review)
        r = ws.select('국외 출장 식비는 하루 얼마인가요?', pinned='b')
        self.assertEqual(r['chunks'], [])
        self.assertNotIn('b', {c.doc_id for c, _ in ws.select('국외 출장 식비는 하루 얼마인가요?')['chunks']})


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
