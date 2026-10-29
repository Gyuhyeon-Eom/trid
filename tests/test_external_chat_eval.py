"""Comparison scoring must not reward extra text or hide missing/failed rows."""
import re
import unittest
from scripts.evaluate_external_chat import content_score, evidence_grammar, format_ok, product_score, summarize, wilson
from trid.prompts import rag_prompt


def grammar_regex(grammar):
    """Expand the builder's GBNF subset (literals, refs, |, (...), ?) into one anchored Python regex."""
    rules = dict(line.split(' ::= ', 1) for line in grammar.splitlines())
    token = re.compile(r'"((?:\\.|[^"\\])*)"|(n\d+|root)|([()|?])|\s+')

    def expand(name):
        out = []
        for literal, ref, op in token.findall(rules[name]):
            if literal:
                out.append(re.escape(literal.encode().decode('unicode_escape').encode('latin-1').decode()))
            elif ref:
                out.append('(?:' + expand(ref) + ')')
            elif op:
                out.append(op.replace('(', '(?:'))
        return ''.join(out)
    return re.compile(expand('root'))


class ExternalChatScoringTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(id='a', origin='authored', task='rag_value', gold=[['10만 원']],
                        sources=['D1'], answer='10만 원 [D1]')

    def test_trailing_prose_is_not_a_product_correct_answer(self):
        exact = product_score(self.row, '10만 원 [D1]')
        extra = product_score(self.row, '10만 원 [D1] 추가로 승인 없이 사용하세요.')
        self.assertTrue(exact['strict_product'])
        self.assertTrue(extra['strict'])  # old scorer deliberately retained for continuity
        self.assertFalse(extra['strict_product'])

    def test_evidence_grammar_admits_only_spans_copied_from_the_cited_passage(self):
        prompt = rag_prompt([('D1', '숙박비 상한은 "광역시" 8만 원이다.'), ('D2', '일비는 2만 원이다.')], '얼마?')
        accepts = grammar_regex(evidence_grammar(prompt)).fullmatch
        for ok in ('8만 원 [D1]', '확인 불가', '근거 충돌', '2만 원 [D2]', '8만 원 [D1], 2만 원 [D2]', '"광역시" [D1]'):
            self.assertTrue(accepts(ok), ok)
        for bad in ('8만 원 [D2]', '9만 원 [D1]', '만 원 [D1]', '8만 원', '[D1] 8만 원', '8만 원 [D1] 입니다',
                    '확인 불가입니다'):
            self.assertFalse(accepts(bad), bad)

    def test_source_first_order_is_content_correct_but_wrong_value_is_not(self):
        swapped = product_score(self.row, '[D1] 10만 원')
        self.assertFalse(swapped['strict'])
        self.assertTrue(swapped['content_strict'])
        self.assertFalse(content_score(self.row, '[D1] 15만 원')['content_value'])
        self.assertFalse(content_score(self.row, '[D2] 10만 원')['content_strict'])
        prose = content_score(self.row, '지원 한도는 10만 원입니다 [D1]')
        self.assertFalse(prose['content_value'])
        self.assertTrue(prose['mentions_gold'])
        abstain = dict(self.row, gold=[], sources=[], answer='확인 불가')
        self.assertTrue(content_score(abstain, '근거에 없어 확인 불가입니다.')['mentions_gold'])

    def test_full_grammar_accepts_two_values_not_brackets_or_empty_values(self):
        self.assertTrue(format_ok('1,000원 [D1], 30일 [D2]'))
        self.assertTrue(format_ok('근거 충돌'))
        for text in (' [D1]', '[전화] [D1]', '확인 불가입니다', '30일 [D1],', '30일 [D1] 설명'):
            with self.subTest(text=text):
                self.assertFalse(format_ok(text))

    def test_failures_remain_in_denominator_and_unknown_telemetry_is_null(self):
        correct = dict(generated='10만 원 [D1]', **product_score(self.row, '10만 원 [D1]'), skipped_context=False)
        failed = dict(generated=None, **product_score(self.row, ''), skipped_context=True)
        summary = summarize([self.row, self.row], [correct, failed])['all']
        self.assertEqual(summary['total'], 2)
        self.assertEqual(summary['strict_product'], 1)
        self.assertEqual(summary['skipped_context'], 1)
        self.assertIsNone(summary['errors'])
        self.assertIsNone(summary['truncated_output'])
        self.assertEqual(summary['format_error'], 0)
        with self.assertRaises(ValueError):
            summarize([self.row, self.row], [correct])

    def test_known_error_and_truncation_are_counted(self):
        result = dict(generated='', **product_score(self.row, ''), exception='TimeoutError',
                      truncated_output=False, context_truncated=False, skipped_context=False)
        summary = summarize([self.row], [result])['all']
        self.assertEqual(summary['errors'], 1)
        self.assertEqual(summary['truncated_output'], 0)
        self.assertEqual(summary['strict_product'], 0)

    def test_zero_success_does_not_imply_zero_population_probability(self):
        lo, hi = wilson(0, 12)
        self.assertAlmostEqual(lo, 0)
        self.assertGreater(hi, .2)
        self.assertIsNone(wilson(0, 0))


if __name__ == '__main__':
    unittest.main()
