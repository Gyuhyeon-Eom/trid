"""별표 grid structure (scripts/structure_statute_tables.py): row splitting, merged cells, irregular grids."""
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from scripts.structure_statute_tables import split, structure


def table(rows, cols, cells):
    """hwp5proc-like XML; cells: (row, col, rowspan, colspan, [paragraph texts])."""
    by_row = {}
    for r, c, rs, cs, parts in cells:
        paras = ''.join(f'<Paragraph><LineSeg><Text>{p}</Text></LineSeg></Paragraph>' for p in parts)
        by_row.setdefault(r, []).append(f'<TableCell row="{r}" col="{c}" rowspan="{rs}" colspan="{cs}"><ColumnSet>{paras}'
                                        f'</ColumnSet></TableCell>')
    body = ''.join(f'<TableRow>{"".join(by_row[r])}</TableRow>' for r in sorted(by_row))
    return ET.fromstring(f'<TableControl><TableBody rows="{rows}" cols="{cols}">{body}</TableBody></TableControl>')


class StructureTests(unittest.TestCase):
    def test_parallel_paragraph_lists_become_rows_and_merged_cells_repeat(self):
        t = structure(table(4, 3, [
            (0, 0, 1, 3, ['국외 여비 지급표', '(단위: 달러)']),
            (1, 0, 1, 1, ['구분']), (1, 1, 1, 1, ['등급']), (1, 2, 1, 1, ['일비']),
            (2, 0, 1, 1, ['1. 장관급']), (2, 1, 1, 1, ['가', '나']), (2, 2, 1, 1, ['60', '50']),
            (3, 0, 1, 3, ['비고: 실비로 한다.'])]))
        self.assertEqual(t['caption'], ['국외 여비 지급표', '(단위: 달러)'])
        self.assertEqual([r['cells'] for r in t['rows']], [{'구분': '1. 장관급', '등급': '가', '일비': '60'},
                                                           {'구분': '1. 장관급', '등급': '나', '일비': '50'}])
        self.assertEqual(t['notes'], ['비고: 실비로 한다.'])
        self.assertTrue(t['grid']['complete'])

    def test_continuation_parts_are_joined_not_split(self):
        t = structure(table(2, 3, [
            (0, 0, 1, 1, ['구분']), (0, 1, 1, 1, ['철도운임']), (0, 2, 1, 1, ['숙박비']),
            (1, 0, 1, 1, ['제1호']), (1, 1, 1, 1, ['실비', '(특실)']), (1, 2, 1, 1, ['실비', '(상한액: 100,000)'])]))
        self.assertEqual([r['cells'] for r in t['rows']],
                         [{'구분': '제1호', '철도운임': '실비 (특실)', '숙박비': '실비 (상한액: 100,000)'}])

    def test_row_merged_label_is_repeated_in_each_row_it_covers(self):
        t = structure(table(3, 2, [
            (0, 0, 1, 1, ['구분']), (0, 1, 1, 1, ['해당 공무원']),
            (1, 0, 2, 1, ['제1호']), (1, 1, 1, 1, ['가. 장관']), (2, 1, 1, 1, ['나. 차관'])]))
        self.assertEqual([r['cells']['구분'] for r in t['rows']], ['제1호', '제1호'])

    def test_side_by_side_tables_in_one_grid_get_no_rows(self):
        t = structure(table(2, 4, [
            (0, 0, 1, 1, ['근무연수']), (0, 1, 1, 1, ['지급액']), (0, 2, 1, 1, ['근무연수']), (0, 3, 1, 1, ['지급액']),
            (1, 0, 1, 1, ['5년 이상']), (1, 1, 1, 1, ['50%']), (1, 2, 1, 1, ['10년 이상']), (1, 3, 1, 1, ['60%'])]))
        self.assertEqual(t['rows'], [])
        self.assertIn('irregular_layout_rows_not_emitted', t['flags'])

    def test_label_broken_after_an_enumerator_stays_whole_beside_split_rows(self):
        t = structure(table(2, 3, [
            (0, 0, 1, 1, ['구분']), (0, 1, 1, 1, ['경력']), (0, 2, 1, 1, ['환산율']),
            (1, 0, 1, 1, ['1.', '공무원경력']), (1, 1, 1, 1, ['가. 공무원 경력', '나. 고용직 경력']),
            (1, 2, 1, 1, ['100퍼센트', '80퍼센트'])]))
        self.assertEqual([r['cells'] for r in t['rows']],
                         [{'구분': '1. 공무원경력', '경력': '가. 공무원 경력', '환산율': '100퍼센트'},
                          {'구분': '1. 공무원경력', '경력': '나. 고용직 경력', '환산율': '80퍼센트'}])
        self.assertEqual(t['rows'][0]['repeated'], ['구분'])


class SplitTests(unittest.TestCase):
    def test_only_regular_tables_of_checked_docs_are_queryable_and_review_is_per_doc(self):
        def doc(i, status, flags, rows):
            return {'id': i, 'title': i, 'status': status, 'unit': None, 'revised': None, 'statute_effective_date': None,
                    'tables': [{'flags': flags, 'caption': [], 'notes': [], 'rows': rows}]}
        row = [{'section': None, 'cells': {'구분': '제1호', '일비': '25,000'}}]
        with tempfile.TemporaryDirectory() as d:
            src, out = Path(d) / 'src', Path(d) / 'out'
            src.mkdir()
            (src / 'structured.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in (
                doc('ok', 'structured', [], row), doc('seen', 'structured', [], row),
                doc('odd', 'partly_structured_irregular_table', ['irregular_layout_rows_not_emitted'], []),
                doc('nest', 'structured', ['nested_table_text_not_included'], row),
                doc('diff', 'structured_check_differs', [], row))))
            review = Path(d) / 'review.jsonl'
            review.write_text(json.dumps({'id': 'seen', 'result': 'match', 'tables_reviewed': [0], 'review_ref': 'r'}) + '\n')
            split(src, review, out)
            rows = [json.loads(x) for x in (out / 'queryable-rows.jsonl').read_text().splitlines()]
            held = [json.loads(x) for x in (out / 'not-queryable.jsonl').read_text().splitlines()]
        self.assertEqual([(r['doc_id'], r['review']) for r in rows],
                         [('ok', 'candidate_unreviewed'), ('seen', 'ai_visual_review_match')])
        self.assertEqual(sorted(h['id'] for h in held), ['diff', 'nest', 'odd'])


if __name__ == '__main__':
    unittest.main()
