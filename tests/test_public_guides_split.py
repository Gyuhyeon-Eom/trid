import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import extract_public_guides as epg  # noqa: E402


def row(rid, answer='국가계약법 제5조에 따라 처리합니다.', flags=()):
    return {'id': rid, 'kind': 'qa_item', 'text': f'질문: 무엇인가요?\n답변: {answer}', 'flags': list(flags),
            'provenance': {'institution': '기관'}, 'meta': {'fields': {}}}


class ExclusiveSplitTest(unittest.TestCase):
    """B-017 shipped a reserved-overlap row inside the candidate file (C-019). The split must hold it, and the
    consumer check must catch any held row that still sits in a candidate part."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_out = epg.OUT
        epg.OUT = Path(self.tmp.name)
        src = epg.OUT / 'src'
        src.mkdir()
        rows = {'qa_items': [row('a:1'), row('a:2'), row('a:3')], 'documents': [], 'held': [row('a:4', flags=['hold:empty'])]}
        for name, sel in rows.items():
            (src / f'{name}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in sel))
        (src / 'overlap-148.json').write_text(json.dumps({'matches': [], 'partial_review': [{'corpus_id': 'a:2'}]}))
        self.notes = epg.OUT / 'notes.json'
        self.notes.write_text(json.dumps({'notes': {'a:3': '원문 절단'}}, ensure_ascii=False))

    def tearDown(self):
        epg.OUT = self.old_out
        self.tmp.cleanup()

    def parts(self):
        out = {}
        for name in epg.CANDIDATE_PARTS + ('held',):
            for x in (epg.OUT / 'split' / f'{name}.jsonl').read_text().split('\n'):
                if x:
                    out[json.loads(x)['id']] = name
        return out

    def test_overlap_and_annotated_rows_are_held_once(self):
        epg.resplit('src', 'split', str(self.notes))
        parts = self.parts()
        self.assertEqual(parts, {'a:1': 'qa_rule_or_procedure', 'a:2': 'held', 'a:3': 'held', 'a:4': 'held'})
        self.assertEqual(epg.check_version('split'), [])

    def test_check_flags_held_row_left_in_candidate_part(self):
        epg.resplit('src', 'split', str(self.notes))
        cand = epg.OUT / 'split' / 'qa_rule_or_procedure.jsonl'
        moved = row('a:2')
        moved['meta']['status_holds'] = []
        cand.write_text(cand.read_text() + json.dumps(moved, ensure_ascii=False) + '\n')
        problems = epg.check_version('split')
        self.assertTrue(any('a:2: held row in candidate part' in p for p in problems))
        self.assertTrue(any('a:2: in' in p for p in problems))
        self.assertTrue(any('manifest rows/sha' in p for p in problems))


if __name__ == '__main__':
    unittest.main()
