import json
import random
import tempfile
import unittest
from pathlib import Path

from scripts.build_teacher_qa import ROOT, assemble

TOKENIZER = ROOT / 'runs/trid-korean-medium-cpt-gpu-complete/tokenizer.json'


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def verified(doc, passage, question, answer, kept=True):
    return dict(doc_id=doc, passage_index=passage, question=question, answer=answer, generator='g', verifier='v',
                verifier_answer=answer if kept else None, kept=kept, reasons=[] if kept else ['answer_not_verbatim'])


@unittest.skipUnless(TOKENIZER.exists(), 'trid tokenizer not available')
class AssembleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        docs, rows = [], []
        for d in range(8):
            texts = [f'문서{d} 문단{p} 본문입니다. 지원 금액은 {d}{p}00만원이고 신청 기간은 {d}{p}일까지입니다.' for p in range(2)]
            docs.append(dict(id=f'doc{d}', title=f't{d}', source=f'https://example/{d}', passages=texts))
            for p in range(2):
                rows += [verified(f'doc{d}', p, f'문서{d} 문단{p}의 지원 금액은?', f'{d}{p}00만원'),
                         verified(f'doc{d}', p, f'문서{d} 문단{p}의 신청 기간은?', f'{d}{p}일까지')]
        # same passage, same question, conflicting answers (one kept, one rejected)
        rows += [verified('doc0', 0, '문서0 충돌 질문은?', '000만원'), verified('doc0', 0, '문서0 충돌 질문은?', '0일', kept=False)]
        # same question on two passages of one document with different kept answers
        rows += [verified('doc1', 0, '문서1 공통 질문은?', '1000만원'), verified('doc1', 1, '문서1 공통 질문은?', '1100만원')]
        write_jsonl(self.tmp / 'selected.jsonl', docs)
        (self.tmp / 'select.json').write_text('{}')
        write_jsonl(self.tmp / 'drafts.jsonl', [])
        self.rows = rows

    def build(self, name, rows):
        work = self.tmp / f'work-{name}'
        work.mkdir()
        for f in ('selected.jsonl', 'select.json', 'drafts.jsonl'):
            (work / f).write_bytes((self.tmp / f).read_bytes())
        write_jsonl(work / 'verified.jsonl', rows)
        out = self.tmp / f'out-{name}'
        assemble(work, out, TOKENIZER, .2, 271031)
        return out

    def test_conflicting_question_keys_are_quarantined_whole(self):
        out = self.build('a', self.rows)
        quarantine = [json.loads(l) for l in (out / 'quarantine.jsonl').read_text().splitlines()]
        self.assertEqual(sorted((q['category'], q['doc_id'], q['passage_index'], len(q['rows'])) for q in quarantine),
                         [('cross_passage_conflict', 'doc1', 0, 1), ('cross_passage_conflict', 'doc1', 1, 1),
                          ('same_passage_conflict', 'doc0', 0, 2)])
        prompts = [json.loads(l)['prompt'] for f in ('train', 'valid') for l in (out / f'{f}.jsonl').read_text().splitlines()]
        self.assertTrue(prompts)
        self.assertFalse([p for p in prompts if '충돌 질문' in p or '공통 질문' in p])

    def test_output_does_not_depend_on_verified_row_order(self):
        shuffled = list(self.rows)
        random.Random(3).shuffle(shuffled)
        first, second = self.build('a', self.rows), self.build('b', shuffled)
        for name in ('train.jsonl', 'valid.jsonl', 'quarantine.jsonl'):
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes(), name)


if __name__ == '__main__':
    unittest.main()
