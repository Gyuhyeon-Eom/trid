import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
import torch
from safetensors.torch import save_file
from trid.model import Config
from trid.text import train_tokenizer
from trid.prompts import rag_prompt, RAG_PREFIX
from trid.extractive import (ExtractiveReader, encode, batch, loss, decode, best_span, parse_answer, load_reader,
                             read, MAX_SPAN, STAGE, WEIGHTS)

PASSAGES = [('D2', '정산 기한 | 14일 전까지\n담당자 | 연정경'), ('D5', '계약 보증금 비율 40%를 적용한다. 전화 1675-3561.')]
QUESTION = '정산 기한은?'


class ExtractiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.prompt = rag_prompt(PASSAGES, QUESTION)
        self.tok = train_tokenizer([{'prompt': self.prompt, 'answer': '14일 전까지 [D2]'}], Path(self.tmp.name) / 'tok.json', 300)
        self.cfg = Config(vocab_size=self.tok.get_vocab_size(), dim=32, layers=1, heads=4, hidden_dim=64, context_length=512)
        self.model = ExtractiveReader(self.cfg)
        self.pad = self.tok.token_to_id('<|pad|>')

    def test_encode_locates_span_and_masks_markers(self):
        row = encode(self.tok, self.prompt, '14일 전까지 [D2]')
        self.assertEqual(row['kind'], 0)
        self.assertEqual(row['prompt'][row['start']:row['end'] + 1], '14일 전까지')
        self.assertEqual(row['passages'][row['char_passage'][row['start']]][0], 'D2')
        self.assertTrue(row['label_decodable'])
        self.assertFalse(any(row['mask'][:len(RAG_PREFIX)]))
        for sid, lo, hi in row['passages']:
            marker = lo - len(f'[{sid}] ')
            self.assertEqual(row['prompt'][marker:lo], f'[{sid}] ')
            self.assertFalse(any(row['mask'][marker:lo]))
        q = row['prompt'].rindex('\n질문: ')
        self.assertFalse(any(row['mask'][q:]))
        self.assertFalse(any(m and row['prompt'][i].isspace() for i, m in enumerate(row['mask'])))
        # value in D5 must resolve inside D5 even though a similar value may appear elsewhere
        row = encode(self.tok, self.prompt, '40% [D5]')
        self.assertEqual(row['passages'][row['char_passage'][row['start']]][0], 'D5')
        for answer, kind in (('확인 불가', 1), ('근거 충돌', 2)):
            r = encode(self.tok, self.prompt, answer)
            self.assertEqual((r['kind'], r['start'], r['end']), (kind, 0, 0))

    def test_unmappable_rows_are_skipped(self):
        self.assertIsNone(parse_answer('14일 [D2], 40% [D5]'))
        self.assertIsNone(parse_answer('자유 문장 답'))
        self.assertIsNone(encode(self.tok, self.prompt, '14일 [D2], 40% [D5]'))
        self.assertIsNone(encode(self.tok, self.prompt, '9999만원 [D2]'))
        self.assertIsNone(encode(self.tok, self.prompt, '40% [D2]'))  # value exists but not in the cited passage
        self.assertIsNone(encode(self.tok, self.prompt, '14일 전까지 [D9]'))
        with self.assertRaises(ValueError):
            encode(self.tok, self.prompt, context=8)
        with self.assertRaises(ValueError):
            encode(self.tok, '자료: 없음\n질문: ?')

    def test_decode_respects_candidate_rules(self):
        row = encode(self.tok, self.prompt)
        n = len(row['mask'])
        for seed in range(30):
            g = torch.Generator().manual_seed(seed)
            scores = torch.randn(2, n, generator=g) * 5
            start, end, _ = best_span(scores, row)
            text = row['prompt'][start:end]
            self.assertTrue(0 < end - start <= MAX_SPAN)
            self.assertEqual(row['char_passage'][start], row['char_passage'][end - 1])
            self.assertGreaterEqual(row['char_passage'][start], 0)
            self.assertTrue(row['start_ok'][start] and row['end_ok'][end - 1])
            self.assertFalse(text[0].isspace() or text[-1].isspace())
            self.assertNotIn('[', text)
            if end < n and row['char_passage'][end] == row['char_passage'][start]:
                self.assertFalse(text[-1].isdigit() and row['prompt'][end].isdigit())
        # a span whose end splits '1675-3561' or crosses into D5 never wins, even with huge scores
        scores = torch.full((2, n), -50.)
        d5_lo = row['passages'][1][1]
        phone = row['prompt'].index('1675-3561')
        scores[0, phone] = 100; scores[1, phone + 1] = 100  # '16' would split the number run
        start, end, _ = best_span(scores, row)
        self.assertNotEqual((start, end), (phone, phone + 2))
        d2_start = row['passages'][0][1]
        scores = torch.full((2, n), -50.); scores[0, d2_start] = 100; scores[1, d5_lo] = 100
        start, end, _ = best_span(scores, row)
        self.assertEqual(row['char_passage'][start], row['char_passage'][end - 1])
        # decode output grammar and source derivation
        kind = torch.tensor([10., -10., -10.])
        gold = encode(self.tok, self.prompt, '14일 전까지 [D2]')
        scores = torch.full((2, n), -50.); scores[0, gold['start']] = 100; scores[1, gold['end']] = 100
        out = decode((kind, scores), row)
        self.assertEqual(out['answer'], '14일 전까지 [D2]')
        self.assertEqual(out['source'], 'D2'); self.assertAlmostEqual(sum(out['presence_probs']), 1., places=5)
        self.assertEqual(decode((torch.tensor([-10., 10., -10.]), scores), row)['answer'], '확인 불가')
        self.assertEqual(decode((torch.tensor([-10., -10., 10.]), scores), row)['answer'], '근거 충돌')

    def test_loss_is_finite_and_backpropagates(self):
        rows = [encode(self.tok, self.prompt, '14일 전까지 [D2]'), encode(self.tok, self.prompt, '확인 불가'),
                encode(self.tok, self.prompt, '근거 충돌')]
        total = loss(self.model, rows, self.pad, 'cpu')
        self.assertTrue(torch.isfinite(total))
        total.backward()
        for weight in [self.model.query.weight, self.model.key.weight, self.model.backbone.embedding.weight, self.model.kind.weight]:
            self.assertTrue(torch.isfinite(weight.grad).all())
            self.assertGreater(float(weight.grad.abs().sum()), 0)
        self.assertTrue(torch.isfinite(loss(self.model, rows[1:], self.pad, 'cpu')))

    def test_save_load_roundtrip(self):
        root = Path(self.tmp.name) / 'run'; root.mkdir()
        (root / 'config.json').write_text(json.dumps(asdict(self.cfg)))
        (root / 'training.json').write_text(json.dumps({'stage': STAGE}))
        self.tok.save(str(root / 'tokenizer.json'))
        save_file({k: v.contiguous() for k, v in self.model.state_dict().items()}, str(root / WEIGHTS))
        model, tok, device = load_reader(root, 'cpu')
        row = encode(self.tok, self.prompt)
        with torch.no_grad():
            a = self.model.eval()(*batch([row], self.pad, 'cpu')); b = model(*batch([row], self.pad, 'cpu'))
        torch.testing.assert_close(a[0], b[0]); torch.testing.assert_close(a[1], b[1])
        result = read(model, tok, self.prompt, device)
        self.assertIn(result['answer'], (f"{result['span_text']} [{result['source']}]", '확인 불가', '근거 충돌'))
        (root / 'training.json').write_text(json.dumps({'stage': 'other'}))
        with self.assertRaises(ValueError):
            load_reader(root, 'cpu')


if __name__ == '__main__':
    unittest.main()
