import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file
from trid.model import Config
from trid.text import train_tokenizer
from trid.retrieval_head import (RetrievalHead, PassageIndex, tokenize, pack, maxsim, encode_passages, encode_query,
                                 score, search, contrastive_loss, load_head, QUERY_MARK, PASSAGE_MARK, STAGE, WEIGHTS)

PAIRS = [('출장 식비 상한은?', '출장 여비 규정 제12조: 식비 상한은 하루 25,000원이다.'),
         ('도서관 휴관일은 언제야?', '도서관 운영 안내: 휴관일은 매주 월요일과 법정 공휴일이다.'),
         ('주차 요금은 얼마지?', '청사 주차장 이용 수칙: 주차 요금은 30분당 500원이다.'),
         ('장학금 신청 마감은?', '장학 안내: 신청 마감은 4월 22일(월) 18시까지이다.')]


class RetrievalHeadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        rows = [{'prompt': q, 'answer': p} for q, p in PAIRS]
        self.tok = train_tokenizer(rows, Path(self.tmp.name) / 'tok.json', 400)
        self.cfg = Config(vocab_size=self.tok.get_vocab_size(), dim=32, layers=1, heads=4, hidden_dim=64, context_length=128)
        torch.manual_seed(0)
        self.model = RetrievalHead(self.cfg, 16).eval()
        self.pad = self.tok.token_to_id('<|pad|>')

    def test_forward_shapes_and_unit_norm(self):
        seqs = [tokenize(self.tok, p, PASSAGE_MARK, 64) for _, p in PAIRS[:2]]
        self.assertEqual(seqs[0][0], self.tok.token_to_id('<|bos|>'))
        tokens, lengths = pack(seqs, self.pad, 'cpu')
        v, mask = self.model(tokens, lengths)
        self.assertEqual(tuple(v.shape), (2, tokens.shape[1] - 1, 16))
        self.assertEqual(tuple(mask.shape), (2, tokens.shape[1] - 1))
        self.assertEqual(mask.sum(-1).tolist(), [len(s) - 1 for s in seqs])
        norms = v.norm(dim=-1)
        self.assertTrue(torch.allclose(norms[mask], torch.ones(int(mask.sum())), atol=1e-5))
        self.assertTrue(torch.all(norms[~mask] == 0))
        self.assertEqual(len(tokenize(self.tok, 'x' * 500, PASSAGE_MARK, 10)), 10)

    def test_maxsim_matches_brute_force(self):
        g = torch.Generator().manual_seed(1)
        q = torch.nn.functional.normalize(torch.randn(2, 5, 8, generator=g), dim=-1)
        p = torch.nn.functional.normalize(torch.randn(3, 7, 8, generator=g), dim=-1)
        q_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
        p_mask = torch.tensor([[1] * 7, [1] * 4 + [0] * 3, [1] * 2 + [0] * 5], dtype=torch.bool)
        got = maxsim(q * q_mask[..., None], q_mask, p, p_mask)
        for b in range(2):
            for n in range(3):
                expected = sum(max(float(q[b, i] @ p[n, j]) for j in range(7) if p_mask[n, j])
                               for i in range(5) if q_mask[b, i])
                self.assertAlmostEqual(float(got[b, n]), expected, places=5)
        # the chunked numpy path (fp16 index) agrees with the dense torch path
        index = PassageIndex(p.numpy(), p_mask.sum(-1).numpy())
        s = score(q[1], index, chunk=2)
        self.assertTrue(np.allclose(s, got[1].numpy(), atol=2e-2))

    def test_encode_passages_and_search_ranks_gold_first_after_training(self):
        texts = [p for _, p in PAIRS]
        index = encode_passages(self.model, self.tok, texts, 'cpu', batch_size=3, max_tokens=64)
        self.assertEqual(index.vectors.dtype, np.float16)
        self.assertEqual(index.vectors.shape[0], len(texts)); self.assertEqual(index.vectors.shape[2], 16)
        self.assertEqual(index.lengths.tolist(), [len(tokenize(self.tok, t, PASSAGE_MARK, 64)) - 1 for t in texts])
        for i, length in enumerate(index.lengths):
            self.assertTrue(np.all(index.vectors[i, length:] == 0))
        self.assertEqual(tuple(encode_query(self.model, self.tok, PAIRS[0][0], 'cpu', 16).shape),
                         (len(tokenize(self.tok, PAIRS[0][0], QUERY_MARK, 16)) - 1, 16))
        torch.manual_seed(0)
        model = RetrievalHead(self.cfg, 16)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
        queries, positives = [q for q, _ in PAIRS], texts
        for _ in range(40):
            opt.zero_grad()
            loss, scores = contrastive_loss(model, self.tok, queries, positives, [], 'cpu', 16, 64, .05)
            self.assertTrue(torch.isfinite(loss)); self.assertEqual(tuple(scores.shape), (4, 4))
            loss.backward(); opt.step()
        model.eval()
        index = encode_passages(model, self.tok, texts, 'cpu', max_tokens=64)
        for i, (q, _) in enumerate(PAIRS):
            top = search(model, self.tok, q, index, 2)
            self.assertEqual(top[0][0], i, f'{q} -> {top}')
            self.assertGreater(top[0][1], top[1][1])
        # frozen backbone: only the projection receives gradients, and hard negatives widen the score matrix
        frozen = RetrievalHead(self.cfg, 16)
        loss, scores = contrastive_loss(frozen, self.tok, queries[:2], positives[:2], positives[2:], 'cpu', 16, 64, .05, freeze_backbone=True)
        loss.backward()
        self.assertEqual(tuple(scores.shape), (2, 4))
        self.assertIsNotNone(frozen.proj.weight.grad)
        self.assertTrue(all(p.grad is None for p in frozen.backbone.parameters()))

    def test_save_load_roundtrip(self):
        root = Path(self.tmp.name) / 'run'; root.mkdir()
        (root / 'config.json').write_text(json.dumps(asdict(self.cfg)))
        self.tok.save(str(root / 'tokenizer.json'))
        (root / 'training.json').write_text(json.dumps({'stage': STAGE, 'projection_dim': 16}))
        save_file({k: v.contiguous() for k, v in self.model.state_dict().items()}, str(root / WEIGHTS))
        loaded, tok, device = load_head(root, 'cpu')
        self.assertEqual(device, 'cpu')
        texts = [p for _, p in PAIRS]
        before = score(encode_query(self.model, self.tok, PAIRS[1][0], 'cpu'), encode_passages(self.model, self.tok, texts, 'cpu'))
        after = score(encode_query(loaded, tok, PAIRS[1][0], 'cpu'), encode_passages(loaded, tok, texts, 'cpu'))
        self.assertTrue(np.allclose(before, after, atol=1e-5))
        index = encode_passages(loaded, tok, texts, 'cpu')
        index.save(root / 'index.npz')
        again = PassageIndex.load(root / 'index.npz')
        self.assertTrue(np.array_equal(index.vectors, again.vectors)); self.assertEqual(again.texts, texts)
        (root / 'training.json').write_text(json.dumps({'stage': 'extractive_reader_v1'}))
        with self.assertRaises(ValueError):
            load_head(root, 'cpu')


if __name__ == '__main__':
    unittest.main()
