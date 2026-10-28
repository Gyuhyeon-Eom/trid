import tempfile
import unittest
from pathlib import Path
import torch
from trid.model import Config
from trid.text import train_tokenizer
from trid.korean_reader import KoreanReader, encode, batch, loss, read, decode


class KoreanReaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.doc = '정산은 14일, 예약은 4일. 식비는 3만원이다. 같은 식비는 3만원이다.'
        self.q = '질문에만 있는 9999만원이 정답인가?'
        self.tok = train_tokenizer([{'prompt': self.doc+self.q, 'answer':'3만원'}], Path(self.tmp.name)/'tok.json', 300)
        self.model = KoreanReader(Config(vocab_size=self.tok.get_vocab_size(),dim=32,layers=1,heads=4,hidden_dim=64,context_length=256))
        self.pad = self.tok.token_to_id('<|pad|>')

    def test_labels_do_not_leak_and_question_is_excluded(self):
        plain = encode(self.tok, self.doc, self.q, 256)
        gold = encode(self.tok, self.doc, self.q, 256, ['3만원'], False)
        for key in plain:
            self.assertEqual(plain[key], gold[key])
        self.assertEqual(len(gold['spans']), 2)
        self.assertFalse(any(plain['mask'][:plain['document_start']]))
        for a,b in gold['spans']:
            self.assertEqual(gold['prompt'][a:b+1], '3만원')
        with self.assertRaisesRegex(ValueError, '정답 구간'):
            encode(self.tok, self.doc, self.q, 256, ['9999만원'], False)
        row = encode(self.tok, self.doc, self.q, 256, ['4일'], False)
        self.assertEqual(len(row['spans']), 1)
        self.assertEqual(row['spans'][0][0]-row['document_start'], self.doc.index('4일', self.doc.index('예약')))

    def test_loss_gradient_and_batch_invariance(self):
        rows = [encode(self.tok,self.doc,self.q,256,['3만원'],False),
                encode(self.tok,self.doc,'없는 정보?',256,[],True)]
        total = loss(self.model, rows, self.pad, 'cpu')
        separate = sum(loss(self.model, [r], self.pad, 'cpu') for r in rows)/2
        torch.testing.assert_close(total, separate, atol=1e-5, rtol=1e-5)
        total.backward()
        for weight in [self.model.query.weight,self.model.backbone.embedding.weight,self.model.kind.weight]:
            self.assertTrue(torch.isfinite(weight.grad).all())
            self.assertGreater(float(weight.grad.abs().sum()),0)
        self.assertTrue(torch.isfinite(loss(self.model,[rows[1]],self.pad,'cpu')))

    def test_decode_order_boundary_and_source_only(self):
        row = encode(self.tok,self.doc,self.q,256,['3만원'],False)
        n = len(row['mask']); scores = torch.full((2,n),-100.)
        a,b = row['spans'][0]
        scores[0,a]=10; scores[1,b]=10
        scores[:,0]=1000  # forbidden question position must never win
        self.assertEqual(decode(scores,row),(a,b+1))
        with torch.no_grad():
            self.model.kind.weight.zero_();self.model.kind.bias.copy_(torch.tensor([100.,-100.]))
        result = read(self.model.eval(),self.tok,self.doc,self.q,'cpu')
        a,b = result['source_character_span']
        self.assertTrue(0<=a<b<=len(self.doc))
        self.assertEqual(result['answer'],self.doc[a:b])
        with torch.no_grad():
            self.model.kind.bias.copy_(torch.tensor([-100.,100.]))
        result = read(self.model,self.tok,self.doc,self.q,'cpu')
        self.assertTrue(result['abstains']);self.assertIsNone(result['source_character_span'])

    def test_hidden_only_matches_generation_backbone(self):
        row=encode(self.tok,self.doc,self.q,256)
        tokens=batch([row],self.pad,'cpu')[0]
        _,_,hidden=self.model.backbone(tokens,return_hidden=True)
        torch.testing.assert_close(hidden,self.model.backbone(tokens,hidden_only=True))
