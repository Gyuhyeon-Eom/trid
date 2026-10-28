import json
from pathlib import Path
import tempfile
import unittest
import torch
from trid.model import Config
from trid.text import train_tokenizer
from trid.korean_reader import encode
from trid.short_reader import SpanModel,PresenceModel,component_loss,read


class ShortReaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.doc='국내 식비는 3만원이다. 정산 기한은 7일이다.';self.q='국내 식비는 얼마인가?'
        self.tok=train_tokenizer([{'prompt':self.doc+self.q,'answer':'3만원'}],Path(self.tmp.name)/'tok.json',300)
        cfg=Config(vocab_size=self.tok.get_vocab_size(),dim=32,layers=1,heads=4,hidden_dim=64,context_length=256)
        self.span=SpanModel(cfg);self.presence=PresenceModel(cfg);self.pad=self.tok.token_to_id('<|pad|>')
        self.pos=encode(self.tok,self.doc,self.q,256,['3만원'],False)
        self.neg=encode(self.tok,self.doc,'담당 부서는?',256,[],True)

    def test_independent_objectives_and_parameters(self):
        self.assertFalse(set(p.data_ptr() for p in self.span.parameters())&set(p.data_ptr() for p in self.presence.parameters()))
        error=component_loss(self.span,[self.pos],self.pad,'cpu','span');error.backward()
        self.assertTrue(torch.isfinite(error));self.assertGreater(float(self.span.query.weight.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in self.presence.parameters()))
        self.assertEqual(len(list(self.span.kind.parameters())),0)
        with self.assertRaisesRegex(ValueError,'answerable'):
            component_loss(self.span,[self.neg],self.pad,'cpu','span')
        self.span.zero_grad(set_to_none=True)
        error=component_loss(self.presence,[self.pos,self.neg],self.pad,'cpu','presence');error.backward()
        self.assertTrue(torch.isfinite(error));self.assertGreater(float(self.presence.kind.weight.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in self.span.parameters()))

    def test_pipeline_gate_does_not_change_span(self):
        self.span.eval();self.presence.eval()
        with torch.no_grad():
            self.presence.kind.weight.zero_();self.presence.kind.bias.copy_(torch.tensor([100.,-100.]))
        yes=read(self.span,self.presence,self.tok,self.doc,self.q,'cpu')
        a,b=yes['source_character_span'];self.assertEqual(yes['answer'],self.doc[a:b])
        with torch.no_grad():self.presence.kind.bias.copy_(torch.tensor([-100.,100.]))
        no=read(self.span,self.presence,self.tok,self.doc,self.q,'cpu')
        self.assertTrue(no['abstains']);self.assertIsNone(no['source_character_span'])
        self.assertEqual(no['best_span_if_answerable'],yes['best_span_if_answerable'])

    def test_curriculum_split_and_balance(self):
        root=Path(__file__).resolve().parents[1]/'data/korean-short-v1'
        if not root.exists():self.skipTest('Local curriculum unavailable')
        sets=[]
        for split in ['train','valid','test']:
            rows=[json.loads(x) for x in (root/f'{split}.jsonl').read_text().splitlines()]
            groups={}
            for r in rows:groups.setdefault(r['group'],[]).append(r)
            for group in groups.values():
                self.assertEqual(len(group),6);self.assertEqual(sum(r['is_impossible'] for r in group),3)
                self.assertEqual(len({r['document'] for r in group}),1)
                for r in group:
                    if not r['is_impossible']:self.assertIn(r['answer'],r['document'])
                    else:self.assertIsNone(r['answer'])
            docs={r['document'] for r in rows}
            for earlier in sets:self.assertFalse(earlier&docs)
            sets.append(docs)
