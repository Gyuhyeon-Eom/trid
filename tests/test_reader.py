import tempfile, unittest
from pathlib import Path
import torch
from trid.model import Config
from trid.text import train_tokenizer
from trid.reader import Reader,encode,batch,loss,read


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.prompt='자료로 답하세요.\n자료: [T01] 식비는 3만원이다.\n질문: 9999만원이 맞나요?'
        self.tok=train_tokenizer([{'prompt':self.prompt,'answer':'3만원'}],Path(self.tmp.name)/'tok.json',300)
        self.model=Reader(Config(vocab_size=self.tok.get_vocab_size(),dim=32,layers=1,heads=4,hidden_dim=64,context_length=256))

    def test_gold_does_not_change_inference_inputs_or_candidate_mask(self):
        plain=encode(self.tok,self.prompt,256)
        gold=encode(self.tok,self.prompt,256,'3만원')
        self.assertEqual(plain['ids'],gold['ids']);self.assertEqual(plain['mask'],gold['mask'])
        a,b=gold['start'],gold['end']+1
        self.assertEqual(self.prompt[a:b].strip(),'3만원')
        for i in range(self.prompt.index('\n질문:'),len(self.prompt)):
            self.assertFalse(gold['mask'][i])
        with self.assertRaises(ValueError):encode(self.tok,self.prompt,256,'9999만원')

    def test_loss_backprop_and_null_examples(self):
        rows=[encode(self.tok,self.prompt,256,a) for a in ('3만원','자료없음','규정충돌')]
        value=loss(self.model,rows,self.tok.token_to_id('<|pad|>'),'cpu');value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertGreater(float(self.model.query.weight.grad.abs().sum()),0)
        self.assertGreater(float(self.model.backbone.embedding.weight.grad.abs().sum()),0)

    def test_label_does_not_match_inside_a_different_number(self):
        text='자료로 답하세요.\n자료: 정산은 14일, 예약은 4일.\n질문: 예약 기간은?'
        row=encode(self.tok,text,256,'4일')
        self.assertEqual(row['start'],text.rindex('4일'))

    def test_predicted_span_is_inside_evidence(self):
        with torch.no_grad():
            self.model.kind.weight.zero_();self.model.kind.bias.copy_(torch.tensor([100.,-100.,-100.]))
        result=read(self.model.eval(),self.tok,self.prompt,'cpu')
        a,b=result['source_character_span']
        self.assertGreaterEqual(a,self.prompt.index('\n자료: ')+len('\n자료: '))
        self.assertLessEqual(b,self.prompt.index('\n질문: '))
        self.assertEqual(result['answer'],self.prompt[a:b].strip())
