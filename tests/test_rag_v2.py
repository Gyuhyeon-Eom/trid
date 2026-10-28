import importlib.util
import random
import re
import unittest
from pathlib import Path

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'

def load(name):
    spec=importlib.util.spec_from_file_location(name,SCRIPTS/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

generator=load('prepare_rag_curriculum_v2')
evaluator=load('evaluate_rag_eval')

def passages(prompt):
    body=prompt.split('\n',1)[1].rsplit('\n질문: ',1)[0]
    parts=re.split(r'(?m)^\[(D[1-9])\] ',body)[1:]
    return dict(zip(parts[::2],parts[1::2]))

class GeneratorTests(unittest.TestCase):
    def test_answers_are_copied_from_the_cited_passage(self):
        r=random.Random(5)
        for kind in generator.KINDS:
            for i in range(200):
                row=generator.synthetic(r,kind,i,'train')
                docs=passages(row['prompt'])
                if kind in ('missing','conflict'):
                    self.assertEqual(row['answer'],{'missing':'확인 불가','conflict':'근거 충돌'}[kind]);continue
                pairs=re.findall(r'([^\[\]]+?)\s*\[(D[1-9])\]',row['answer'])
                self.assertEqual(len(pairs),2 if kind=='combine' else 1)
                for value,source in pairs:
                    self.assertIn(value.strip().lstrip(',').strip(),docs[source],row['prompt'])

    def test_revision_answer_comes_from_current_not_abolished_passage(self):
        r=random.Random(9)
        for i in range(200):
            row=generator.synthetic(r,'revision',i,'train')
            source=re.search(r'\[(D[1-9])\]$',row['answer']).group(1)
            cited=passages(row['prompt'])[source]
            self.assertTrue(any(m in cited for m in generator.NEW_MARKS))
            self.assertFalse(any(m in cited for m in generator.OLD_MARKS))

class ScoreTests(unittest.TestCase):
    row={'gold':[['18만 원'],['11월 12일']],'sources':['D8','D3'],'answer':'18만 원 [D8], 11월 12일 [D3]'}

    def test_value_and_source_are_scored_separately(self):
        self.assertTrue(evaluator.score(self.row,'18만원 [D8], 11월 12일 [D3]')['strict'])
        wrong_source=evaluator.score(self.row,'18만 원 [D3], 11월 12일 [D3]')
        self.assertEqual((wrong_source['strict'],wrong_source['value'],wrong_source['source']),(False,True,False))
        self.assertFalse(evaluator.score(self.row,'18만 원 [D8]')['value'])  # one of two facts missing
        self.assertFalse(evaluator.score(self.row,'11월 12일 [D3], 18만 원 [D8]')['value'])  # asked order

    def test_abstention(self):
        missing={'gold':[],'sources':[],'answer':'확인 불가'}
        self.assertTrue(evaluator.score(missing,'확인 불가.')['strict'])
        self.assertFalse(evaluator.score(missing,'근거 충돌')['strict'])
        self.assertTrue(evaluator.score(self.row,'확인 불가')['abstained'])

class EvidenceConstraintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tokenizers import Tokenizer
        from trid.constrained import token_bytes
        cls.tok=Tokenizer.from_file(str(SCRIPTS.parent/'runs/trid-korean-medium-base-v1/tokenizer.json'))
        cls.vocab=token_bytes(cls.tok)

    def make(self):
        from trid.constrained import EvidenceConstraint
        return EvidenceConstraint([('D3','지원금은 30만 원이다.'),('D7','담당자는 강하늘이다.')],self.vocab)

    def test_token_bytes_round_trip_korean(self):
        text='제5조(국외 출장) 숙박비는 1박당 미화 180달러, ☎ 044-201-3375'
        self.assertEqual(b''.join(self.vocab[i] for i in self.tok.encode(text).ids).decode(),text)

    def test_values_must_be_copied_from_the_cited_passage(self):
        c=self.make()
        for answer in ['30만 원 [D3]','30만 원 [D3], 강하늘 [D7]','확인 불가','근거 충돌']:
            b=answer.encode()
            self.assertTrue(all(c.allows(b[:k]) for k in range(len(b)+1)),answer)
            self.assertTrue(c.complete(b),answer)
        self.assertFalse(c.allows('40만'.encode()))  # invented value
        self.assertFalse(c.allows('30만 원 [D7'.encode()))  # value not in cited passage
        self.assertFalse(c.allows('강'.encode()[:2]+b' [D7]'))  # byte slice of a character
        self.assertFalse(c.allows('강'.encode()[1:]))  # value starting inside a character
        self.assertFalse(c.allows('30만 원 [D3], '.encode()+'강'.encode()[2:]))
        self.assertFalse(c.allows('30만 원 [D3], 강하늘 [D7], '.encode()))  # pair limit
        self.assertFalse(c.complete('30만 원'.encode()))

    def test_choose_skips_invalid_tokens_and_gates_eos(self):
        import torch
        c=self.make();eos=self.tok.token_to_id('<|eos|>')
        scores=torch.zeros(len(self.vocab))
        scores[eos]=10.0  # EOS is best but the answer is empty
        bad=self.tok.encode('40').ids[0];scores[bad]=9.0
        good=self.tok.encode('30').ids[0];scores[good]=1.0
        self.assertEqual(c.choose(scores,eos,48),good)
        self.assertEqual(c.prefix,self.vocab[good])

    def run_constrained(self,c,scores_for,budget):
        """Generation loop as in trid.runtime.generate: returns (answer bytes, stopped on EOS)."""
        eos=self.tok.token_to_id('<|eos|>')
        for step in range(budget):
            token=c.choose(scores_for(c.prefix),eos,budget-step)
            if token==eos:return c.prefix,True
        return c.prefix,False

    def assert_valid_answer(self,c,answer,stopped):
        self.assertTrue(stopped);self.assertTrue(c.complete(answer))
        text=answer.decode()  # raises on a broken character
        self.assertNotIn('\ufffd',text)
        if text not in ('확인 불가','근거 충돌'):
            passages={'D3':'지원금은 30만 원이다.','D7':'담당자는 강하늘이다.'}
            for value,sid in re.findall(r'(.+?) \[(D\d)\](?:, |$)',text):
                self.assertIn(value,passages[sid])

    def test_generation_ends_on_a_complete_answer_under_adversarial_scores(self):
        import random,torch
        eos=self.tok.token_to_id('<|eos|>')
        # Continuation bytes and long copies rank highest, EOS ranks high; budgets from the minimum up.
        cont=[i for i,p in enumerate(self.vocab) if p and p[0]&0xC0==0x80]
        for seed in range(40):
            rng=random.Random(seed)
            def scores_for(prefix):
                s=torch.tensor([rng.random() for _ in self.vocab])
                s[cont]+=2.0;s[eos]=1.5
                return s
            c=self.make();budget=c.cost(b'')+1+seed%8
            answer,stopped=self.run_constrained(c,scores_for,budget)
            self.assert_valid_answer(c,answer,stopped)

    def test_budget_forces_closing_a_long_copy(self):
        import torch
        c=self.make();eos=self.tok.token_to_id('<|eos|>')
        keep=self.tok.encode('담당자는 강하늘이다.').ids
        def scores_for(prefix):  # the model wants to keep copying D7 and never stop
            s=torch.zeros(len(self.vocab));s[keep]=5.0;s[eos]=-5.0
            return s
        answer,stopped=self.run_constrained(c,scores_for,12)
        self.assert_valid_answer(c,answer,stopped)

    def test_budget_too_small_is_rejected(self):
        import torch
        c=self.make();eos=self.tok.token_to_id('<|eos|>')
        with self.assertRaises(ValueError):
            c.choose(torch.zeros(len(self.vocab)),eos,c.cost(b''))

if __name__=='__main__':unittest.main()
