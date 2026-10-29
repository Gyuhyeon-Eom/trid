"""Behavioral regressions for scripts/train_extractive.py: strict answer scoring, checkpoint selection, single-forward
partition-invariant validation, micro-batch gradient accumulation, row accounting and a tiny real CPU training run."""
import contextlib
import io
import json
import math
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
import torch
from safetensors.torch import save_file
from trid.corpus import write_json
from trid.model import Config, Trid
from trid.prompts import rag_prompt
from trid.text import train_tokenizer
from trid.extractive import ExtractiveReader, encode, batch, loss, decode, load_reader, WEIGHTS
from scripts.train_extractive import (load_rows, expected_answer, strict_correct, validate, train_step, better, main,
                                      presence_weights, macro_strict_accuracy, KINDS, NORMALIZATION, SELECTION)

PASSAGES = [('D1', '기본 요금 500원. 추가 요금 500원.'), ('D2', '야간 요금 500원.'), ('D3', '담당자 | 연정경')]
QUESTION = '기본 요금은?'


def point(row, lo, hi, presence=0):
    """Controlled outputs: presence logit at `presence`, span scores that force the span [lo, hi)."""
    n = len(row['mask'])
    kind = torch.full((3,), -10.); kind[presence] = 10.
    scores = torch.full((2, n), -50.); scores[0, lo] = 100.; scores[1, hi - 1] = 100.
    return kind, scores


class StrictScoringTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.prompt = rag_prompt(PASSAGES, QUESTION)
        self.tok = train_tokenizer([{'prompt': self.prompt, 'answer': '500원 [D1]'}], Path(self.tmp.name) / 'tok.json', 300)
        self.gold = encode(self.tok, self.prompt, '500원 [D1]')

    def test_right_span_wrong_presence_fails_strict(self):
        row = self.gold
        lo, hi = row['start'], row['end'] + 1
        self.assertEqual(expected_answer(row), '500원 [D1]')
        out = decode(point(row, lo, hi), row)
        self.assertEqual(out['span'], [lo, hi]); self.assertTrue(strict_correct(out, row))
        for presence in (1, 2):
            out = decode(point(row, lo, hi, presence), row)
            self.assertEqual(out['span'], [lo, hi])  # span head is exactly right ...
            self.assertFalse(strict_correct(out, row))  # ... but the emitted answer is an abstain phrase

    def test_equal_text_at_other_occurrence_in_cited_passage_passes(self):
        row = self.gold
        sid, lo, hi = row['passages'][0]
        second = self.prompt.index('500원', row['end'] + 1)
        self.assertLess(second, hi)
        out = decode(point(row, second, second + len('500원')), row)
        self.assertNotEqual(out['span'], [row['start'], row['end'] + 1])
        self.assertEqual(out['answer'], '500원 [D1]'); self.assertTrue(strict_correct(out, row))

    def test_equal_text_from_wrong_passage_fails(self):
        row = self.gold
        sid, lo, hi = row['passages'][1]
        other = self.prompt.index('500원', lo)
        out = decode(point(row, other, other + len('500원')), row)
        self.assertEqual(out['answer'], '500원 [D2]'); self.assertFalse(strict_correct(out, row))

    def test_abstain_rows_need_the_exact_phrase(self):
        for answer, kind in (('확인 불가', 1), ('근거 충돌', 2)):
            row = encode(self.tok, self.prompt, answer)
            self.assertEqual(expected_answer(row), answer)
            lo, hi = self.gold['start'], self.gold['end'] + 1
            self.assertTrue(strict_correct(decode(point(row, lo, hi, kind), row), row))
            self.assertFalse(strict_correct(decode(point(row, lo, hi, 0), row), row))
            self.assertFalse(strict_correct(decode(point(row, lo, hi, 3 - kind), row), row))


class SelectionTests(unittest.TestCase):
    def test_higher_macro_strict_accuracy_wins_then_lower_loss(self):
        initial = {'macro_strict_accuracy': .5, 'strict_accuracy': .9, 'valid_loss': 2.}
        self.assertTrue(better(initial, None))
        self.assertFalse(better({'macro_strict_accuracy': .5, 'strict_accuracy': .9, 'valid_loss': 2.}, initial))  # equal never replaces
        self.assertTrue(better({'macro_strict_accuracy': .5, 'strict_accuracy': .9, 'valid_loss': 1.9}, initial))  # tie: lower loss
        self.assertFalse(better({'macro_strict_accuracy': .5, 'strict_accuracy': .9, 'valid_loss': 2.1}, initial))
        self.assertTrue(better({'macro_strict_accuracy': .6, 'strict_accuracy': .1, 'valid_loss': 9.}, initial))  # macro beats loss and overall strict
        self.assertFalse(better({'macro_strict_accuracy': .4, 'strict_accuracy': 1., 'valid_loss': .1}, initial))  # overall strict / loss alone never win
        self.assertIn('macro_strict_accuracy', SELECTION)

    def test_macro_mean_is_unweighted_over_kinds_with_rows(self):
        # 67/22/10-like split: a checkpoint that answers every present row but no abstention has high overall strict
        # accuracy and a lower macro mean than one with balanced per-kind accuracy
        skewed = {'present_strict_accuracy': 1., 'missing_strict_accuracy': 0., 'conflict_strict_accuracy': 0.}
        balanced = {'present_strict_accuracy': .6, 'missing_strict_accuracy': .6, 'conflict_strict_accuracy': .6}
        self.assertAlmostEqual(macro_strict_accuracy(skewed), 1 / 3)
        self.assertAlmostEqual(macro_strict_accuracy(balanced), .6)
        self.assertAlmostEqual(macro_strict_accuracy({'present_strict_accuracy': .5, 'missing_strict_accuracy': 1., 'conflict_strict_accuracy': None}), .75)
        self.assertIsNone(macro_strict_accuracy({f'{k}_strict_accuracy': None for k in KINDS}))

    def test_presence_weights_are_inverse_frequency_with_mean_one(self):
        rows = [{'kind': 0}] * 67 + [{'kind': 1}] * 22 + [{'kind': 2}] * 10
        w = presence_weights(rows)
        self.assertAlmostEqual(sum(w) / 3, 1.)
        self.assertAlmostEqual(w[0] * 67, w[1] * 22); self.assertAlmostEqual(w[1] * 22, w[2] * 10)
        self.assertLess(w[0], w[1]); self.assertLess(w[1], w[2])
        self.assertEqual(presence_weights([{'kind': 0}, {'kind': 1}, {'kind': 2}]), [1., 1., 1.])
        self.assertEqual(len(presence_weights([{'kind': 0}] * 4)), 3)  # a kind without rows still has a finite weight


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.prompt = rag_prompt(PASSAGES, QUESTION)
        self.tok = train_tokenizer([{'prompt': self.prompt, 'answer': '500원 [D1]'}], Path(self.tmp.name) / 'tok.json', 300)
        self.cfg = Config(vocab_size=self.tok.get_vocab_size(), dim=32, layers=1, heads=4, hidden_dim=64, context_length=512)
        torch.manual_seed(0); self.model = ExtractiveReader(self.cfg)
        self.pad = self.tok.token_to_id('<|pad|>')
        answers = ['500원 [D1]', '확인 불가', '근거 충돌', '연정경 [D3]', '확인 불가']
        self.rows = [encode(self.tok, self.prompt, a) for a in answers]

    def test_single_forward_per_batch_reproduces_loss_and_exact_answers(self):
        calls = []
        forward = self.model.forward
        self.model.forward = lambda *args: calls.append(1) or forward(*args)
        result = validate(self.model, self.rows, self.pad, 'cpu', 2)
        self.assertEqual(len(calls), math.ceil(len(self.rows) / 2))
        with torch.inference_mode():
            self.model.eval()
            expected_loss = float(loss(self.model, self.rows, self.pad, 'cpu'))  # whole-set N and P, not per-chunk means
            kind, scores = forward(*batch(self.rows, self.pad, 'cpu'))
            outs = [decode((kind[i], scores[i]), r) for i, r in enumerate(self.rows)]
        self.assertAlmostEqual(result['valid_loss'], expected_loss, places=5)
        strict = [strict_correct(o, r) for o, r in zip(outs, self.rows)]
        self.assertAlmostEqual(result['strict_accuracy'], sum(strict) / len(strict))
        self.assertEqual((result['present_rows'], result['missing_rows'], result['conflict_rows']), (2, 2, 1))
        for name in KINDS:
            idx = [i for i, r in enumerate(self.rows) if KINDS[r['kind']] == name]
            self.assertAlmostEqual(result[f'{name}_strict_accuracy'], sum(strict[i] for i in idx) / len(idx))
            self.assertGreater(result[f'{name}_presence_loss'], 0)
        self.assertEqual(result['rows'], 5)
        self.assertTrue(self.model.training is False)

    def test_valid_loss_is_invariant_to_the_batch_partition(self):
        # rows [present, missing, conflict, present, missing, present(longer prompt)]: with chunks of 2 or 4 the
        # present fraction varies per chunk, so a per-chunk mean weighted by chunk length would differ from the
        # whole-set normalisation; chunks also pad independently
        rows = self.rows + [encode(self.tok, rag_prompt(PASSAGES, '담당자는?'), '연정경 [D3]')]  # a longer prompt
        reference = validate(self.model, rows, self.pad, 'cpu', len(rows))
        for size in (1, 2, 4, 5):
            result = validate(self.model, rows, self.pad, 'cpu', size)
            self.assertAlmostEqual(result['valid_loss'], reference['valid_loss'], places=5)
            self.assertEqual(result['strict_accuracy'], reference['strict_accuracy'])
            for name in KINDS:
                self.assertAlmostEqual(result[f'{name}_presence_loss'], reference[f'{name}_presence_loss'], places=5)
        with torch.inference_mode():
            self.model.eval()
            self.assertAlmostEqual(reference['valid_loss'], float(loss(self.model, rows, self.pad, 'cpu')), places=5)
        absent = [r for r in rows if r['kind'] != 0]
        result = validate(self.model, absent, self.pad, 'cpu', 2)
        self.assertIsNone(result['span_exact_on_present'])
        self.assertAlmostEqual(result['valid_loss'], sum(result[f'{k}_presence_loss'] * result[f'{k}_rows'] for k in ('missing', 'conflict')) / len(absent), places=5)

    def test_validation_applies_the_training_class_weights(self):
        rows = self.rows + [encode(self.tok, rag_prompt(PASSAGES, '담당자는?'), '연정경 [D3]')]
        weights = presence_weights(rows)  # 3 present, 2 missing, 1 conflict -> rising weights, mean 1
        self.assertLess(weights[0], weights[2])
        reference = validate(self.model, rows, self.pad, 'cpu', len(rows), weights)
        plain = validate(self.model, rows, self.pad, 'cpu', len(rows))
        self.assertNotAlmostEqual(reference['valid_loss'], plain['valid_loss'], places=5)
        self.assertEqual(reference['strict_accuracy'], plain['strict_accuracy'])  # weights touch the loss only
        self.assertEqual(reference['macro_strict_accuracy'], macro_strict_accuracy(reference))
        for name, w in zip(KINDS, weights):
            self.assertAlmostEqual(reference[f'{name}_presence_loss'], plain[f'{name}_presence_loss'] * w, places=5)
        for size in (1, 2, 4):
            result = validate(self.model, rows, self.pad, 'cpu', size, weights)
            self.assertAlmostEqual(result['valid_loss'], reference['valid_loss'], places=5)
        with torch.inference_mode():
            self.model.eval()
            self.assertAlmostEqual(reference['valid_loss'], float(loss(self.model, rows, self.pad, 'cpu', weights=weights)), places=5)

    def test_strict_accuracy_is_not_presence_or_span_accuracy(self):
        # Scripted outputs: presence always right, span text right for one present row only.
        rows = self.rows
        outs = {0: point(rows[0], rows[0]['start'], rows[0]['end'] + 1, 0), 3: point(rows[3], rows[0]['start'], rows[0]['end'] + 1, 0)}
        for i in (1, 2, 4):
            outs[i] = point(rows[i], rows[0]['start'], rows[0]['end'] + 1, rows[i]['kind'])

        class Scripted(torch.nn.Module):
            def forward(self, tokens, *rest):
                n = tokens.shape[0]; width = max(len(r['mask']) for r in rows)
                kind = torch.stack([outs[i][0] for i in range(n)])
                scores = torch.full((n, 2, width), -50.)
                for i in range(n):
                    scores[i, :, :outs[i][1].shape[1]] = outs[i][1]
                return kind, scores
        result = validate(Scripted(), rows, self.pad, 'cpu', 5)
        self.assertEqual(result['presence_accuracy'], 1.)
        self.assertEqual(result['strict_accuracy'], 4 / 5)
        self.assertEqual(result['present_strict_accuracy'], .5)
        self.assertEqual(result['missing_strict_accuracy'], 1.); self.assertEqual(result['conflict_strict_accuracy'], 1.)
        self.assertEqual(result['span_exact_on_present'], .5)


class MicroBatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.prompt = rag_prompt(PASSAGES, QUESTION)
        self.tok = train_tokenizer([{'prompt': self.prompt, 'answer': '500원 [D1]'}], Path(self.tmp.name) / 'tok.json', 300)
        cfg = Config(vocab_size=self.tok.get_vocab_size(), dim=32, layers=1, heads=4, hidden_dim=64, context_length=512)
        torch.manual_seed(0); self.model = ExtractiveReader(cfg)
        self.pad = self.tok.token_to_id('<|pad|>')
        longer = rag_prompt(PASSAGES, '담당자는 누구입니까?')
        # micro-batches of 2: [present, present | missing, conflict | present(longer prompt)] -> mixed presence,
        # an all-absent chunk, unequal chunk sizes and independently padded lengths
        self.draws = [encode(self.tok, self.prompt, '500원 [D1]'), encode(self.tok, self.prompt, '연정경 [D3]'),
                      encode(self.tok, self.prompt, '확인 불가'), encode(self.tok, self.prompt, '근거 충돌'),
                      encode(self.tok, longer, '연정경 [D3]')]
        self.assertNotEqual(len(self.draws[0]['ids']), len(self.draws[-1]['ids']))

    def reference(self, draws):
        self.model.zero_grad(set_to_none=True)
        total = loss(self.model, draws, self.pad, 'cpu')
        total.backward()
        # None for parameters the objective never touches (span head on an all-absent batch); kept, not skipped
        grads = {name: None if p.grad is None else p.grad.clone() for name, p in self.model.named_parameters()}
        return float(total.detach()), grads

    def assert_matches_unsplit(self, draws, micro):
        expected, grads = self.reference(draws)
        self.model.zero_grad(set_to_none=True)
        got = train_step(self.model, draws, self.pad, 'cpu', micro)
        self.assertAlmostEqual(got, expected, places=5)
        for name, p in self.model.named_parameters():
            if grads[name] is None:
                self.assertIsNone(p.grad, name)  # both paths leave the same parameters untouched
            else:
                self.assertIsNotNone(p.grad, name)
                torch.testing.assert_close(p.grad, grads[name], atol=1e-5, rtol=1e-4, msg=name)

    def test_accumulated_gradient_equals_the_unsplit_batch(self):
        for micro in (1, 2, 3, 5, 8):
            self.assert_matches_unsplit(self.draws, micro)

    def test_all_absent_batch(self):
        absent = [r for r in self.draws if r['kind'] != 0]
        self.assert_matches_unsplit(absent, 1)
        self.assert_matches_unsplit(absent, 2)

    def test_one_optimizer_update_per_step(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        steps = []
        original = opt.step
        opt.step = lambda *args, **kw: steps.append(1) or original(*args, **kw)
        before = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        opt.zero_grad(set_to_none=True)
        train_step(self.model, self.draws, self.pad, 'cpu', 2)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.); opt.step()
        self.assertEqual(len(steps), 1)
        self.assertTrue(any(not torch.equal(before[n], p.detach()) for n, p in self.model.named_parameters()))

    def test_weighted_accumulation_matches_the_unsplit_weighted_batch(self):
        weights = [.4, 1.1, 1.5]
        self.model.zero_grad(set_to_none=True)
        total = loss(self.model, self.draws, self.pad, 'cpu', weights=weights); total.backward()
        grads = {name: p.grad.clone() for name, p in self.model.named_parameters() if p.grad is not None}
        for micro in (1, 2, 3):
            self.model.zero_grad(set_to_none=True)
            got = train_step(self.model, self.draws, self.pad, 'cpu', micro, weights)
            self.assertAlmostEqual(got, float(total.detach()), places=5)
            for name, p in self.model.named_parameters():
                if name in grads:
                    torch.testing.assert_close(p.grad, grads[name], atol=1e-5, rtol=1e-4, msg=name)
        self.assertNotAlmostEqual(float(total.detach()), float(loss(self.model, self.draws, self.pad, 'cpu').detach()), places=5)


class RowAccountingTests(unittest.TestCase):
    def test_skipped_reasons_and_counts_survive_subsampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp); prompt = rag_prompt(PASSAGES, QUESTION)
            tok = train_tokenizer([{'prompt': prompt, 'answer': '500원 [D1]'}], data / 'tok.json', 300)
            rows = [{'prompt': prompt, 'answer': '500원 [D1]', 'task': 'rag_lookup', 'group': 'rag'},
                    {'prompt': prompt, 'answer': '연정경 [D3]', 'task': 'rag_lookup', 'group': 'rag_real'},
                    {'prompt': prompt, 'answer': '확인 불가', 'task': 'rag_missing', 'group': 'rag'},
                    {'prompt': prompt, 'answer': '근거 충돌', 'task': 'rag_conflict', 'group': 'rag'},
                    {'prompt': prompt, 'answer': '00원 [D1]', 'task': 'rag_lookup', 'group': 'rag'},  # starts inside a word
                    {'prompt': prompt, 'answer': '500원 [D1], 500원 [D2]', 'task': 'rag_combine', 'group': 'rag'},
                    {'prompt': prompt, 'answer': '없는 값 [D1]', 'task': 'rag_lookup', 'group': 'rag'},
                    {'prompt': prompt, 'answer': '500원 [D1]', 'task': 'rag_lookup', 'group': 'dialogue'},
                    {'prompt': prompt, 'answer': '500원 [D1]', 'task': 'rag_lookup', 'group': 'rag', 'history': [['a', 'b']]}]
            (data / 'valid.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
            got, skipped, counts = load_rows(tok, data, 'valid', 512, 0, 1)
            self.assertEqual(len(got), 4)
            self.assertEqual(counts['usable'], {'present': 2, 'missing': 1, 'conflict': 1})
            self.assertEqual(counts['sampled'], counts['usable'])
            self.assertEqual(skipped['combine'], 1); self.assertEqual(skipped['unmapped_answer'], 1)
            self.assertEqual(skipped['other_group'], 2)
            self.assertEqual(sum(v for k, v in skipped.items() if k.startswith('label_')), 1)
            self.assertEqual(sum(skipped.values()) + len(got), len(rows))
            got, skipped2, counts = load_rows(tok, data, 'valid', 512, 2, 1)
            self.assertEqual(len(got), 2); self.assertEqual(skipped2, skipped)
            self.assertEqual(counts['usable'], {'present': 2, 'missing': 1, 'conflict': 1})
            self.assertEqual(sum(counts['sampled'].values()), 2)
            with self.assertRaises(ValueError):
                load_rows(tok, data, 'valid', 8, 0, 1)  # every row over the context


class TinyTrainingSmokeTests(unittest.TestCase):
    def run_tiny(self, root, extra):
        """3-row data, 1-layer base; returns (meta, metrics, output dir) of a CPU run with `extra` CLI options."""
        data = root / 'data'; data.mkdir(); base = root / 'base'; base.mkdir()
        prompt = rag_prompt(PASSAGES, QUESTION)
        tok = train_tokenizer([{'prompt': prompt, 'answer': '500원 [D1]'}], base / 'tokenizer.json', 300)
        cfg = Config(vocab_size=tok.get_vocab_size(), dim=32, layers=1, heads=4, hidden_dim=64, context_length=256)
        torch.manual_seed(0); trid = Trid(cfg)
        save_file({k: v.contiguous() for k, v in trid.state_dict().items()}, str(base / 'model.safetensors'))
        write_json(base / 'config.json', asdict(cfg))
        rows = [{'prompt': prompt, 'answer': a, 'task': t, 'group': 'rag'} for a, t in
                (('500원 [D1]', 'rag_lookup'), ('연정경 [D3]', 'rag_lookup'), ('확인 불가', 'rag_missing'), ('근거 충돌', 'rag_conflict'))]
        for split in ('train', 'valid'):
            (data / f'{split}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        out = root / 'out'
        argv = ['train_extractive', '--init-run', str(base), '--data', str(data), '--output', str(out),
                '--batch-size', '2', '--eval-every', '1', '--context', '256', '--threads', '2', '--lr', '0.001', *extra]
        old = sys.argv; sys.argv = argv
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                main()
        finally:
            sys.argv = old
        return json.loads((out / 'training.json').read_text()), json.loads((out / 'metrics.json').read_text()), out

    def test_cpu_run_records_the_checkpoint_the_policy_selects(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta, metrics, out = self.run_tiny(Path(tmp), ['--steps', '3', '--head', 'v1'])
            self.assertEqual(meta['status'], 'complete'); self.assertEqual(meta['completed_steps'], 3)
            self.assertEqual((meta['reader_head'], meta['reader_attention']), ('v1', 'causal'))
            self.assertEqual(meta['selection_metric'], 'macro_strict_accuracy'); self.assertEqual(meta['checkpoint_selection'], SELECTION)
            self.assertEqual(meta['micro_batch_size'], 2)  # default 4 capped at --batch-size 2
            self.assertEqual(meta['loss_normalization'], NORMALIZATION)
            weights = meta['presence_class_weights']
            self.assertEqual(list(weights), list(KINDS)); self.assertAlmostEqual(sum(weights.values()) / 3, 1.)
            self.assertAlmostEqual(weights['present'] * 2, weights['missing']); self.assertEqual(weights['missing'], weights['conflict'])
            self.assertTrue(all(math.isfinite(m['train_loss']) for m in metrics[1:]))
            self.assertEqual([m['step'] for m in metrics], [0, 1, 2, 3])
            for m in metrics:
                for key in ('valid_loss', 'strict_accuracy', 'macro_strict_accuracy', 'present_strict_accuracy',
                            'missing_strict_accuracy', 'conflict_strict_accuracy', 'present_presence_loss',
                            'missing_presence_loss', 'conflict_presence_loss'):
                    self.assertIn(key, m)
                self.assertAlmostEqual(m['macro_strict_accuracy'], macro_strict_accuracy(m))
            # the recorded best is exactly what the policy picks over all evaluations, step 0 included
            best = None
            for m in metrics:
                if better(m, best):
                    best = m
            self.assertEqual(meta['best_step'], best['step'])
            self.assertEqual(meta['best_macro_strict_accuracy'], best['macro_strict_accuracy'])
            self.assertEqual(meta['best_strict_accuracy'], best['strict_accuracy'])
            self.assertEqual(meta['best_validation_loss'], best['valid_loss'])
            self.assertEqual(meta['best_validation']['macro_strict_accuracy'], best['macro_strict_accuracy'])
            for split in ('train', 'valid'):
                self.assertEqual(meta['counts'][split], {'usable': {'present': 2, 'missing': 1, 'conflict': 1},
                                                         'sampled': {'present': 2, 'missing': 1, 'conflict': 1}})
                self.assertEqual(meta['skipped_rows'][split], {})
            self.assertTrue((out / WEIGHTS).exists())  # the selected checkpoint is what evaluate_extractive loads
            model, tok, device = load_reader(out, 'cpu')
            self.assertEqual((model.head, model.attention), ('v1', 'causal'))

    def test_v2_bidirectional_run_is_the_default_head_and_loads_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta, metrics, out = self.run_tiny(Path(tmp), ['--steps', '2', '--attention', 'bidirectional'])
            self.assertEqual(meta['status'], 'complete'); self.assertEqual(meta['completed_steps'], 2)
            self.assertEqual((meta['reader_head'], meta['reader_attention']), ('v2', 'bidirectional'))  # --head default
            self.assertEqual([m['step'] for m in metrics], [0, 1, 2])
            self.assertTrue(all(math.isfinite(m['train_loss']) for m in metrics[1:]))
            model, tok, device = load_reader(out, 'cpu')
            self.assertEqual((model.head, model.attention), ('v2', 'bidirectional'))
            self.assertTrue(hasattr(model, 'key_mlp') and not hasattr(model, 'kind'))
            prompt = rag_prompt(PASSAGES, QUESTION)
            row = encode(tok, prompt, '500원 [D1]', 256)
            with torch.inference_mode():
                kind, scores = model(*batch([row], tok.token_to_id('<|pad|>'), 'cpu'))
            out_row = decode((kind[0], scores[0]), row)
            self.assertIn(out_row['presence'], (0, 1, 2))


if __name__ == '__main__':
    unittest.main()
