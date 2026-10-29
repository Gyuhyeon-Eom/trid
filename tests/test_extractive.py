import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
import torch
from safetensors.torch import save_file
from trid.model import Config
from trid.reader import Reader
from trid.text import train_tokenizer
from trid.prompts import rag_prompt, RAG_PREFIX
from trid.extractive import (ExtractiveReader, encode, batch, loss, loss_terms, decode, best_span, parse_answer, span_issue,
                             span_evidence, load_reader, read, MAX_SPAN, STAGE, WEIGHTS, ATTENTION, HEADS, NEIGHBOURS)

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

    def test_best_span_and_label_placement_share_span_issue(self):
        row = encode(self.tok, self.prompt)
        n = len(row['mask'])
        rng = torch.Generator().manual_seed(7)
        for _ in range(150):
            start = int(torch.randint(0, n, (1,), generator=rng)); end = start + 1 + int(torch.randint(0, MAX_SPAN + 2, (1,), generator=rng))
            scores = torch.full((2, n), -50.)
            if end <= n:
                scores[0, start] = 100; scores[1, end - 1] = 100
            best = best_span(scores, row)
            self.assertIsNotNone(best)
            self.assertEqual(best[:2] == (start, end), span_issue(row, start, end) is None, (start, end))
            self.assertIsNone(span_issue(row, best[0], best[1]))
        # the same rule words the rejection reasons
        self.assertEqual(span_issue(row, 0, 1), 'outside_passage')
        d2 = row['passages'][0][1]; d5 = row['passages'][1][1]
        self.assertEqual(span_issue(row, d2, d5 + 1), 'crosses_passage')
        self.assertEqual(span_issue(row, d2 + 1, d2 + 2), 'start_inside_word')
        phone = row['prompt'].index('1675-3561')
        self.assertEqual(span_issue(row, phone, phone + 2), 'end_splits_run')
        self.assertIsNone(span_issue(row, phone, phone + 4))  # '1675' ends before '-'

    def test_bracket_characters_are_never_emitted(self):
        # the service grammar (parse_answer / PAIR) rejects '[' or ']' inside a value, so decode must never produce one
        prompt = rag_prompt([('D1', '참고 [별표 1] 기준 30일'), ('D2', '담당자 연정경')], '기준은?')
        row = encode(self.tok, prompt)
        for pos, ch in enumerate(prompt):
            if ch in '[]':
                self.assertFalse(row['mask'][pos])
        n = len(row['mask'])
        a, b = prompt.index('[별표'), prompt.index('1] 기준') + 1
        for start, end in ((a, b), (a, b + 3), (a - 3, a + 1), (a + 1, b + 3)):  # spans containing a bracket
            self.assertIsNotNone(span_issue(row, start, end + 1))
            scores = torch.full((2, n), -50.); scores[0, start] = 100; scores[1, end] = 100
            out = decode((torch.tensor([10., -10., -10.]), scores), row)
            self.assertNotIn('[', out['span_text']); self.assertNotIn(']', out['span_text'])
            self.assertEqual(parse_answer(out['answer']), (0, out['span_text'], out['source']))
        self.assertEqual(span_issue(row, prompt.index('기준'), prompt.index('30일') + 3), None)
        self.assertIsNone(encode(self.tok, prompt, '[별표 1] [D1]'))

    def test_no_admissible_span_decodes_to_abstention(self):
        prompt = rag_prompt([('D1', 'A' * (MAX_SPAN + 6))], '코드는?')  # one latin run longer than MAX_SPAN
        row = encode(self.tok, prompt)
        n = len(row['mask'])
        self.assertIsNone(best_span(torch.randn(2, n), row))
        out = decode((torch.tensor([10., -10., -10.]), torch.randn(2, n)), row)
        self.assertEqual((out['answer'], out['presence'], out['presence_label'], out['reason']), ('확인 불가', 1, '확인 불가', 'no_valid_span'))
        self.assertEqual((out['span'], out['span_text'], out['source'], out['span_score']), (None, None, None, None))
        self.assertEqual(parse_answer(out['answer']), (1, None, None))
        self.assertAlmostEqual(sum(out['presence_probs']), 1., places=5)
        self.assertIsNone(decode((torch.tensor([10., -10., -10.]), torch.randn(2, n)), encode(self.tok, self.prompt))['reason'])
        self.assertEqual(encode(self.tok, prompt, 'A' * 10 + ' [D1]')['label_reason'], 'end_splits_run')

    def test_encode_picks_a_decodable_occurrence_or_names_the_failed_rule(self):
        prompt = rag_prompt([('D1', '기한 114일 아님. 실제 기한 14일 이내'), ('D2', '전화 1675-3561 입니다')], '기한은?')
        row = encode(self.tok, prompt, '14일 [D1]')
        self.assertTrue(row['label_decodable']); self.assertIsNone(row['label_reason'])
        self.assertEqual(row['start'], prompt.index(' 14일') + 1)  # second occurrence: the first sits inside '114일'
        self.assertIsNone(span_issue(row, row['start'], row['end'] + 1))
        row = encode(self.tok, prompt, '일 아님 [D1]')
        self.assertEqual((row['label_decodable'], row['label_reason']), (False, 'start_inside_word'))
        self.assertEqual(prompt[row['start']:row['end'] + 1], '일 아님')
        row = encode(self.tok, prompt, '16 [D2]')
        self.assertEqual((row['label_decodable'], row['label_reason']), (False, 'end_splits_run'))
        self.assertEqual((row['kind'], prompt[row['start']:row['end'] + 1]), (0, '16'))
        for answer in ('확인 불가', '근거 충돌'):
            self.assertNotIn('label_reason', encode(self.tok, prompt, answer))

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
        with torch.no_grad():
            outputs = self.model(*batch(rows, self.pad, 'cpu'))
            torch.testing.assert_close(loss(self.model, rows, self.pad, 'cpu', outputs=outputs), loss(self.model, rows, self.pad, 'cpu'))

    def test_loss_terms_sum_over_chunks_to_the_whole_loss(self):
        rows = [encode(self.tok, self.prompt, '14일 전까지 [D2]'), encode(self.tok, self.prompt, '확인 불가'),
                encode(self.tok, self.prompt, '근거 충돌'), encode(self.tok, self.prompt, '40% [D5]')]
        with torch.no_grad():
            presence, span, present = loss_terms(self.model, rows, self.pad, 'cpu')
            self.assertEqual((tuple(presence.shape), span.shape, present), ((4,), torch.Size([]), 2))
            torch.testing.assert_close(presence.mean() + span / present, loss(self.model, rows, self.pad, 'cpu'))
            # unequal chunks [present | missing, conflict | present]: one all-absent chunk contributes no span term
            chunks = [rows[:1], rows[1:3], rows[3:]]
            terms = [loss_terms(self.model, c, self.pad, 'cpu') for c in chunks]
            self.assertEqual(float(terms[1][1]), 0.); self.assertEqual(terms[1][2], 0)
            total = sum(t[0].sum() for t in terms) / len(rows) + sum(t[1] for t in terms) / sum(t[2] for t in terms)
            torch.testing.assert_close(total, loss(self.model, rows, self.pad, 'cpu'))
            absent_presence, _, _ = loss_terms(self.model, rows[1:3], self.pad, 'cpu')
            torch.testing.assert_close(absent_presence.mean(), loss(self.model, rows[1:3], self.pad, 'cpu'))

    def test_presence_weights_scale_rows_and_keep_chunk_invariance(self):
        rows = [encode(self.tok, self.prompt, '14일 전까지 [D2]'), encode(self.tok, self.prompt, '확인 불가'),
                encode(self.tok, self.prompt, '근거 충돌'), encode(self.tok, self.prompt, '40% [D5]')]
        weights = [.5, 2., 3.]
        with torch.no_grad():
            plain, span, present = loss_terms(self.model, rows, self.pad, 'cpu')
            weighted, span_w, present_w = loss_terms(self.model, rows, self.pad, 'cpu', weights=weights)
            torch.testing.assert_close(weighted, plain * torch.tensor([.5, 2., 3., .5]))  # each row by its kind's weight
            torch.testing.assert_close(span_w, span); self.assertEqual(present_w, present)  # span terms untouched
            torch.testing.assert_close(loss(self.model, rows, self.pad, 'cpu', weights=weights),
                                       weighted.mean() + span / present)
            self.assertFalse(torch.equal(loss(self.model, rows, self.pad, 'cpu', weights=weights), loss(self.model, rows, self.pad, 'cpu')))
            terms = [loss_terms(self.model, c, self.pad, 'cpu', weights=torch.tensor(weights)) for c in (rows[:1], rows[1:3], rows[3:])]
            total = sum(t[0].sum() for t in terms) / len(rows) + sum(t[1] for t in terms) / sum(t[2] for t in terms)
            torch.testing.assert_close(total, loss(self.model, rows, self.pad, 'cpu', weights=weights))
            torch.testing.assert_close(loss(self.model, rows, self.pad, 'cpu', weights=[1., 1., 1.]), loss(self.model, rows, self.pad, 'cpu'))

    def test_v1_head_is_the_reader_layout_and_arithmetic(self):
        self.assertEqual(HEADS, ('v1', 'v2'))
        self.assertEqual(self.model.head, 'v1')
        reference = Reader(self.cfg)
        self.assertEqual([(k, tuple(v.shape)) for k, v in self.model.state_dict().items()],
                         [(k, tuple(v.shape)) for k, v in reference.state_dict().items()])
        rows = self.padded_rows()
        tokens, lengths, char_tokens, char_ids, char_offsets, mask, prev_ids, next_ids = batch(rows, self.pad, 'cpu')
        self.assertEqual(prev_ids.shape, mask.shape)
        m = self.model.eval()
        with torch.no_grad():
            h = m.backbone(tokens, hidden_only=True)
            last = h[torch.arange(len(rows)), lengths - 1]
            q = m.query(last).reshape(-1, 2, 64)
            source = h.gather(1, char_tokens[:, :, None].expand(-1, -1, h.shape[-1]))
            k = (m.key(source) + m.character_key(m.character(char_ids) + m.within_token(char_offsets))).reshape(len(rows), -1, 2, 64)
            expected = (torch.einsum('bsd,btsd->bst', q, k) / 8).masked_fill(~mask[:, None, :], -1e9)
            kind, scores = m(tokens, lengths, char_tokens, char_ids, char_offsets, mask)  # neighbours optional for v1
            with_neighbours = m(*batch(rows, self.pad, 'cpu'))
            perturbed = m(tokens, lengths, char_tokens, char_ids, char_offsets, mask, (prev_ids + 7) % NEIGHBOURS, next_ids)
        self.assertTrue(torch.equal(kind, m.kind(last)))
        self.assertTrue(torch.equal(scores, expected))
        self.assertTrue(torch.equal(kind, with_neighbours[0]) and torch.equal(scores, with_neighbours[1]))
        self.assertTrue(torch.equal(scores, perturbed[1]))  # v1 never reads the neighbour ids

    def test_encode_neighbour_ids_follow_the_mask(self):
        row = encode(self.tok, self.prompt, '14일 전까지 [D2]')
        n = len(row['mask'])
        self.assertEqual((len(row['prev_ids']), len(row['next_ids'])), (n, n))
        for pos in range(n):
            if not row['mask'][pos]:
                self.assertEqual((row['prev_ids'][pos], row['next_ids'][pos]), (0, 0))
                continue
            self.assertEqual(row['prev_ids'][pos], row['char_ids'][pos - 1] + 1 if pos and row['mask'][pos - 1] else 0)
            self.assertEqual(row['next_ids'][pos], row['char_ids'][pos + 1] + 1 if pos + 1 < n and row['mask'][pos + 1] else 0)
        self.assertLess(max(row['prev_ids'] + row['next_ids']), NEIGHBOURS)
        # '14일 전까지': the end of '14일' sees the following space as 0, the start of '전까지' sees the space as 0
        end_of_word = row['start'] + 2
        self.assertEqual(self.prompt[end_of_word], '일'); self.assertEqual(row['next_ids'][end_of_word], 0)
        self.assertEqual(row['prev_ids'][end_of_word + 2], 0)

    def test_v2_head_shapes_and_neighbour_sensitivity(self):
        rows = self.padded_rows()
        torch.manual_seed(1); model = ExtractiveReader(self.cfg, 'bidirectional', 'v2').eval()
        self.assertEqual(model.head, 'v2')
        for name in ('key', 'kind', 'character_key'):
            self.assertFalse(hasattr(model, name))
        for name in ('previous', 'following', 'key_mlp', 'presence', 'query', 'character', 'within_token'):
            self.assertTrue(hasattr(model, name))
        tensors = batch(rows, self.pad, 'cpu')
        tokens, lengths, char_tokens, char_ids, char_offsets, mask, prev_ids, next_ids = tensors
        with torch.no_grad():
            kind, scores = model(*tensors)
            self.assertEqual(tuple(kind.shape), (2, 3)); self.assertEqual(tuple(scores.shape), (2, 2, mask.shape[1]))
            self.assertTrue(torch.all(scores[~mask[:, None, :].expand(-1, 2, -1)] == -1e9))
            # the end score of a character depends on the id of the next character; other characters are untouched
            row = rows[0]; pos = row['start'] + 2
            self.assertTrue(row['mask'][pos] and row['mask'][pos + 2])
            changed = next_ids.clone(); changed[0, pos] = (int(next_ids[0, pos]) + 1) % NEIGHBOURS
            kind2, scores2 = model(tokens, lengths, char_tokens, char_ids, char_offsets, mask, prev_ids, changed)
            self.assertNotEqual(float(scores[0, 1, pos]), float(scores2[0, 1, pos]))
            self.assertNotEqual(float(scores[0, 0, pos]), float(scores2[0, 0, pos]))
            keep = torch.ones_like(mask); keep[0, pos] = False
            self.assertTrue(torch.equal(scores[keep[:, None, :].expand(-1, 2, -1)], scores2[keep[:, None, :].expand(-1, 2, -1)]))
            changed = prev_ids.clone(); changed[0, pos] = (int(prev_ids[0, pos]) + 1) % NEIGHBOURS
            scores3 = model(tokens, lengths, char_tokens, char_ids, char_offsets, mask, changed, next_ids)[1]
            self.assertNotEqual(float(scores[0, 0, pos]), float(scores3[0, 0, pos]))
            # presence sees the span evidence: the logits move with the scores while the hidden states stay put
            evidence = span_evidence(scores)
            self.assertEqual(tuple(evidence.shape), (2, 3))
            for i, r in enumerate(rows):
                best = best_span(scores[i], r)
                self.assertGreaterEqual(float(evidence[i, 0]) + 1e-4, best[2])  # mask-window max >= rule-admissible max
                torch.testing.assert_close(evidence[i, 1], scores[i, 0].logsumexp(-1))
            h = model.backbone(tokens, hidden_only=True, attention='bidirectional', lengths=lengths)
            last = h[torch.arange(2), lengths - 1]
            torch.testing.assert_close(kind, model.presence(torch.cat([last, evidence], -1)))
            self.assertFalse(torch.equal(kind, kind2))  # the perturbed scores reach the presence logits
        with self.assertRaises(ValueError):
            ExtractiveReader(self.cfg, 'causal', 'v3')
        # the evidence is detached: the span scores receive no gradient through the presence CE
        row = encode(self.tok, self.prompt, '확인 불가')
        model.train(); model.zero_grad()
        presence, span, present = loss_terms(model, [row], self.pad, 'cpu')
        presence.sum().backward()
        self.assertIsNone(model.key_mlp[0].weight.grad); self.assertIsNone(model.previous.weight.grad)
        self.assertIsNotNone(model.presence[0].weight.grad); self.assertIsNotNone(model.backbone.norm.weight.grad)

    def test_span_evidence_window_is_max_span(self):
        scores = torch.full((1, 2, 200), -1e9)
        scores[0, 0, 10] = 1.; scores[0, 1, 10 + MAX_SPAN - 1] = 2.; scores[0, 1, 10 + MAX_SPAN] = 5.; scores[0, 1, 5] = 9.
        best, lse_start, lse_end = span_evidence(scores)[0].tolist()
        self.assertAlmostEqual(best, 3., places=5)  # end at start+63 counts, start+64 and end<start do not
        self.assertAlmostEqual(lse_start, 1., places=5); self.assertAlmostEqual(lse_end, torch.tensor([2., 5., 9.]).logsumexp(0).item(), places=5)

    def test_load_reader_restores_head_and_read_reports_it(self):
        root = Path(self.tmp.name) / 'v2'; root.mkdir()
        (root / 'config.json').write_text(json.dumps(asdict(self.cfg)))
        self.tok.save(str(root / 'tokenizer.json'))
        torch.manual_seed(2); trained = ExtractiveReader(self.cfg, 'bidirectional', 'v2')
        save_file({k: v.contiguous() for k, v in trained.state_dict().items()}, str(root / WEIGHTS))
        (root / 'training.json').write_text(json.dumps({'stage': STAGE, 'reader_attention': 'bidirectional', 'reader_head': 'v2'}))
        model, tok, device = load_reader(root, 'cpu')
        self.assertEqual((model.head, model.attention), ('v2', 'bidirectional'))
        row = encode(self.tok, self.prompt)
        with torch.no_grad():
            a = trained.eval()(*batch([row], self.pad, 'cpu')); b = model(*batch([row], self.pad, 'cpu'))
        torch.testing.assert_close(a[0], b[0]); torch.testing.assert_close(a[1], b[1])
        result = read(model, tok, self.prompt, device)
        self.assertEqual((result['head'], result['attention']), ('v2', 'bidirectional'))
        self.assertEqual(read(self.model.eval(), self.tok, self.prompt, 'cpu')['head'], 'v1')
        # the head is metadata: absent means v1, and v1 cannot load v2 weights (no silent reinterpretation)
        (root / 'training.json').write_text(json.dumps({'stage': STAGE, 'reader_attention': 'bidirectional'}))
        with self.assertRaises(RuntimeError):
            load_reader(root, 'cpu')
        (root / 'training.json').write_text(json.dumps({'stage': STAGE, 'reader_head': 'v9'}))
        with self.assertRaises(ValueError):
            load_reader(root, 'cpu')

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

    def padded_rows(self):
        """Two rows of different token length so batch() right-pads the shorter one."""
        longer = rag_prompt(PASSAGES, '담당자는 누구이며 전화번호는 무엇입니까?')
        return [encode(self.tok, self.prompt, '14일 전까지 [D2]'), encode(self.tok, longer, '연정경 [D2]')]

    def test_bidirectional_padded_batch_matches_unpadded_rows(self):
        rows = self.padded_rows()
        model = ExtractiveReader(self.cfg, 'bidirectional').eval()
        model.load_state_dict(self.model.state_dict())
        tokens, lengths, *_ = batch(rows, self.pad, 'cpu')
        self.assertNotEqual(len(rows[0]['ids']), len(rows[1]['ids']))
        self.assertEqual(tokens.shape[1], int(lengths.max()))
        with torch.no_grad():
            padded = model.backbone(tokens, hidden_only=True, attention='bidirectional', lengths=lengths)
            kind, scores = model(*batch(rows, self.pad, 'cpu'))
            for i, row in enumerate(rows):
                n = len(row['ids'])
                alone = model.backbone(tokens[i:i + 1, :n], hidden_only=True, attention='bidirectional')
                torch.testing.assert_close(padded[i, :n], alone[0], atol=1e-5, rtol=1e-5)
                kind_i, scores_i = model(*batch([row], self.pad, 'cpu'))
                torch.testing.assert_close(kind[i], kind_i[0], atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(scores[i, :, :scores_i.shape[-1]], scores_i[0], atol=1e-5, rtol=1e-5)

    def test_causal_mode_is_unchanged_by_the_attention_argument(self):
        rows = self.padded_rows()
        tokens, lengths, *rest = batch(rows, self.pad, 'cpu')
        self.model.eval()
        with torch.no_grad():
            before = self.model.backbone(tokens, hidden_only=True)
            explicit = self.model.backbone(tokens, hidden_only=True, attention='causal', lengths=lengths)
            reader = self.model(tokens, lengths, *rest)
        self.assertTrue(torch.equal(before, explicit))
        self.assertEqual(self.model.attention, 'causal')
        last = before[torch.arange(len(rows)), lengths - 1]
        self.assertTrue(torch.equal(reader[0], self.model.kind(last)))

    def test_bidirectional_lets_early_tokens_see_late_tokens_and_causal_does_not(self):
        row = self.padded_rows()[0]
        tokens = batch([row], self.pad, 'cpu')[0]
        changed = tokens.clone(); changed[0, -1] = (int(tokens[0, -1]) + 1) % self.cfg.vocab_size
        backbone = self.model.backbone.eval()
        with torch.no_grad():
            for mode, differs in (('causal', False), ('bidirectional', True)):
                a = backbone(tokens, hidden_only=True, attention=mode)[0, :-1]
                b = backbone(changed, hidden_only=True, attention=mode)[0, :-1]
                self.assertEqual(not torch.equal(a, b), differs, mode)

    def test_bidirectional_rejects_cache_and_unknown_modes(self):
        tokens = batch([self.padded_rows()[0]], self.pad, 'cpu')[0]
        with self.assertRaises(ValueError):
            self.model.backbone(tokens, use_cache=True, attention='bidirectional')
        with torch.no_grad():
            _, cache = self.model.backbone(tokens[:, :4], use_cache=True)
        with self.assertRaises(ValueError):
            self.model.backbone(tokens[:, 4:6], cache=cache, attention='bidirectional')
        with self.assertRaises(ValueError):
            self.model.backbone(tokens, attention='full')
        with self.assertRaises(ValueError):
            ExtractiveReader(self.cfg, 'full')

    def test_load_reader_restores_attention_mode(self):
        self.assertEqual(ATTENTION, ('causal', 'bidirectional'))
        root = Path(self.tmp.name) / 'bidir'; root.mkdir()
        (root / 'config.json').write_text(json.dumps(asdict(self.cfg)))
        self.tok.save(str(root / 'tokenizer.json'))
        trained = ExtractiveReader(self.cfg, 'bidirectional')
        save_file({k: v.contiguous() for k, v in trained.state_dict().items()}, str(root / WEIGHTS))
        self.assertEqual(set(trained.state_dict()), set(self.model.state_dict()))
        (root / 'training.json').write_text(json.dumps({'stage': STAGE, 'reader_attention': 'bidirectional'}))
        model, tok, device = load_reader(root, 'cpu')
        self.assertEqual(model.attention, 'bidirectional')
        row = encode(self.tok, self.prompt)
        with torch.no_grad():
            a = trained.eval()(*batch([row], self.pad, 'cpu')); b = model(*batch([row], self.pad, 'cpu'))
            causal = ExtractiveReader(self.cfg).eval(); causal.load_state_dict(trained.state_dict())
            c = causal(*batch([row], self.pad, 'cpu'))
        torch.testing.assert_close(a[1], b[1])
        self.assertFalse(torch.equal(a[1], c[1]))
        self.assertEqual(read(model, tok, self.prompt, device)['attention'], 'bidirectional')
        (root / 'training.json').write_text(json.dumps({'stage': STAGE}))
        self.assertEqual(load_reader(root, 'cpu')[0].attention, 'causal')
        (root / 'training.json').write_text(json.dumps({'stage': STAGE, 'reader_attention': 'full'}))
        with self.assertRaises(ValueError):
            load_reader(root, 'cpu')


if __name__ == '__main__':
    unittest.main()
