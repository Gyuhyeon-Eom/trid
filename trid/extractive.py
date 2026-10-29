"""Extractive answer head on the RAG prompt (reports/closed-network-architecture-survey.md §7, §8 experiment 1).

Input is `trid.prompts.rag_prompt(passages, question)` unchanged. The trid backbone (all weights trainable)
feeds two heads: 3-way presence (답 있음 / 확인 불가 / 근거 충돌) and character-level start/end scores over passage
characters. Two head layouts exist, named by training.json 'reader_head' (HEADS; absent means 'v1'):
  v1  trid.korean_reader layout: presence from the last hidden state alone; key = linear(token hidden) +
      linear(char emb + within-token emb), so the start/end score is linear in per-character features and the
      boundary choice inside a token cannot depend on the neighbouring characters.
  v2  key = MLP(concat(token hidden, char emb + within-token emb + previous-char emb + next-char emb)); presence
      from concat(last hidden state, detached span evidence: best mask-admissible start+end score within MAX_SPAN,
      logsumexp of start scores, logsumexp of end scores) through a small MLP, so the presence decision can see how
      confident the span head is.
`[Dn] ` markers, whitespace, bracket characters and the question are masked out; the source id is derived from
the passage of the chosen span, never generated. Encoding (label placement) and decoding (best_span) share one
span admissibility rule, `span_issue`: same passage, <=MAX_SPAN characters, start/end cuts admitted by
trid.span_boundaries (the rule trid.span_scoring.candidates enumerates with; BOUNDARY_POLICY names its version),
no `[`/`]` inside (the service answer grammar `parse_answer`).
"""
import json
import re
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file
from tokenizers import Tokenizer
from .model import Config, Trid
from .reader import batch as reader_batch
from .runtime import device_for, sync
from .text import prompt_ids
from .character_features import character_id
from .prompts import rag_passages, RAG_PREFIX
from .span_boundaries import BOUNDARY_POLICY, start_ok as cut_start_ok, end_ok as cut_end_ok
from .constrained import ABSTAIN

MAX_SPAN = 64
STAGE = 'extractive_reader_v1'
WEIGHTS = 'reader.safetensors'
ATTENTION = ('causal', 'bidirectional')  # training.json 'reader_attention'; absent (older runs) means causal
HEADS = ('v1', 'v2')  # training.json 'reader_head'; absent (older runs) means v1
NEIGHBOURS = 513  # prev/next character ids: 0 none (outside the mask), 1 + character_id otherwise
PRESENCE = ('present',) + ABSTAIN  # label index: 0 답 있음, 1 확인 불가, 2 근거 충돌
PAIR = re.compile(r'^\s*([^\[\]]+?)\s*\[(D[1-9])\]\s*$')  # exactly one `<value> [Dn]`


class ExtractiveReader(nn.Module):
    """3-way presence head + 2x64 query/key character span head over the backbone's hidden states.
    `attention` 'causal' (passage hidden states never see the question) or 'bidirectional' (full attention inside
    each prompt, padding keys masked); the state_dict layout is identical, the mode lives in training.json.
    `head` 'v1' (trid.reader.Reader's parameter layout and arithmetic, state_dict keys unchanged) or 'v2' (module
    docstring); the layout is named in training.json 'reader_head', never guessed from the weights."""

    def __init__(self, cfg, attention='causal', head='v1'):
        if attention not in ATTENTION:
            raise ValueError(f"attention은 {ATTENTION} 중 하나여야 합니다.")
        if head not in HEADS:
            raise ValueError(f"head는 {HEADS} 중 하나여야 합니다.")
        super().__init__()
        self.attention, self.head = attention, head
        self.backbone = Trid(cfg)
        self.query = nn.Linear(cfg.dim, 128, bias=False)
        if head == 'v1':
            self.key = nn.Linear(cfg.dim, 128, bias=False)
            self.kind = nn.Linear(cfg.dim, 3)
        self.character = nn.Embedding(512, 32)
        self.within_token = nn.Embedding(32, 32)
        if head == 'v1':
            self.character_key = nn.Linear(32, 128, bias=False)
        else:
            self.previous = nn.Embedding(NEIGHBOURS, 32)
            self.following = nn.Embedding(NEIGHBOURS, 32)
            self.key_mlp = nn.Sequential(nn.Linear(cfg.dim + 32, 256), nn.GELU(), nn.Linear(256, 128))
            self.presence = nn.Sequential(nn.Linear(cfg.dim + 3, 64), nn.GELU(), nn.Linear(64, 3))

    def forward(self, tokens, lengths, char_tokens, char_ids, char_offsets, mask, prev_ids=None, next_ids=None):
        """prev_ids/next_ids (batch() always supplies them) are read by the v2 head only."""
        h = self.backbone(tokens, hidden_only=True, attention=self.attention, lengths=lengths)
        last = h[torch.arange(len(h), device=h.device), lengths - 1]
        q = self.query(last).reshape(-1, 2, 64)
        source = h.gather(1, char_tokens[:, :, None].expand(-1, -1, h.shape[-1]))
        features = self.character(char_ids) + self.within_token(char_offsets)
        if self.head == 'v1':
            k = self.key(source) + self.character_key(features)
        else:
            features = features + self.previous(prev_ids) + self.following(next_ids)
            k = self.key_mlp(torch.cat([source, features], -1))
        k = k.reshape(len(h), char_tokens.shape[1], 2, 64)
        scores = torch.einsum('bsd,btsd->bst', q, k) / 8
        scores = scores.masked_fill(~mask[:, None, :], -1e9)
        if self.head == 'v1':
            return self.kind(last), scores
        return self.presence(torch.cat([last, span_evidence(scores).detach()], -1)), scores


def span_evidence(scores):
    """[batch, 3] from masked scores [batch, 2, chars]: best start+end score over start <= end < start + MAX_SPAN
    (passage-length admissibility is best_span()'s job; masked characters cannot win), logsumexp of the start
    scores, logsumexp of the end scores."""
    start, end = scores[:, 0], scores[:, 1]
    window = F.pad(end, (0, MAX_SPAN - 1), value=-1e9).unfold(1, MAX_SPAN, 1).amax(-1)  # [batch, chars]
    return torch.stack([(start + window).amax(-1), start.logsumexp(-1), end.logsumexp(-1)], -1)


def batch(rows, pad, device):
    """trid.reader.batch's tensors (tokens, lengths, char_tokens, char_ids, char_offsets, mask) plus the right-padded
    prev_ids and next_ids of encode()."""
    tensors = reader_batch(rows, pad, device)
    mask = tensors[-1]
    neighbours = [torch.zeros_like(mask, dtype=torch.long) for _ in range(2)]
    for i, r in enumerate(rows):
        for field, tensor in zip(('prev_ids', 'next_ids'), neighbours):
            tensor[i, :len(r[field])] = torch.tensor(r[field], device=device)
    return *tensors, *neighbours


def parse_answer(answer):
    """'확인 불가' | '근거 충돌' -> (presence, None, None); '<value> [Dn]' -> (0, value, sid); else None
    (two pairs, free text, empty value)."""
    answer = answer.strip()
    if answer in ABSTAIN:
        return ABSTAIN.index(answer) + 1, None, None
    m = PAIR.match(answer)
    if not m or not m.group(1).strip():
        return None
    return 0, m.group(1).strip(), m.group(2)


def passage_spans(prompt):
    """[(sid, start, end)] character offsets of each passage text inside the prompt (end exclusive)."""
    passages = rag_passages(prompt)
    out, pos = [], len(RAG_PREFIX)
    for sid, text in passages:
        start = pos + len(f'[{sid}] ')
        end = start + len(text)
        if prompt[start:end] != text:
            raise ValueError('passage offsets do not match the prompt')
        out.append((sid, start, end))
        pos = end + 1
    return out


def span_issue(row, start, end):
    """None when prompt[start:end] is an admissible answer span of `row`, else the first failed rule:
    'outside_passage' | 'crosses_passage' | 'too_long' | 'start_inside_word' | 'end_splits_run' | 'bracket'.
    best_span() enforces exactly these rules vectorised; encode() uses it to place labels."""
    last = end - 1
    if not 0 <= start <= last < len(row['mask']) or row['char_passage'][start] < 0:
        return 'outside_passage'
    if row['char_passage'][start] != row['char_passage'][last]:
        return 'crosses_passage'
    if end - start > MAX_SPAN:
        return 'too_long'
    if not row['start_ok'][start]:
        return 'start_inside_word'
    if not row['end_ok'][last]:
        return 'end_splits_run'
    if row['brackets'][end] - row['brackets'][start]:
        return 'bracket'
    return None


def encode(tok, prompt, answer=None, context=1024, character_encoding='hashed512'):
    """Row for batch()/loss()/decode(). Raises ValueError when the prompt is not a RAG prompt or exceeds
    `context` tokens. With `answer`, returns None when the label cannot be mapped (two pairs, free text, unknown
    source id, value absent from the cited passage, value longer than MAX_SPAN). Otherwise the label is the first
    occurrence of the value inside the cited passage that passes span_issue(); when no occurrence does, it is the
    first occurrence with `label_decodable` False and `label_reason` naming the failed rule (callers decide whether
    to train on such rows; decode() can never emit them)."""
    spans = passage_spans(prompt)
    ids = prompt_ids(tok, prompt)
    if len(ids) > context:
        raise ValueError(f'입력 {len(ids)} 토큰이 문맥 한도 {context}를 초과합니다.')
    encoded = tok.encode(prompt)
    n = len(prompt)
    char_tokens, char_offsets = [0] * n, [0] * n
    for i, (a, b) in enumerate(encoded.offsets):
        for pos in range(a, b):
            char_tokens[pos] = i + 2  # after <|bos|><|user|>
            char_offsets[pos] = min(pos - a, 31)
    mask, char_passage, start_ok, end_ok = [False] * n, [-1] * n, [False] * n, [False] * n
    brackets = [0] * (n + 1)  # prefix count of '[' / ']' characters
    for pos, ch in enumerate(prompt):
        brackets[pos + 1] = brackets[pos] + (ch in '[]')
    for k, (sid, lo, hi) in enumerate(spans):
        for pos in range(lo, hi):
            if prompt[pos].isspace() or prompt[pos] in '[]':
                continue
            mask[pos] = True
            char_passage[pos] = k
            start_ok[pos] = cut_start_ok(prompt, pos, lo, hi)
            end_ok[pos] = cut_end_ok(prompt, pos, lo, hi)
    if not any(mask):
        raise ValueError('읽을 근거가 없습니다.')
    char_ids = [character_id(ch, character_encoding) for ch in prompt]
    # neighbouring passage characters for the v2 key: 0 when the character or its neighbour is outside the mask
    # (a space, a marker, a passage edge, the question), else 1 + the neighbour's id
    prev_ids = [char_ids[pos - 1] + 1 if mask[pos] and pos and mask[pos - 1] else 0 for pos in range(n)]
    next_ids = [char_ids[pos + 1] + 1 if mask[pos] and pos + 1 < n and mask[pos + 1] else 0 for pos in range(n)]
    row = {'ids': ids, 'mask': mask, 'char_tokens': char_tokens, 'char_offsets': char_offsets, 'char_ids': char_ids,
           'prev_ids': prev_ids, 'next_ids': next_ids,
           'char_passage': char_passage, 'start_ok': start_ok, 'end_ok': end_ok, 'brackets': brackets,
           'passages': spans, 'prompt': prompt}
    if answer is not None:
        parsed = parse_answer(answer)
        if parsed is None:
            return None
        row['kind'], value, sid = parsed
        row['start'] = row['end'] = 0
        if row['kind'] == 0:
            found = [(lo, hi) for s, lo, hi in spans if s == sid]
            if not found or len(value) > MAX_SPAN:
                return None
            lo, hi = found[0]
            issues = []
            loc = prompt.find(value, lo, hi)
            while loc >= 0 and loc + len(value) <= hi:
                issues.append((loc, span_issue(row, loc, loc + len(value))))
                if issues[-1][1] is None:
                    break
                loc = prompt.find(value, loc + 1, hi)
            if not issues:
                return None
            loc, reason = issues[-1] if issues[-1][1] is None else issues[0]
            row['start'], row['end'] = loc, loc + len(value) - 1
            row['label_decodable'], row['label_reason'] = reason is None, reason
    return row


def loss_terms(model, rows, pad, device, outputs=None, weights=None):
    """Unnormalised objective components of `rows`: (presence, span, present) where presence is the per-row
    presence CE [rows], span the summed CE(start) + CE(end) over present rows (a 0-dim tensor; 0 without present
    rows) and present the number of present rows. Summing terms over disjoint chunks and dividing once by the
    row and present counts of the union gives exactly loss() of the union, whatever the chunking: train steps
    accumulate gradients over micro-batches and validation sums over its batches with this.
    `weights` (3 floats, PRESENCE order; None = all 1) scale each row's presence CE by the weight of its kind;
    the row count stays the denominator, so weights with mean 1 keep the scale of the unweighted objective.
    `outputs=(kind, scores)` reuses an already computed forward of exactly these rows instead of running one."""
    kind, scores = model(*batch(rows, pad, device)) if outputs is None else outputs
    kinds = torch.tensor([r['kind'] for r in rows], device=device)
    presence = F.cross_entropy(kind, kinds, reduction='none')
    if weights is not None:
        presence = presence * torch.as_tensor(weights, dtype=presence.dtype, device=presence.device)[kinds]
    present = kinds == 0
    span = kind.new_zeros(())
    if present.any():
        for head, field in enumerate(('start', 'end')):
            labels = torch.tensor([r[field] for r in rows], device=device)
            span = span + F.cross_entropy(scores[present, head], labels[present], reduction='sum')
    return presence, span, int(present.sum())


def loss(model, rows, pad, device, outputs=None, weights=None):
    """(Weighted) CE(presence) summed over all rows / row count + CE(start) + CE(end) over present rows / present
    count. Inactive characters carry -1e9. `outputs`, `weights` as in loss_terms()."""
    presence, span, present = loss_terms(model, rows, pad, device, outputs, weights)
    return presence.mean() + span / present if present else presence.mean()


def best_span(scores, row):
    """(start, end_exclusive, score) of the highest start+end score among spans with span_issue() None
    (end>=start, same passage, <=MAX_SPAN chars, trid.span_boundaries start/end cuts, no brackets);
    None when the row has no admissible span at all."""
    size = len(row['mask'])
    dev = scores.device
    starts = torch.arange(size, device=dev)[:, None]
    ends = starts + torch.arange(MAX_SPAN, device=dev)[None, :]
    safe = ends.clamp(max=size - 1)
    start_ok = torch.tensor(row['start_ok'], device=dev)
    end_ok = torch.tensor(row['end_ok'], device=dev)
    passage = torch.tensor(row['char_passage'], device=dev)
    brackets = torch.tensor(row['brackets'], device=dev)
    allowed = ((ends < size) & start_ok[:, None] & end_ok[safe] & (passage[:, None] == passage[safe])
               & (brackets[safe + 1] == brackets[starts]))
    if not allowed.any():
        return None
    joint = (scores[0, :size, None] + scores[1, safe]).masked_fill(~allowed, -float('inf'))
    flat = int(joint.argmax())
    start, length_minus_one = divmod(flat, MAX_SPAN)
    return start, start + length_minus_one + 1, float(joint.flatten()[flat])


def decode(outputs, row):
    """outputs: (kind logits [3], scores [2, chars]) of one example. Returns the answer text in the RAG answer
    grammar plus presence probabilities and the span score for later calibration. A row without any admissible
    span (best_span None) decodes to '확인 불가' with reason 'no_valid_span' and None span fields, whatever the
    presence head says: a span is never invented."""
    kind, scores = outputs
    probs = kind.softmax(-1).tolist()
    presence = int(kind.argmax())
    best = best_span(scores, row)
    if best is None:
        return {'answer': PRESENCE[1], 'presence': 1, 'presence_label': PRESENCE[1], 'presence_probs': probs,
                'span': None, 'span_text': None, 'source': None, 'span_score': None, 'reason': 'no_valid_span'}
    start, end, span_score = best
    sid = row['passages'][row['char_passage'][start]][0]
    text = row['prompt'][start:end]
    return {'answer': f'{text} [{sid}]' if presence == 0 else PRESENCE[presence],
            'presence': presence, 'presence_label': PRESENCE[presence], 'presence_probs': probs,
            'span': [start, end], 'span_text': text, 'source': sid, 'span_score': span_score, 'reason': None}


def load_reader(path, device='auto'):
    """Rebuilds the reader named by training.json: stage STAGE, 'reader_attention' (absent: causal) and
    'reader_head' (absent: v1, the layout of every run before the key lived in the metadata)."""
    root = Path(path)
    meta = json.loads((root / 'training.json').read_text())
    if meta.get('stage') != STAGE:
        raise ValueError('추출 헤드 모델 경로를 지정하세요.')
    attention = meta.get('reader_attention', 'causal')
    if attention not in ATTENTION:
        raise ValueError(f"training.json의 reader_attention은 {ATTENTION} 중 하나여야 합니다.")
    head = meta.get('reader_head', 'v1')
    if head not in HEADS:
        raise ValueError(f"training.json의 reader_head는 {HEADS} 중 하나여야 합니다.")
    model = ExtractiveReader(Config(**json.loads((root / 'config.json').read_text())), attention, head)
    model.load_state_dict(load_file(str(root / WEIGHTS)))
    device = device_for(device)
    return model.to(device).eval(), Tokenizer.from_file(str(root / 'tokenizer.json')), device


@torch.inference_mode()
def read(model, tok, prompt, device):
    row = encode(tok, prompt, context=model.backbone.cfg.context_length)
    sync(device)
    started = time.perf_counter()
    kind, scores = model(*batch([row], tok.token_to_id('<|pad|>'), device))
    result = decode((kind[0], scores[0]), row)
    sync(device)
    result.update(input_tokens=len(row['ids']), total_seconds=round(time.perf_counter() - started, 4),
                  boundary_policy=BOUNDARY_POLICY, attention=model.attention, head=model.head,
                  method='learned passage span with source id from span position; probabilities are not calibrated')
    return result
