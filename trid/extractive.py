"""Extractive answer head on the RAG prompt (reports/closed-network-architecture-survey.md §7, §8 experiment 1).

Input is `trid.prompts.rag_prompt(passages, question)` unchanged. The trid backbone (all weights trainable)
feeds two heads: 3-way presence (답 있음 / 확인 불가 / 근거 충돌) from the last hidden state, and character-level
start/end scores over passage characters, built like trid.korean_reader (query from the last hidden state, keys
from the character's token hidden state + character features + within-token offset). `[Dn] ` markers,
whitespace and the question are masked out; the source id is derived from the passage of the chosen span,
never generated. Decoding obeys the same boundary rules as trid.span_scoring.candidates.
"""
import json
import re
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from safetensors.torch import load_file
from tokenizers import Tokenizer
from .model import Config
from .reader import Reader, batch
from .runtime import device_for, sync
from .text import prompt_ids
from .character_features import character_id
from .prompts import rag_passages, RAG_PREFIX
from .span_scoring import GLUE
from .constrained import ABSTAIN

MAX_SPAN = 64
STAGE = 'extractive_reader_v1'
WEIGHTS = 'reader.safetensors'
PRESENCE = ('present',) + ABSTAIN  # label index: 0 답 있음, 1 확인 불가, 2 근거 충돌
PAIR = re.compile(r'^\s*([^\[\]]+?)\s*\[(D[1-9])\]\s*$')  # exactly one `<value> [Dn]`


class ExtractiveReader(Reader):
    """Reader's parameter layout (3-way kind head, 2x64 query/key, character features) over hidden states only."""

    def forward(self, tokens, lengths, char_tokens, char_ids, char_offsets, mask):
        h = self.backbone(tokens, hidden_only=True)
        last = h[torch.arange(len(h), device=h.device), lengths - 1]
        q = self.query(last).reshape(-1, 2, 64)
        source = h.gather(1, char_tokens[:, :, None].expand(-1, -1, h.shape[-1]))
        features = self.character(char_ids) + self.within_token(char_offsets)
        k = (self.key(source) + self.character_key(features)).reshape(len(h), char_tokens.shape[1], 2, 64)
        scores = torch.einsum('bsd,btsd->bst', q, k) / 8
        return self.kind(last), scores.masked_fill(~mask[:, None, :], -1e9)


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


def encode(tok, prompt, answer=None, context=1024, character_encoding='hashed512'):
    """Row for batch()/loss()/decode(). Raises ValueError when the prompt is not a RAG prompt or exceeds
    `context` tokens. With `answer`, returns None when the label cannot be mapped (two pairs, free text, unknown
    source id, value absent from the cited passage, value longer than MAX_SPAN)."""
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
    for k, (sid, lo, hi) in enumerate(spans):
        for pos in range(lo, hi):
            if prompt[pos].isspace():
                continue
            mask[pos] = True
            char_passage[pos] = k
            start_ok[pos] = pos == lo or prompt[pos - 1].isspace()
            end_ok[pos] = pos + 1 >= hi or not (GLUE.match(prompt[pos]) and GLUE.match(prompt[pos + 1]))
    if not any(mask):
        raise ValueError('읽을 근거가 없습니다.')
    row = {'ids': ids, 'mask': mask, 'char_tokens': char_tokens, 'char_offsets': char_offsets,
           'char_ids': [character_id(ch, character_encoding) for ch in prompt],
           'char_passage': char_passage, 'start_ok': start_ok, 'end_ok': end_ok,
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
            loc = prompt.find(value, lo, hi)
            if loc < 0 or loc + len(value) > hi:
                return None
            row['start'], row['end'] = loc, loc + len(value) - 1
            row['label_decodable'] = start_ok[row['start']] and end_ok[row['end']]
    return row


def loss(model, rows, pad, device):
    """CE(presence) + CE(start) + CE(end); span terms only on present rows. Inactive characters carry -1e9."""
    kind, scores = model(*batch(rows, pad, device))
    kinds = torch.tensor([r['kind'] for r in rows], device=device)
    result = F.cross_entropy(kind, kinds)
    present = kinds == 0
    if present.any():
        for head, field in enumerate(('start', 'end')):
            labels = torch.tensor([r[field] for r in rows], device=device)
            result = result + F.cross_entropy(scores[present, head], labels[present])
    return result


def best_span(scores, row):
    """(start, end_exclusive, score): highest start+end score with end>=start, same passage, <=MAX_SPAN chars,
    start on a word start and end not splitting a [0-9A-Za-z,.] run."""
    size = len(row['mask'])
    dev = scores.device
    starts = torch.arange(size, device=dev)[:, None]
    ends = starts + torch.arange(MAX_SPAN, device=dev)[None, :]
    safe = ends.clamp(max=size - 1)
    start_ok = torch.tensor(row['start_ok'], device=dev)
    end_ok = torch.tensor(row['end_ok'], device=dev)
    passage = torch.tensor(row['char_passage'], device=dev)
    allowed = (ends < size) & start_ok[:, None] & end_ok[safe] & (passage[:, None] == passage[safe])
    joint = (scores[0, :size, None] + scores[1, safe]).masked_fill(~allowed, -float('inf'))
    flat = int(joint.argmax())
    start, length_minus_one = divmod(flat, MAX_SPAN)
    return start, start + length_minus_one + 1, float(joint.flatten()[flat])


def decode(outputs, row):
    """outputs: (kind logits [3], scores [2, chars]) of one example. Returns the answer text in the RAG answer
    grammar plus presence probabilities and the span score for later calibration."""
    kind, scores = outputs
    probs = kind.softmax(-1).tolist()
    presence = int(kind.argmax())
    start, end, span_score = best_span(scores, row)
    sid = row['passages'][row['char_passage'][start]][0]
    text = row['prompt'][start:end]
    return {'answer': f'{text} [{sid}]' if presence == 0 else PRESENCE[presence],
            'presence': presence, 'presence_label': PRESENCE[presence], 'presence_probs': probs,
            'span': [start, end], 'span_text': text, 'source': sid, 'span_score': span_score}


def load_reader(path, device='auto'):
    root = Path(path)
    meta = json.loads((root / 'training.json').read_text())
    if meta.get('stage') != STAGE:
        raise ValueError('추출 헤드 모델 경로를 지정하세요.')
    model = ExtractiveReader(Config(**json.loads((root / 'config.json').read_text())))
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
                  method='learned passage span with source id from span position; probabilities are not calibrated')
    return result
