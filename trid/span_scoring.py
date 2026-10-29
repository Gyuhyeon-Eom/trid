"""Span-level answer selection: score every candidate `값 [Dn]` (plus 확인 불가 / 근거 충돌) by its full
log-likelihood under the model and return the best. Replaces token-greedy constrained decoding, which
commits to the first token before seeing the rest of the span (reports/closed-network-architecture-survey.md §8).
Candidates are substrings of the cited passage cut by trid.span_boundaries, so the copy guarantee of trid.constrained holds.
Only single-fact answers are enumerated; two-fact (combine) answers are out of scope here.
"""
import re
import torch

from .constrained import ABSTAIN
from .span_boundaries import start_ok, end_ok
from .text import prompt_ids

WORD = re.compile(r"\S+")


def candidates(passages, max_words=8, max_chars=64):
    """[(answer_text, span, source_id)]: spans start where trid.span_boundaries.start_ok allows (word starts and
    after punctuation delimiters, never inside a word or number) and end where end_ok allows (so '3일 전' comes out
    of '3일 전까지', '102%' out of '102%를' and '1911건' out of '(1911건)'), contain no bracket character, cover at
    most max_words whitespace words and max_chars characters, plus the two abstentions."""
    out, seen = [], set()
    for sid, text in passages:
        words = [(m.start(), m.end()) for m in WORD.finditer(text)]
        for i, (word_start, word_end) in enumerate(words):
            limit = words[min(len(words), i + max_words) - 1][1]
            for start in range(word_start, word_end):
                if not start_ok(text, start):
                    continue
                for end in range(start + 1, min(limit, start + max_chars) + 1):
                    if text[end - 1].isspace() or not end_ok(text, end - 1):
                        continue
                    span = text[start:end]
                    if "[" in span or "]" in span or (span, sid) in seen:
                        continue
                    seen.add((span, sid))
                    out.append((f"{span} [{sid}]", span, sid))
    out += [(a, None, None) for a in ABSTAIN]
    return out


@torch.inference_mode()
def continuation_logprobs(model, tok, prompt, answers, device, batch_size=64):
    """Sum log P(answer + eos | prompt) for each answer, sharing one prefix forward through the KV cache."""
    prefix = prompt_ids(tok, prompt)
    eos, pad = tok.token_to_id("<|eos|>"), tok.token_to_id("<|pad|>")
    encoded = [tok.encode(a).ids + [eos] for a in answers]
    longest = max(map(len, encoded))
    if len(prefix) + longest > model.cfg.context_length:
        raise ValueError("후보 답이 문맥 한도를 초과합니다.")
    logits, cache = model(torch.tensor([prefix], device=device), use_cache=True)
    first = logits[0, -1].log_softmax(-1)  # predicts the first answer token
    scores = []
    for start in range(0, len(encoded), batch_size):
        chunk = encoded[start:start + batch_size]
        b, t = len(chunk), max(map(len, chunk))
        x = torch.full((b, t), pad, device=device)
        for row, ids in enumerate(chunk):
            x[row, :len(ids)] = torch.tensor(ids, device=device)
        expanded = [(k.expand(b, -1, -1, -1), v.expand(b, -1, -1, -1)) for k, v in cache]
        out, _ = model(x, cache=expanded, use_cache=True)
        logp = out.log_softmax(-1)
        for row, ids in enumerate(chunk):
            total = first[ids[0]].item()
            for pos in range(1, len(ids)):
                total += logp[row, pos - 1, ids[pos]].item()
            scores.append(total)
    return scores


def select(model, tok, prompt, passages, device, **kw):
    """Best-scoring answer text and its score; `passages` as from trid.prompts.rag_passages."""
    cands = candidates(passages)
    scores = continuation_logprobs(model, tok, prompt, [c[0] for c in cands], device, **kw)
    best = max(range(len(cands)), key=scores.__getitem__)
    return cands[best][0], scores[best], len(cands)
