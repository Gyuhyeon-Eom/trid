"""ask(query, user): ACL → BM25+graph search → rag_prompt over up to 5 passages that fit the model context →
answer backend → Answer located in the source document → audit row.

Passages go into the prompt verbatim (the normalised document slice, newlines and all): the RAG curricula
render tables, bullets and articles with line breaks inside one [Dn] passage (scripts/prepare_rag_curriculum_v2.py
render()), so the service must not flatten what the reader was trained on. A passage is shown only when
rag_passages(rag_prompt(...)) returns it unchanged (no `\\n[Dn] ` marker injection) and it carries no reserved
control token; the question is whitespace-collapsed so it cannot start a new passage or `질문:` line.

Backend contract (chosen from the run's training.json by load_backend):
  name                 label for /health and logs
  context              model context length in tokens
  tokens(prompt)       prompt length in tokens including the control tokens prompt_ids adds
  reserve(passages)    tokens the backend needs after the prompt for these [(sid, text)] passages; the engine
                       keeps the running maximum over the shown passages and admits a passage only when
                       tokens(prompt) + reserve <= context
  answer(prompt, passages) -> Reply
  ExtractiveBackend  trid.extractive reader (stage extractive_reader_v1): learned span + presence head. Consumes no
                     answer tokens (reserve 0) and reports the exact prompt span it selected.
  SpanScoringBackend trid.span_scoring.select on a dialogue/base run: every word-boundary span of the cited
                     passages plus the two abstentions, ranked by full log-likelihood. reserve is the longest
                     candidate + eos measured with the model tokenizer on the actual passages; it ranks strings,
                     so the located position is the first occurrence of the chosen string in the cited passage.
Both only ever emit '값 [Dn]' with 값 a substring of passage Dn, or '확인 불가' / '근거 충돌'; the engine verifies
값 at the reported (or first) position of the cited passage and refuses (RuntimeError) anything it cannot verify.
"""
import json
import time
from dataclasses import dataclass
from pathlib import Path

from ..constrained import ABSTAIN
from ..extractive import STAGE, PRESENCE, parse_answer, passage_spans
from ..prompts import rag_prompt, rag_passages
from ..text import SPECIAL, prompt_ids
from .acl import user_tags
from .types import Answer


@dataclass(frozen=True)
class Reply:
    text: str                      # model output in the RAG grammar: '값 [Dn]' | '확인 불가' | '근거 충돌'
    score: float
    span: tuple[int, int] | None   # [start, end) of 값 inside the prompt when the backend selected a position


def flat(text):
    return " ".join(text.split())


class SpanScoringBackend:
    name = "span_scoring"

    def __init__(self, run, device="cpu"):
        from ..runtime import load_run
        self.model, self.tok, self.device = load_run(run, device)
        self.context = self.model.cfg.context_length

    def tokens(self, prompt):
        return len(prompt_ids(self.tok, prompt))

    def reserve(self, passages):
        """Longest candidate answer of `passages` (+ eos) in tokens: what continuation_logprobs appends."""
        from ..span_scoring import candidates
        encoded = self.tok.encode_batch([text for text, _, _ in candidates(passages)])
        return max(len(e.ids) for e in encoded) + 1

    def answer(self, prompt, passages):
        from ..span_scoring import select
        text, score, _ = select(self.model, self.tok, prompt, passages, self.device)
        return Reply(text, float(score), None)


class ExtractiveBackend:
    name = "extractive"

    def __init__(self, run, device="cpu"):
        from ..extractive import load_reader
        self.model, self.tok, self.device = load_reader(run, device)
        self.context = self.model.backbone.cfg.context_length

    def tokens(self, prompt):
        return len(prompt_ids(self.tok, prompt))

    def reserve(self, passages):
        return 0  # the reader scores characters of the prompt; nothing is generated after it

    def answer(self, prompt, passages):
        from ..extractive import read
        r = read(self.model, self.tok, prompt, self.device)
        score = r["span_score"] if r["presence"] == 0 else r["presence_probs"][r["presence"]]
        return Reply(r["answer"], float(score), tuple(r["span"]) if r["presence"] == 0 else None)


def load_backend(run, device="cpu"):
    meta_path = Path(run) / "training.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return (ExtractiveBackend if meta.get("stage") == STAGE else SpanScoringBackend)(run, device)


class Engine:
    def __init__(self, index, backend, audit, users, k=3, max_passages=5, hops=1):
        self.index, self.backend, self.audit, self.users = index, backend, audit, users
        self.k, self.max_passages, self.hops = k, max_passages, hops

    def select_passages(self, query, hits):
        """Prompt passages [(sid, text)] and their Hits: in hit order, as many as fit the context (max 5).
        Skipped: passages that would not survive rag_passages unchanged or contain a control token, and passages
        after which prompt tokens + the backend's reserve exceed the context."""
        chosen, shown, reserve = [], [], 0
        for hit in hits:
            if len(chosen) >= self.max_passages:
                break
            pair = (f"D{len(shown) + 1}", hit.passage.text)
            trial = shown + [pair]
            if any(s in pair[1] for s in SPECIAL) or rag_passages(rag_prompt(trial, query)) != trial:
                continue
            need = max(reserve, self.backend.reserve([pair]))
            if self.backend.tokens(rag_prompt(trial, query)) + need > self.backend.context:
                continue
            chosen.append(hit)
            shown, reserve = trial, need
        return shown, chosen

    def ask(self, query, user):
        started = time.perf_counter()
        query = flat(query)
        if not query:
            raise ValueError("질문을 입력해 주세요.")
        if any(s in query for s in SPECIAL):
            raise ValueError("질문에 예약된 제어 토큰을 사용할 수 없습니다.")
        hits = self.index.search(query, user_tags(self.users, user), self.k, self.hops)
        shown, chosen = self.select_passages(query, hits)
        candidates = [(h.passage.id, h.score, h.via) for h in chosen]
        if not chosen:
            answer = Answer(ABSTAIN[0], ABSTAIN[0], None, None, None, 0.0, candidates)
        else:
            prompt = rag_prompt(shown, query)
            reply = self.backend.answer(prompt, shown)
            parsed = parse_answer(reply.text)
            if parsed is None:
                raise RuntimeError(f"답 형식 오류: {reply.text!r}")
            presence, value, sid = parsed
            if presence:
                answer = Answer(reply.text, PRESENCE[presence], None, None, None, reply.score, candidates)
            else:
                passage = chosen[int(sid[1:]) - 1].passage
                offset = self.locate(reply, value, sid, prompt, passage)
                answer = Answer(reply.text, "answer", value, passage, passage.char_start + offset, reply.score, candidates)
        self.audit.record(user, query, answer, self.index.documents, time.perf_counter() - started)
        return answer

    @staticmethod
    def locate(reply, value, sid, prompt, passage):
        """Offset of `value` inside `passage.text`: the backend's prompt span mapped through the passage's
        position in the prompt, or the first occurrence when the backend selected a string. The shown text is
        passage.text verbatim, so the prompt offset minus the passage start is the document-local offset."""
        if reply.span is None:
            offset = passage.text.find(value)
            if offset < 0:
                raise RuntimeError(f"답이 인용한 문단에 없습니다: {value!r} / {passage.id}")
        else:
            start, end = reply.span
            if prompt[start:end] != value:
                raise RuntimeError(f"답 위치가 답과 다릅니다: {prompt[start:end]!r} != {value!r}")
            lo, hi = next((lo, hi) for s, lo, hi in passage_spans(prompt) if s == sid)
            if not lo <= start < end <= hi:
                raise RuntimeError(f"답 위치가 인용한 문단 {sid} 밖입니다: {reply.span}")
            offset = start - lo
        if passage.text[offset:offset + len(value)] != value:
            raise RuntimeError(f"문단 위치 {offset}가 원문과 맞지 않습니다: {passage.id}")
        return offset
