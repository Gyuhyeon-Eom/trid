"""ask(query, user): ACL → BM25+graph search → rag_prompt over up to 5 passages that fit the model context →
answer backend → Answer located in the source document → audit row.

Backends (chosen from the run's training.json):
  ExtractiveBackend  trid.extractive reader (stage extractive_reader_v1): learned span + presence head.
  SpanScoringBackend trid.span_scoring.select on a dialogue/base run: every word-boundary span of the cited
                     passages plus the two abstentions, ranked by full log-likelihood.
Both only ever emit '값 [Dn]' with 값 a substring of passage Dn, or '확인 불가' / '근거 충돌'; the engine
re-locates 값 in the passage and refuses (RuntimeError) to return anything it cannot find there.
"""
import json
import re
import time
from pathlib import Path

from ..constrained import ABSTAIN
from ..extractive import STAGE, PRESENCE, parse_answer
from ..prompts import rag_prompt
from ..text import prompt_ids
from .acl import user_tags
from .types import Answer

ANSWER_BUDGET = 48  # tokens reserved after the prompt: the longest candidate (64 chars) + ' [Dn]' + eos


def locate(value, text):
    """(offset, exact slice) of `value` in `text`, tolerant to the whitespace flattening of the prompt."""
    m = re.search(r"\s+".join(map(re.escape, value.split())), text)
    return (m.start(), m.group()) if m else None


def flat(text):
    return re.sub(r"\s+", " ", text).strip()


class SpanScoringBackend:
    name = "span_scoring"

    def __init__(self, run, device="cpu"):
        from ..runtime import load_run
        self.model, self.tok, self.device = load_run(run, device)
        self.context = self.model.cfg.context_length

    def tokens(self, prompt):
        return len(prompt_ids(self.tok, prompt))

    def answer(self, prompt, passages):
        from ..span_scoring import select
        text, score, _ = select(self.model, self.tok, prompt, passages, self.device)
        return text, score


class ExtractiveBackend:
    name = "extractive"

    def __init__(self, run, device="cpu"):
        from ..extractive import load_reader
        self.model, self.tok, self.device = load_reader(run, device)
        self.context = self.model.backbone.cfg.context_length

    def tokens(self, prompt):
        return len(prompt_ids(self.tok, prompt))

    def answer(self, prompt, passages):
        from ..extractive import read
        r = read(self.model, self.tok, prompt, self.device)
        return r["answer"], (r["span_score"] if r["presence"] == 0 else r["presence_probs"][r["presence"]])


def load_backend(run, device="cpu"):
    meta_path = Path(run) / "training.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return (ExtractiveBackend if meta.get("stage") == STAGE else SpanScoringBackend)(run, device)


class Engine:
    def __init__(self, index, backend, audit, users, k=3, max_passages=5, hops=1):
        self.index, self.backend, self.audit, self.users = index, backend, audit, users
        self.k, self.max_passages, self.hops = k, max_passages, hops

    def select_passages(self, query, hits):
        """Prompt passages [(sid, text)] and their Hits: in hit order, as many as fit the context (max 5)."""
        chosen, shown = [], []
        for hit in hits:
            if len(chosen) >= self.max_passages:
                break
            trial = shown + [(f"D{len(shown) + 1}", flat(hit.passage.text))]
            if self.backend.tokens(rag_prompt(trial, query)) + ANSWER_BUDGET > self.backend.context:
                continue
            chosen.append(hit)
            shown = trial
        return shown, chosen

    def ask(self, query, user):
        started = time.perf_counter()
        query = query.strip()
        if not query:
            raise ValueError("질문을 입력해 주세요.")
        hits = self.index.search(query, user_tags(self.users, user), self.k, self.hops)
        shown, chosen = self.select_passages(query, hits)
        candidates = [(h.passage.id, h.score, h.via) for h in chosen]
        if not chosen:
            answer = Answer(ABSTAIN[0], ABSTAIN[0], None, None, None, 0.0, candidates)
        else:
            text, score = self.backend.answer(rag_prompt(shown, query), shown)
            parsed = parse_answer(text)
            if parsed is None:
                raise RuntimeError(f"답 형식 오류: {text!r}")
            presence, value, sid = parsed
            if presence:
                answer = Answer(text, PRESENCE[presence], None, None, None, float(score), candidates)
            else:
                passage = chosen[int(sid[1:]) - 1].passage
                found = locate(value, passage.text)
                if found is None:
                    raise RuntimeError(f"답이 인용한 문단에 없습니다: {value!r} / {passage.id}")
                offset, exact = found
                assert passage.text[offset:offset + len(exact)] == exact
                answer = Answer(text, "answer", exact, passage, passage.char_start + offset, float(score), candidates)
        self.audit.record(user, query, answer, self.index.documents, time.perf_counter() - started)
        return answer
