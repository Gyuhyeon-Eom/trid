"""Shared value types of the service layer (ingest → structure → index → engine → audit).

Offsets are character offsets into `Document.text` (the NFC-normalised, optionally PII-redacted text that was
indexed), so `document.text[p.char_start:p.char_end] == p.text` for every passage and an audit row's
(sha256, char_offset) points at the exact characters an answer was copied from.
"""
from dataclasses import dataclass, field, asdict


@dataclass(frozen=True)
class Passage:
    id: str                 # f"{doc_id}:{n:04d}", stable for one ingest of one document
    doc_id: str
    text: str
    char_start: int
    char_end: int           # exclusive
    path: tuple[str, ...]   # e.g. ("출장여비규정", "제3장", "제12조", "②"); non-regulation: (title, "¶3")

    def to_dict(self):
        d = asdict(self)
        d["path"] = list(self.path)
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(d["id"], d["doc_id"], d["text"], d["char_start"], d["char_end"], tuple(d["path"]))


@dataclass
class Document:
    doc_id: str
    title: str
    sha256: str             # of the source file bytes
    acl: set[str]           # tags; empty = visible to nobody
    text: str
    passages: list[Passage] = field(default_factory=list)
    refs: list[tuple[str, str]] = field(default_factory=list)  # (from_passage_id, to_passage_id), resolved

    def to_dict(self):
        return {"doc_id": self.doc_id, "title": self.title, "sha256": self.sha256, "acl": sorted(self.acl),
                "text": self.text, "passages": [p.to_dict() for p in self.passages],
                "refs": [list(r) for r in self.refs]}

    @classmethod
    def from_dict(cls, d):
        return cls(d["doc_id"], d["title"], d["sha256"], set(d["acl"]), d["text"],
                   [Passage.from_dict(p) for p in d["passages"]], [tuple(r) for r in d["refs"]])


@dataclass
class Answer:
    text: str                       # model output in the RAG grammar: '값 [Dn]' | '확인 불가' | '근거 충돌'
    kind: str                       # 'answer' | '확인 불가' | '근거 충돌'
    value: str | None               # copied value, None when abstaining
    source: Passage | None          # passage the value was copied from
    char_offset: int | None         # offset of `value` in the source document's text
    score: float
    candidates: list                # [(passage_id, bm25_score, via)] shown to the model, prompt order

    def to_dict(self):
        return {"text": self.text, "kind": self.kind, "value": self.value,
                "source": self.source.to_dict() if self.source else None, "char_offset": self.char_offset,
                "score": self.score, "candidates": [list(c) for c in self.candidates]}
