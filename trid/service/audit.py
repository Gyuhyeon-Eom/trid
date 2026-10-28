"""Append-only JSONL audit log. One row per question: who asked (user), what (sha256 of the query, never the
text), which documents were candidates, which document (sha256) and character range the answer was copied
from, the answer kind and score. No document or query text is ever written, so the log can live outside the
document ACL; the text is recovered from the index by (doc sha256, offsets) when an auditor needs it.
"""
import datetime
import hashlib
import json
from pathlib import Path


def query_hash(query):
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


class AuditLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def row(self, user, query, answer, documents, seconds=None):
        """documents: {doc_id: Document} to look up sha256 for the candidate and chosen documents."""
        src = answer.source
        source = None
        if src is not None and answer.value is not None:
            source = {"doc_id": src.doc_id, "sha256": documents[src.doc_id].sha256, "passage_id": src.id,
                      "path": list(src.path), "char_start": answer.char_offset,
                      "char_end": answer.char_offset + len(answer.value)}
        candidate_docs = []
        for pid, _, _ in answer.candidates:
            doc_id = pid.rsplit(":", 1)[0]
            if doc_id not in candidate_docs:
                candidate_docs.append(doc_id)
        return {"ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds"),
                "user": user, "query_sha256": query_hash(query),
                "candidate_docs": [{"doc_id": d, "sha256": documents[d].sha256} for d in candidate_docs],
                "candidates": [[pid, round(score, 4), via] for pid, score, via in answer.candidates],
                "kind": answer.kind, "score": round(answer.score, 4), "source": source,
                "seconds": None if seconds is None else round(seconds, 4)}

    def record(self, user, query, answer, documents, seconds=None):
        row = self.row(user, query, answer, documents, seconds)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
        return row
