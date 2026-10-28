"""Closed-network grounded QA service: ingest → structure → index → engine (ACL, audit) → server.
Product path: Python stdlib + torch/tokenizers/safetensors/numpy + kiwipiepy/bm25s; no network."""
from .types import Answer, Document, Passage

__all__ = ["Answer", "Document", "Passage"]
