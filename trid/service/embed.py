"""BAAI/bge-m3 dense embeddings (MIT) from the local Hugging Face snapshot; never downloads at run time.

The indexer (scripts/build_embeddings.py) and the service (scripts/shared_demo.py Workspace) share `passage_input`
and the model settings, so query and passage vectors come from the same model, truncation and normalisation.
"""
import hashlib
from pathlib import Path

MODEL = 'BAAI/bge-m3'
MAX_TOKENS = 512


def passage_input(meta, passage):
    """The string embedded for a passage (also its cache key): document identity first, then the verbatim text."""
    m = meta[passage.doc_id]
    return f"{m.get('institution') or ''} {m['title']}\n{passage.text}".strip()


def load_model(device='cpu'):
    """(SentenceTransformer, snapshot path). float16 off the CPU; float32 on the CPU (query-time encoding)."""
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
    path = snapshot_download(MODEL, local_files_only=True)
    model = SentenceTransformer(path, device=device)
    model.max_seq_length = MAX_TOKENS
    if device != 'cpu':
        model.half()
    return model, path


def weight_sha256(path):
    with open(Path(path) / 'pytorch_model.bin', 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
