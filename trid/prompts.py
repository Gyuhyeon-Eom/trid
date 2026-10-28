"""Versioned input formatting, shared by Korean training and inference."""
import json
import re
from pathlib import Path

KOREAN_GROUNDED_STYLE = "korean_grounded_sentences_v1"


def korean_grounded_prompt(document, question):
    return ("자료를 근거로 한국어로 답하세요. 없는 정보는 없다고 말하고, 모호하면 확인하세요.\n"
            f"자료: {document}\n질문: {question}")


def run_uses_korean_sentences(run):
    path = Path(run) / "training.json"
    if not path.exists():
        return False
    metadata = json.loads(path.read_text())
    return metadata.get("data_provenance", {}).get("grounded_prompt_style") == KOREAN_GROUNDED_STYLE


def korean_mrc_prompt(document, question):
    return ("자료를 읽고 질문에 답하세요. 답이 없으면 자료에서 확인할 수 없습니다라고 답하세요.\n"
            f"자료: {document}\n질문: {question}")


RAG_PREFIX = ("제공된 근거로만 답하세요. 근거가 없으면 확인 불가, 서로 다르면 근거 충돌이라고 답하세요. "
              "답에 출처 번호를 붙이세요.\n")


def rag_prompt(passages, question):
    """passages: [(source_id, text)] in display order. Shared by RAG curricula v1/v2 and RAG eval v1."""
    return RAG_PREFIX + "\n".join(f"[{sid}] {text}" for sid, text in passages) + "\n질문: " + question


def rag_passages(prompt):
    """Inverse of rag_prompt: [(source_id, text)]. Passage texts never contain '[D' (enforced by data builders)."""
    if not prompt.startswith(RAG_PREFIX) or "\n질문: " not in prompt:
        raise ValueError("not a RAG prompt")
    body = prompt[len(RAG_PREFIX):].rsplit("\n질문: ", 1)[0]
    parts = re.split(r"(?:^|\n)\[(D[1-9])\] ", body)
    if parts[0]:
        raise ValueError("text before first passage")
    return list(zip(parts[1::2], parts[2::2]))
