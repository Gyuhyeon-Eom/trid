import json
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

SPECIAL = ["<|pad|>", "<|bos|>", "<|user|>", "<|assistant|>", "<|eos|>"]


def validate_history(history):
    if not isinstance(history, list):
        raise ValueError("history는 대화 쌍의 목록이어야 합니다.")
    for turn in history:
        if not isinstance(turn, dict) or set(turn) != {"user", "assistant"}:
            raise ValueError("history 항목에는 user와 assistant만 필요합니다.")
        if any(not isinstance(v, str) or not v.strip() for v in turn.values()):
            raise ValueError("대화 내용은 비어 있지 않은 문자열이어야 합니다.")
        if any(s in v for s in SPECIAL for v in turn.values()):
            raise ValueError("대화 이력에 예약된 제어 토큰이 들어 있습니다.")


def read_rows(path):
    rows = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not all(isinstance(row.get(k), str) and row[k].strip() for k in ("prompt", "answer")):
            raise ValueError(f"{path}:{n}: 비어 있지 않은 prompt와 answer 문자열이 필요합니다.")
        if any(s in row[k] for s in SPECIAL for k in ("prompt", "answer")):
            raise ValueError(f"{path}:{n}: 예약된 제어 토큰이 들어 있습니다.")
        validate_history(row.get("history", []))
        rows.append(row)
    if not rows:
        raise ValueError(f"빈 데이터 파일: {path}")
    return rows


def train_tokenizer(rows, path, vocab_size=2048):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2,
                                 special_tokens=SPECIAL,
                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    # Train split only. Byte alphabet gives unseen UTF-8 text a representation.
    tok.train_from_iterator((r[k] for r in rows for k in ("prompt", "answer")), trainer)
    tok.save(str(path))
    return tok


def prompt_ids(tok, text, history=None):
    if any(s in text for s in SPECIAL):
        raise ValueError("입력에 예약된 제어 토큰을 사용할 수 없습니다.")
    history = [] if history is None else history
    validate_history(history)
    ids = [tok.token_to_id("<|bos|>")]
    for turn in history:
        ids += [tok.token_to_id("<|user|>")] + tok.encode(turn["user"]).ids
        ids += [tok.token_to_id("<|assistant|>")] + tok.encode(turn["assistant"]).ids + [tok.token_to_id("<|eos|>")]
    return ids + [tok.token_to_id("<|user|>")] + tok.encode(text).ids + [tok.token_to_id("<|assistant|>")]


def encode_example(tok, row, context_length):
    prefix = prompt_ids(tok, row["prompt"], row.get("history"))
    ids = prefix + tok.encode(row["answer"]).ids + [tok.token_to_id("<|eos|>")]
    if len(ids) - 1 > context_length:
        raise ValueError(f"학습 예제가 {context_length} 토큰을 초과합니다. 분할하거나 문맥 설정을 늘리세요.")
    # Supervised next-token objective. Prompt and right padding are not target labels.
    labels = [-100] * (len(prefix) - 1) + ids[len(prefix):]
    return ids[:-1], labels
