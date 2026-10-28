"""Evidence-copy constrained decoding for the RAG answer format.

Allowed outputs (UTF-8 bytes, checked token by token on byte-level BPE):
    ABSTAIN                          -> '확인 불가' | '근거 충돌'
    VALUE ' [' ID ']' (', ' PAIR)*   -> at most `max_pairs` pairs
VALUE must be a non-empty substring of the passage it cites, starting and ending on a character
boundary, so values are copied, never invented; ID must be a passage id present in the prompt.
Every accepted prefix can still be completed, and `choose` only accepts tokens whose shortest
completion fits the remaining token budget, so generation always ends on a complete answer.
The model still chooses which span and whether to abstain.
"""
from functools import lru_cache

ABSTAIN = ("확인 불가", "근거 충돌")


@lru_cache(maxsize=4)
def _byte_decoder():
    # Inverse of the GPT-2 byte-to-unicode table used by tokenizers' ByteLevel pre-tokenizer.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def token_bytes(tok):
    """Byte string for every token id; None for special tokens."""
    decoder = _byte_decoder()
    special = {t.content for t in tok.get_added_tokens_decoder().values()}
    out = []
    for i in range(tok.get_vocab_size()):
        piece = tok.id_to_token(i)
        out.append(None if piece is None or piece in special else bytes(decoder[ch] for ch in piece))
    return out


def _boundary(text, j):
    return j == len(text) or text[j] & 0xC0 != 0x80


class EvidenceConstraint:
    def __init__(self, passages, vocab_bytes, max_pairs=2):
        """passages: [(source_id, text)]; vocab_bytes: token_bytes(tok). One instance per generation."""
        singles = {p for p in vocab_bytes if p is not None and len(p) == 1}
        if len(singles) < 256:
            raise ValueError("어휘에 모든 단일 바이트 토큰이 있어야 제한 디코딩을 쓸 수 있습니다.")
        self.passages = {sid.encode(): text.encode() for sid, text in passages}
        self.vocab_bytes = vocab_bytes
        self.max_pairs = max_pairs
        self.abstain = [a.encode() for a in ABSTAIN]
        self.prefix = b""
        # Cheapest whole first character per passage (a value can start with it), '[' and ']' excluded.
        self.first_char = {}
        for sid, text in self.passages.items():
            sizes = [len(ch.encode()) for ch in text.decode() if ch not in "[]"]
            if sizes:
                self.first_char[sid] = min(sizes)

    def choose(self, scores, eos, remaining):
        """Highest-scoring token that keeps the output completable within `remaining` tokens (this one
        included); EOS only once the answer is complete. Bytes bound tokens since every byte is a token."""
        if remaining < 1:
            raise ValueError("남은 출력 토큰이 없습니다.")
        have = self.cost(self.prefix)
        if have is None or have > remaining - 1:
            raise ValueError(f"출력 토큰 {remaining}개로는 답 형식을 완성할 수 없습니다.")
        values = scores.float().cpu()
        finite = values.tolist()
        for token in values.argsort(descending=True).tolist():
            if finite[token] == -float("inf"):
                break
            if token == eos:
                if self.complete(self.prefix):
                    return token
                continue
            piece = self.vocab_bytes[token]
            if not piece:
                continue
            need = self.cost(self.prefix + piece)
            if need is not None and need <= remaining - 2:
                self.prefix += piece
                return token
        raise RuntimeError("제한 디코딩에서 허용되는 토큰이 없습니다.")

    def _holders(self, value):
        """sid -> fewest bytes that finish the character `value` ends inside (0 if it ends on a boundary),
        over occurrences of `value` starting on a character boundary."""
        out = {}
        for sid, text in self.passages.items():
            i = text.find(value)
            while i >= 0:
                if _boundary(text, i):
                    j = i + len(value)
                    d = 0
                    while not _boundary(text, j + d):
                        d += 1
                    out[sid] = min(out.get(sid, d), d)
                    if d == 0:
                        break
                i = text.find(value, i + 1)
        return out

    def _open_value(self, value):
        """Cost to finish an answer whose last pair has the unclosed value `value`, or None."""
        if not value:
            return min((n + 3 + len(sid) for sid, n in self.first_char.items()), default=None)
        if value[:1].isspace() or b"[" in value or b"]" in value:  # a value never starts with whitespace
            return None
        costs = [d + 3 + len(sid) for sid, d in self._holders(value).items()]
        if value.endswith(b" ") and len(value) > 1:  # the space may open the ' [' separator
            costs += [2 + len(sid) for sid, d in self._holders(value[:-1]).items() if d == 0]
        return min(costs, default=None)

    def _pairs(self, b):
        pos, pairs = 0, 0
        while True:
            rest = b[pos:]
            cut = rest.find(b" [")
            if cut < 0:
                return self._open_value(rest)
            value, tail = rest[:cut], rest[cut + 2:]
            if not value.strip() or value[:1].isspace() or b"[" in value or b"]" in value:
                return None
            closed = [sid for sid, d in self._holders(value).items() if d == 0]
            done = [sid for sid in closed if tail.startswith(sid + b"]")]
            if not done:
                return min((len(sid) + 1 - len(tail) for sid in closed if (sid + b"]").startswith(tail)), default=None)
            pairs += 1
            pos += cut + 2 + len(done[0]) + 1
            after = b[pos:]
            if not after:
                return 0
            if pairs >= self.max_pairs or not (after.startswith(b", ") or b", ".startswith(after)):
                return None
            if len(after) < 2:
                more = self._open_value(b"")
                return None if more is None else 1 + more
            pos += 2

    def cost(self, prefix):
        """Fewest bytes that complete `prefix` into a valid answer; 0 if complete, None if invalid."""
        costs = [len(a) - len(prefix) for a in self.abstain if a.startswith(prefix)]
        value = self._pairs(prefix)
        if value is not None:
            costs.append(value)
        return min(costs, default=None)

    def allows(self, prefix):
        return self.cost(prefix) is not None

    def complete(self, prefix):
        return self.cost(prefix) == 0
