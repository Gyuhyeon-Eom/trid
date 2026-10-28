"""Question-conditioned character-span reader using trid's own causal backbone."""
import json
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file
from tokenizers import Tokenizer
from .model import Config
from .reader import Reader, batch
from .runtime import device_for, sync
from .text import prompt_ids
from .character_features import character_id

MAX_SPAN = 64


class KoreanReader(Reader):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.kind = nn.Linear(cfg.dim, 2)  # answer present / absent; no conflict labels

    def forward(self, tokens, lengths, char_tokens, char_ids, char_offsets, mask):
        h = self.backbone(tokens, hidden_only=True)
        last = h[torch.arange(len(h), device=h.device), lengths - 1]
        q = self.query(last).reshape(-1, 2, 64)
        source = h.gather(1, char_tokens[:, :, None].expand(-1, -1, h.shape[-1]))
        features = self.character(char_ids) + self.within_token(char_offsets)
        k = (self.key(source) + self.character_key(features)).reshape(len(h), char_tokens.shape[1], 2, 64)
        scores = torch.einsum('bsd,btsd->bst', q, k) / 8
        return self.kind(last), scores.masked_fill(~mask[:, None, :], -1e9)


def source_parts(row):
    content = row['prompt'].split('\n자료: ', 1)[1]
    return content.rsplit('\n질문: ', 1)


def encode(tok, document, question, context=512, answers=None, impossible=None, character_encoding='hashed512'):
    # Source tokens can attend to the question with the inherited causal mask.
    prefix = f'질문: {question}\n자료: '
    prompt = prefix + document
    lo, hi = len(prefix), len(prompt)
    ids = prompt_ids(tok, prompt)
    if len(ids) > context:
        raise ValueError(f'입력 {len(ids)} 토큰이 문맥 한도 {context}를 초과합니다.')
    encoded = tok.encode(prompt)
    char_tokens, char_offsets = [0] * hi, [0] * hi
    for i, (a, b) in enumerate(encoded.offsets):
        for pos in range(a, b):
            char_tokens[pos] = i + 2
            char_offsets[pos] = min(pos - a, 31)
    mask = [lo <= i < hi and not ch.isspace() for i, ch in enumerate(prompt)]
    if not any(mask):
        raise ValueError('읽을 문서가 없습니다.')
    row = {'ids': ids, 'mask': mask, 'char_tokens': char_tokens,
           'char_ids': [character_id(ch,character_encoding) for ch in prompt], 'char_offsets': char_offsets,
           'prompt': prompt, 'document_start': lo, 'document': document, 'question': question}
    if impossible is not None:
        row['kind'] = int(impossible)
        row['spans'] = []
        if not impossible:
            for answer in answers or []:
                answer = answer.strip()
                if not answer or len(answer) > MAX_SPAN:
                    continue
                start = document.find(answer)
                while start >= 0:
                    end = start + len(answer)
                    inside_number = ((answer[0].isdigit() and start > 0 and document[start-1].isdigit()) or
                                     (answer[-1].isdigit() and end < len(document) and document[end].isdigit()))
                    if not inside_number:
                        row['spans'].append((lo + start, lo + end - 1))
                    start = document.find(answer, start + 1)
            row['spans'] = sorted(set(row['spans']))
            if not row['spans']:
                raise ValueError('표현 가능한 정답 구간이 문서에 없습니다.')
    return row


def loss(model, rows, pad, device):
    kind, scores = model(*batch(rows, pad, device))
    kinds = torch.tensor([r['kind'] for r in rows], device=device)
    per_example = F.cross_entropy(kind, kinds, reduction='none')
    logp = scores.log_softmax(-1)
    span_losses = []
    for i, row in enumerate(rows):
        if row['kind']:
            span_losses.append(logp[i].sum() * 0)
        else:
            positions = torch.tensor(row['spans'], device=device)
            # Marginalize complete valid pairs, not unrelated start/end combinations.
            paired = logp[i, 0, positions[:, 0]] + logp[i, 1, positions[:, 1]]
            span_losses.append(-torch.logsumexp(paired, 0) / 2)
    return (per_example + torch.stack(span_losses)).mean()


def decode(scores, row):
    size = len(row['mask'])
    starts = torch.arange(size, device=scores.device)[:, None]
    ends = starts + torch.arange(MAX_SPAN, device=scores.device)[None, :]
    safe = ends.clamp(max=size-1)
    mask = torch.tensor(row['mask'], device=scores.device)
    allowed = (ends < size) & mask[:, None] & mask[safe]
    joint = (scores[0, :size, None] + scores[1, safe]).masked_fill(~allowed, -float('inf'))
    flat = int(joint.argmax())
    start, length_minus_one = divmod(flat, MAX_SPAN)
    return start, start + length_minus_one + 1


def load_reader(path, device='auto'):
    root = Path(path)
    meta = json.loads((root/'training.json').read_text())
    if meta.get('stage') != 'korean_span_reader_v1':
        raise ValueError('한국어 구간 추출 모델 경로를 지정하세요.')
    model = KoreanReader(Config(**json.loads((root/'config.json').read_text())))
    model.load_state_dict(load_file(str(root/'reader.safetensors')))
    device = device_for(device)
    return model.to(device).eval(), Tokenizer.from_file(str(root/'tokenizer.json')), device


@torch.inference_mode()
def read(model, tok, document, question, device):
    row = encode(tok, document, question, model.backbone.cfg.context_length)
    sync(device)
    started = time.perf_counter()
    kinds, scores = model(*batch([row], tok.token_to_id('<|pad|>'), device))
    absent = bool(kinds[0].argmax())
    start, end = decode(scores[0], row)
    a, b = start-row['document_start'], end-row['document_start']
    best_span = document[a:b]
    sync(device)
    return {'answer': '자료에서 확인할 수 없습니다.' if absent else best_span,
            'abstains': absent, 'source_character_span': None if absent else [a, b],
            'best_span_if_answerable': best_span, 'best_span_positions': [a, b],
            'raw_absent_softmax': float(kinds.softmax(-1)[0, 1]),
            'input_tokens': len(row['ids']), 'total_seconds': round(time.perf_counter()-started, 4),
            'method': 'learned source span or absence; probabilities are not calibrated'}
