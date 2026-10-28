"""Independent learned span and presence models. No template lookup at inference."""
import json
from pathlib import Path
import time
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file
from tokenizers import Tokenizer
from .model import Trid, Config
from .korean_reader import KoreanReader, encode, decode, batch
from .runtime import device_for, sync
from .corpus import sha256
from .character_features import HANGUL_VOCAB_SIZE,inherited_hash_indices


class SpanModel(KoreanReader):
    def __init__(self,cfg,character_encoding='hashed512'):
        super().__init__(cfg)
        # The independent span model has no presence-classification parameters.
        self.kind=nn.Identity()
        self.character_encoding='hashed512'
        if character_encoding!='hashed512':self.expand_hangul_characters(character_encoding)

    def expand_hangul_characters(self,encoding='hangul_v1'):
        if encoding==self.character_encoding:return
        if encoding!='hangul_v1' or self.character_encoding!='hashed512':
            raise ValueError('Only hashed512 -> hangul_v1 migration is supported')
        old=self.character.weight
        expanded=nn.Embedding(HANGUL_VOCAB_SIZE,old.shape[1]).to(device=old.device,dtype=old.dtype)
        with torch.no_grad():
            indices=torch.tensor(inherited_hash_indices(),device=old.device)
            expanded.weight.copy_(old.index_select(0,indices))
        self.character=expanded;self.character_encoding=encoding


class PresenceModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.backbone=Trid(cfg)
        self.kind=nn.Linear(cfg.dim,2)

    def forward(self,tokens,lengths):
        h=self.backbone(tokens,hidden_only=True)
        last=h[torch.arange(len(h),device=h.device),lengths-1]
        return self.kind(last)


def component_loss(model,rows,pad,device,mode):
    inputs=batch(rows,pad,device)
    if mode=='presence':
        labels=torch.tensor([r['kind'] for r in rows],device=device)
        return F.cross_entropy(model(*inputs[:2]),labels)
    if mode!='span':raise ValueError('Unknown component')
    if any(r['kind'] or not r['spans'] for r in rows):
        raise ValueError('Span training requires answerable examples only')
    _,scores=model(*inputs);logp=scores.log_softmax(-1);losses=[]
    for i,row in enumerate(rows):
        pairs=torch.tensor(row['spans'],device=device)
        paired=logp[i,0,pairs[:,0]]+logp[i,1,pairs[:,1]]
        losses.append(-torch.logsumexp(paired,0)/2)
    return torch.stack(losses).mean()


def load_component(path,device='auto'):
    root=Path(path);meta=json.loads((root/'training.json').read_text())
    if meta.get('stage')!='short_reader_component_v1':raise ValueError('Wrong model stage')
    mode=meta['component'];cfg=Config(**json.loads((root/'config.json').read_text()))
    model=SpanModel(cfg,meta.get('character_encoding','hashed512')) if mode=='span' else PresenceModel(cfg)
    model.load_state_dict(load_file(str(root/'model.safetensors')))
    return model.to(device_for(device)).eval()


def load_pipeline(root,device='auto'):
    root=Path(root);device=device_for(device)
    if sha256(root/'span/tokenizer.json')!=sha256(root/'presence/tokenizer.json'):
        raise ValueError('Component tokenizers differ')
    span=load_component(root/'span',device);presence=load_component(root/'presence',device)
    if span.backbone.cfg!=presence.backbone.cfg:raise ValueError('Component configurations differ')
    return span,presence,Tokenizer.from_file(str(root/'span/tokenizer.json')),device


@torch.inference_mode()
def read(span,presence,tok,document,question,device):
    row=encode(tok,document,question,span.backbone.cfg.context_length,character_encoding=span.character_encoding)
    sync(device);started=time.perf_counter();inputs=batch([row],tok.token_to_id('<|pad|>'),device)
    logits=presence(*inputs[:2]);missing=bool(logits[0].argmax())
    _,scores=span(*inputs);start,end=decode(scores[0],row)
    a,b=start-row['document_start'],end-row['document_start'];answer=document[a:b]
    sync(device)
    return {'answer':'자료에서 확인할 수 없습니다.' if missing else answer,
            'abstains':missing,'source_character_span':None if missing else [a,b],
            'best_span_if_answerable':answer,'best_span_positions':[a,b],
            'raw_absent_softmax':float(logits.softmax(-1)[0,1]),
            'input_tokens':len(row['ids']),'total_seconds':round(time.perf_counter()-started,4),
            'method':'two independently trained local models; source-span extraction plus uncalibrated presence argmax'}
