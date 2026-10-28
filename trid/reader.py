"""Learned extractive reader over trid's own Transformer.

Selects a source span or a learned missing/conflict class. It does not generate
an answer or claim that its chosen source span is necessarily correct.
"""
import json, math, random, shutil, time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from .model import Config, Trid
from .runtime import device_for, sync
from .text import prompt_ids
from .corpus import sha256

KINDS=('span','자료없음','규정충돌')


class Reader(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.backbone=Trid(cfg)
        self.query=nn.Linear(cfg.dim,128,bias=False)
        self.key=nn.Linear(cfg.dim,128,bias=False)
        self.kind=nn.Linear(cfg.dim,3)
        self.character=nn.Embedding(512,32)
        self.within_token=nn.Embedding(32,32)
        self.character_key=nn.Linear(32,128,bias=False)

    def forward(self,tokens,lengths,char_tokens,char_ids,char_offsets,mask):
        _,_,h=self.backbone(tokens,return_hidden=True)
        last=h[torch.arange(len(h),device=h.device),lengths-1]
        q=self.query(last).reshape(-1,2,64)
        source=h.gather(1,char_tokens[:,:,None].expand(-1,-1,h.shape[-1]))
        features=self.character(char_ids)+self.within_token(char_offsets)
        k=(self.key(source)+self.character_key(features)).reshape(len(h),char_tokens.shape[1],2,64)
        scores=torch.einsum('bsd,btsd->bst',q,k)/8
        return self.kind(last),scores.masked_fill(~mask[:,None,:],-1e9)


def encode(tok,prompt,context,answer=None):
    prefix='\n자료: '
    if prefix not in prompt or '\n질문: ' not in prompt:raise ValueError('자료와 질문 구분이 필요합니다.')
    lo=prompt.index(prefix)+len(prefix);hi=prompt.index('\n질문: ',lo)
    encoded=tok.encode(prompt)
    ids=prompt_ids(tok,prompt)
    if len(ids)>context:raise ValueError('문맥 한도를 초과했습니다. 문서를 줄여 주세요.')
    # Character boundaries avoid dropping answers whose suffix shares a BPE token.
    char_tokens=[0]*len(prompt);char_offsets=[0]*len(prompt)
    for i,(a,b) in enumerate(encoded.offsets):
        for pos in range(a,b):
            char_tokens[pos]=i+2
            char_offsets[pos]=min(pos-a,31)
    mask=[lo<=pos<hi and not ch.isspace() for pos,ch in enumerate(prompt)]
    if not any(mask):raise ValueError('읽을 자료가 없습니다.')
    row={'ids':ids,'mask':mask,'char_tokens':char_tokens,
         'char_ids':[ord(ch)%512 for ch in prompt],'char_offsets':char_offsets,'prompt':prompt}
    if answer is not None:
        row['kind']=KINDS.index(answer) if answer in KINDS[1:] else 0
        row['start']=row['end']=0
        if row['kind']==0:
            loc=prompt.find(answer,lo,hi)
            while loc>=0:
                end=loc+len(answer)
                inside_number=(answer[0].isdigit() and loc>lo and prompt[loc-1].isdigit()) or (answer[-1].isdigit() and end<hi and prompt[end].isdigit())
                if not inside_number:break
                loc=prompt.find(answer,loc+1,hi)
            if loc<0:raise ValueError('정답이 자료에 없습니다.')
            row['start'],row['end']=loc,loc+len(answer)-1
    return row



def batch(rows,pad,device):
    size=max(len(r['ids']) for r in rows)
    ids=torch.full((len(rows),size),pad,dtype=torch.long,device=device)
    chars=max(len(r['mask']) for r in rows)
    mask=torch.zeros((len(rows),chars),dtype=torch.bool,device=device)
    features=[torch.zeros((len(rows),chars),dtype=torch.long,device=device) for _ in range(3)]
    for i,r in enumerate(rows):
        ids[i,:len(r['ids'])]=torch.tensor(r['ids'],device=device)
        mask[i,:len(r['mask'])]=torch.tensor(r['mask'],device=device)
        for field,tensor in zip(('char_tokens','char_ids','char_offsets'),features):
            tensor[i,:len(r[field])]=torch.tensor(r[field],device=device)
    return ids,torch.tensor([len(r['ids']) for r in rows],device=device),*features,mask



def loss(model,rows,pad,device):
    kind,scores=model(*batch(rows,pad,device))
    kinds=torch.tensor([r['kind'] for r in rows],device=device)
    result=F.cross_entropy(kind,kinds)
    selected=kinds==0
    if selected.any():
        for head,field in enumerate(('start','end')):
            labels=torch.tensor([r[field] for r in rows],device=device)
            result=result+F.cross_entropy(scores[selected,head],labels[selected])/2
    return result


def load_reader(path,device='auto'):
    root=Path(path);device=device_for(device)
    model=Reader(Config(**json.loads((root/'config.json').read_text())))
    model.load_state_dict(load_file(str(root/'reader.safetensors')))
    model.to(device).eval()
    return model,Tokenizer.from_file(str(root/'tokenizer.json')),device


@torch.inference_mode()
def read(model,tok,prompt,device):
    row=encode(tok,prompt,model.backbone.cfg.context_length)
    sync(device);started=time.perf_counter()
    kinds,scores=model(*batch([row],tok.token_to_id('<|pad|>'),device))
    kind=int(kinds[0].argmax())
    start=end=None;span=None
    if kind:
        answer=KINDS[kind]
    else:
        # Joint decoding permits only spans within the supplied evidence and in order, up to 32 characters.
        size=len(row['mask']);idx=torch.arange(size,device=device)
        valid=(idx[:,None]<=idx[None,:]) & (idx[None,:]-idx[:,None]<32)
        allowed=torch.tensor(row['mask'],device=device)
        valid &= allowed[:,None]&allowed[None,:]
        joint=(scores[0,0,:,None]+scores[0,1,None,:]).masked_fill(~valid,-float('inf'))
        best=int(joint.argmax());start,end=divmod(best,size)
        a,b=start,end+1
        answer=prompt[a:b].strip();span=[a,b]
    sync(device)
    return {'answer':answer,'kind':KINDS[kind],'source_character_span':span,
            'input_tokens':len(row['ids']),'total_seconds':round(time.perf_counter()-started,4),
            'method':'learned source-span selection, not free-form generation'}


def train_reader(data,init_run,output,steps=1400):
    from .runtime import load_run
    if steps<1:raise ValueError('학습 단계는 양수여야 합니다.')
    root=Path(output);data=Path(data)
    if root.exists():raise ValueError('Use a new output directory.')
    torch.set_num_threads(4);torch.manual_seed(72)
    base,tok,device=load_run(init_run,'auto')
    model=Reader(base.cfg);model.backbone.load_state_dict(base.state_dict());del base
    model.to(device)
    rows={};skipped={}
    for split in ('train','valid'):
        rows[split]=[];skipped[split]=0
        for line in (data/f'{split}.jsonl').read_text().splitlines():
            r=json.loads(line)
            if not r['task'].startswith('grounded-'):continue
            try:rows[split].append(encode(tok,r['prompt'],model.backbone.cfg.context_length,r['answer']))
            except ValueError:skipped[split]+=1
    if not all(rows.values()):raise ValueError('Empty training or validation data.')
    if {r['prompt'] for r in rows['train']} & {r['prompt'] for r in rows['valid']}:
        raise ValueError('학습과 검증 자료에 동일한 입력이 있습니다.')
    valid=random.Random(73).sample(rows['valid'],min(256,len(rows['valid'])))
    root.mkdir();shutil.copy2(Path(init_run)/'config.json',root/'config.json');shutil.copy2(Path(init_run)/'tokenizer.json',root/'tokenizer.json')
    meta={'stage':'extractive_reader','initialization':str(init_run),'initial_weights_sha256':sha256(Path(init_run)/'model.safetensors'),'parameters':sum(p.numel() for p in model.parameters()),'steps_requested':steps,'batch_size':24,'learning_rate':.00015,'seed':72,'validation_sample_seed':73,'train_rows':len(rows['train']),'valid_rows':len(valid),'unrepresentable_spans_skipped':skipped,'span_unit':'Unicode character; contextual BPE features plus character/within-token features','objective':'learned answer kind plus start/end source positions; no language-generation objective','data_sha256':{s:sha256(data/f'{s}.jsonl') for s in ('train','valid')},'status':'training','limits':'Developer-authored synthetic corpus. Extracts a predicted source span; no retrieval, calibrated probabilities, or general dialogue claims.'}
    (root/'training.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    opt=torch.optim.AdamW(model.parameters(),lr=.00015,betas=(.9,.95),weight_decay=.1)
    rng=random.Random(72);pad=tok.token_to_id('<|pad|>');metrics=[]
    @torch.inference_mode()
    def validate():
        model.eval()
        vals=[float(loss(model,valid[i:i+24],pad,device)) for i in range(0,len(valid),24)]
        model.train();return sum(vals)/len(vals)
    best=validate();beststep=0
    def save():
        save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(root/'reader.tmp.safetensors'))
        (root/'reader.tmp.safetensors').replace(root/'reader.safetensors')
    save();sync(device);start=time.perf_counter()
    print(f'Reader baseline loss {best:.4f}; {meta["parameters"]:,} parameters; skipped {skipped}',flush=True)
    for step in range(1,steps+1):
        ratio=min(step/50,1.)*(.1+.9*.5*(1+math.cos(math.pi*step/steps)))
        for group in opt.param_groups:group['lr']=.00015*ratio
        model.train();opt.zero_grad(set_to_none=True)
        error=loss(model,rng.choices(rows['train'],k=24),pad,device)
        if not torch.isfinite(error):raise RuntimeError('Nonfinite loss')
        error.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step%100==0 or step==steps:
            score=validate();sync(device)
            if score<best:best=score;beststep=step;save()
            record={'step':step,'train_loss':float(error.detach()),'valid_loss':score,'elapsed_seconds':round(time.perf_counter()-start,2)}
            metrics.append(record);(root/'metrics.json').write_text(json.dumps(metrics,indent=2))
            print(json.dumps(record),flush=True)
    meta.update(status='complete',best_step=beststep,best_validation_loss=best,completed_steps=steps,training_seconds=round(time.perf_counter()-start,2),weight_bytes=(root/'reader.safetensors').stat().st_size)
    (root/'training.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
