"""Train distinct span and presence models on disjoint objectives."""
import argparse
from dataclasses import asdict,replace
import json,math,random,shutil,time
from pathlib import Path
import torch
from safetensors.torch import save_file
from trid.corpus import sha256,write_json
from trid.korean_reader import encode,batch
from torch.nn import functional as F
from trid.short_reader import SpanModel,PresenceModel,component_loss,load_component
from tokenizers import Tokenizer
from trid.runtime import device_for
from trid.runtime import load_run,sync


def sample_balanced_groups(groups,rng,batch_size):
    """Keep whole counterfactual groups, including mixed two/eight-row groups."""
    selected=[]
    while len(selected)<batch_size:
        remaining=batch_size-len(selected)
        eligible=[g for g in groups if len(g)<=remaining and remaining%len(g)==0]
        if not eligible:raise ValueError('Cannot fill a balanced batch with these groups')
        selected.extend(rng.choice(eligible))
    return selected


def train_component(a,mode):
    root=Path(a.output)/mode;data=Path(a.data)
    if root.exists():raise ValueError('Use a new output directory')
    torch.manual_seed(a.seed)
    if a.init_components:
        parent=Path(a.init_components)/mode
        parent_meta=json.loads((parent/'training.json').read_text())
        if parent_meta.get('status')!='complete' or parent_meta.get('component')!=mode:
            raise ValueError('Initialization must be a completed matching component')
        device=device_for(getattr(a,'device','auto'));model=load_component(parent,device)
        cfg=model.backbone.cfg;tok=Tokenizer.from_file(str(parent/'tokenizer.json'))
    else:
        parent=Path(a.init_run)
        base,tok,device=load_run(parent,getattr(a,'device','auto'))
        cfg=replace(base.cfg,context_length=512)
        model=(SpanModel if mode=='span' else PresenceModel)(cfg)
        model.backbone.load_state_dict(base.state_dict());del base
    if mode=='span' and getattr(a,'character_encoding',None):model.expand_hangul_characters(a.character_encoding)
    if getattr(a,'freeze_backbone',False):
        for parameter in model.backbone.parameters():parameter.requires_grad_(False)
    model.to(device);rows={};groups={};documents={}
    for split in ('train','valid'):
        rows[split]=[];groups[split]={};documents[split]=set()
        for line in (data/f'{split}.jsonl').read_text().splitlines():
            r=json.loads(line);documents[split].add(r['document'])
            if mode=='span' and r['is_impossible']:continue
            row=encode(tok,r['document'],r['question'],a.max_input_tokens,[r['answer']] if not r['is_impossible'] else [],r['is_impossible'],character_encoding=getattr(model,'character_encoding','hashed512'))
            rows[split].append(row);groups[split].setdefault(r['group'],[]).append(row)
    if documents['train']&documents['valid']:raise ValueError('Document leakage')
    if mode=='presence':
        for split in groups.values():
            for group in split.values():
                if not group or len(group)%2 or sum(r['kind'] for r in group)!=len(group)//2:
                    raise ValueError('Presence groups must contain equal answerable and unanswerable rows')
                if a.batch_size%len(group):
                    raise ValueError('Batch size must be divisible by presence group size')
    root.mkdir(parents=True);shutil.copy2(parent/'tokenizer.json',root/'tokenizer.json')
    write_json(root/'config.json',asdict(cfg))
    meta={'stage':'short_reader_component_v1','component':mode,'status':'training','initialization':str(parent),
          'initial_weights_sha256':sha256(parent/'model.safetensors'),'parameters':sum(p.numel() for p in model.parameters()),
          'objective':'span marginal loss on answerable only' if mode=='span' else 'balanced two-class CE only; no span loss',
          'train_rows':len(rows['train']),'valid_rows':len(rows['valid']),'steps_requested':a.steps,'batch_size':a.batch_size,'continued_component':bool(a.init_components),'max_input_tokens':a.max_input_tokens,
          'learning_rate':a.lr,'seed':a.seed,'data_sha256':{s:sha256(data/f'{s}.jsonl') for s in rows},
          'data_provenance':json.loads((data/'provenance.json').read_text()),'device':device,
          'selection':a.selection if mode=='presence' else 'loss',
          'all_backbone_weights_trainable':not getattr(a,'freeze_backbone',False),
          'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
          'character_encoding':getattr(model,'character_encoding','not_used'),
          'presence_sampling':'whole balanced groups to exact batch size'}
    write_json(root/'training.json',meta)
    rng=random.Random(a.seed);pad=tok.token_to_id('<|pad|>');opt=torch.optim.AdamW(model.parameters(),lr=a.lr,betas=(.9,.95),weight_decay=.1)
    train_groups=list(groups['train'].values())
    @torch.inference_mode()
    def validate():
        model.eval();total=0.;correct=0
        for i in range(0,len(rows['valid']),a.batch_size):
            chunk=rows['valid'][i:i+a.batch_size]
            if mode=='presence':
                inputs=batch(chunk,pad,device);logits=model(*inputs[:2])
                labels=torch.tensor([r['kind'] for r in chunk],device=device)
                error=F.cross_entropy(logits,labels)
                correct+=int((logits.argmax(-1)==labels).sum())
            else:error=component_loss(model,chunk,pad,device,mode)
            total+=float(error)*len(chunk)
        model.train();return total/len(rows['valid']),correct/len(rows['valid']) if mode=='presence' else None
    def save():
        p=root/'model.tmp.safetensors';save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(p));p.replace(root/'model.safetensors')
    sync(device);started=time.perf_counter();initial,best_accuracy=validate();best=initial;beststep=0;save()
    metrics=[{'step':0,'valid_loss':initial}]
    print(f'{mode}: {meta["parameters"]:,} parameters; train={len(rows["train"])} valid={len(rows["valid"])} initial={initial:.4f}',flush=True)
    for step in range(1,a.steps+1):
        ratio=step/40 if step<=40 else .1+.9*.5*(1+math.cos(math.pi*(step-40)/max(1,a.steps-40)))
        for group in opt.param_groups:group['lr']=a.lr*ratio
        selected=rng.choices(rows['train'],k=a.batch_size) if mode=='span' else sample_balanced_groups(train_groups,rng,a.batch_size)
        opt.zero_grad(set_to_none=True);error=component_loss(model,selected,pad,device,mode)
        if not torch.isfinite(error):raise RuntimeError('Nonfinite loss')
        error.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step%100==0 or step==a.steps:
            valid,accuracy=validate();sync(device)
            improved=(accuracy>best_accuracy or (accuracy==best_accuracy and valid<best)) if mode=='presence' and a.selection=='accuracy' else valid<best
            if improved:best,beststep,best_accuracy=valid,step,accuracy;save()
            item={'component':mode,'step':step,'train_loss':float(error.detach()),'valid_loss':valid,'valid_accuracy':accuracy,'elapsed_seconds':round(time.perf_counter()-started,2)}
            metrics.append(item);write_json(root/'metrics.json',metrics)
            meta.update(completed_steps=step,best_step=beststep,best_validation_loss=best,best_validation_accuracy=best_accuracy);write_json(root/'training.json',meta)
            print(json.dumps(item),flush=True)
    meta.update(status='complete',initial_validation_loss=initial,training_seconds=round(time.perf_counter()-started,2),weight_bytes=(root/'model.safetensors').stat().st_size,optimizer_resume_supported=False)
    write_json(root/'training.json',meta)
    del model,opt
    if device=='mps':torch.mps.empty_cache()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data/korean-short-v1');p.add_argument('--init-run',default='runs/trid-korean-base-v1')
    p.add_argument('--output',default='runs/trid-korean-short-v1');p.add_argument('--component',choices=['span','presence','both'],default='both')
    p.add_argument('--steps',type=int,default=800);p.add_argument('--lr',type=float,default=.0002);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--init-components',help='Continue completed own span/presence weights with a fresh optimizer')
    p.add_argument('--batch-size',type=int,default=12);p.add_argument('--max-input-tokens',type=int,default=192)
    p.add_argument('--selection',choices=['loss','accuracy'],default='loss',help='Presence checkpoint criterion; accuracy uses balanced validation, CE breaks ties')
    p.add_argument('--character-encoding',choices=['hangul_v1'],help='Expand the span model character features, preserving initial outputs')
    p.add_argument('--freeze-backbone',action='store_true',help='Train the reader heads only; preserve all backbone weights')
    p.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    a=p.parse_args()
    if min(a.steps,a.batch_size,a.max_input_tokens)<1 or a.max_input_tokens>512 or not math.isfinite(a.lr) or a.lr<=0:p.error('Invalid schedule')
    torch.set_num_threads(4)
    for mode in (['span','presence'] if a.component=='both' else [a.component]):train_component(a,mode)


if __name__=='__main__':main()
