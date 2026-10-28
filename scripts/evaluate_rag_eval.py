"""Held-out RAG evaluation (data/korean-rag-eval-v1): value, source and abstention scored separately.

strict: every requested value and its source id are correct (abstain items: exact abstain phrase).
value: values correct ignoring source ids. lenient_value: generated value contains the gold value.
Greedy CPU generation, no answer substitution. Items over the model context are counted as failures
and listed under `skipped_context`. `--decoding evidence` restricts output to the answer grammar with
values copied from the cited passage (trid.constrained); the model still picks span, source and abstention.
`--decoding span` scores every word-boundary span of every passage as a whole answer (trid.span_scoring) and
takes the best; `--span-norm mean` divides by answer tokens to remove the short-answer bias.
"""
import argparse,hashlib,json,re,time,unicodedata
from collections import Counter,defaultdict
from pathlib import Path
import torch
from trid.runtime import load_run,generate
from trid.text import prompt_ids
from trid.prompts import rag_passages
from trid.constrained import EvidenceConstraint,token_bytes
from trid.span_scoring import candidates,continuation_logprobs

ABSTAIN=('확인불가','근거충돌')
PAIR=re.compile(r'([^\[\]]+?)\s*\[(D[1-9])\]')

def normalized(s):
    return ''.join(unicodedata.normalize('NFKC',s).split()).rstrip('.')

def parse(text):
    """'<value> [Dn], <value> [Dm]' -> [(value, source)]; values normalized."""
    return [(normalized(v.strip().lstrip(',').strip()),s) for v,s in PAIR.findall(text)]

def score(row,generated):
    pred=normalized(generated)
    if not row['gold']:
        ok=pred==normalized(row['answer'])
        return {'strict':ok,'value':ok,'lenient_value':ok,'source':ok,'abstained':pred in ABSTAIN}
    pairs=parse(generated);gold=[[normalized(g) for g in variants] for variants in row['gold']]
    n=len(gold);aligned=len(pairs)==n
    value=aligned and all(p[0] in g for p,g in zip(pairs,gold))
    lenient=aligned and all(any(x in p[0] for x in g) for p,g in zip(pairs,gold))
    source=aligned and all(p[1]==s for p,s in zip(pairs,row['sources']))
    return {'strict':value and source,'value':value,'lenient_value':lenient,'source':source,'abstained':pred in ABSTAIN}

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    p.add_argument('--data',default='data/korean-rag-eval-v1');p.add_argument('--device',default='cpu');p.add_argument('--limit',type=int,default=0)
    p.add_argument('--decoding',choices=['free','evidence','span'],default='free');p.add_argument('--span-norm',choices=['sum','mean'],default='sum')
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise ValueError('Use a new output file')
    root=Path(a.data);test=root/'test.jsonl';prov=json.loads((root/'provenance.json').read_text())
    digest=hashlib.sha256(test.read_bytes()).hexdigest()
    if digest!=prov['hashes']['test']:raise ValueError('Evaluation data changed')
    weights=Path(a.run)/'model.safetensors';model_digest=hashlib.sha256(weights.read_bytes()).hexdigest()
    torch.set_num_threads(4);model,tok,device=load_run(a.run,a.device)
    vocab=token_bytes(tok) if a.decoding=='evidence' else None
    rows=[json.loads(l) for l in test.read_text().splitlines()]
    if a.limit:rows=rows[:a.limit]
    results=[];agg=defaultdict(Counter)
    for row in rows:
        if len(prompt_ids(tok,row['prompt']))+64>model.cfg.context_length:
            s={'strict':False,'value':False,'lenient_value':False,'source':False,'abstained':False};generated=None;skip=True
        else:
            if a.decoding=='span':
                cands=candidates(rag_passages(row['prompt']));texts=[c[0] for c in cands]
                scores=continuation_logprobs(model,tok,row['prompt'],texts,device)
                if a.span_norm=='mean':scores=[s/(len(tok.encode(x).ids)+1) for s,x in zip(scores,texts)]
                generated=texts[max(range(len(texts)),key=scores.__getitem__)]
            else:
                constraint=EvidenceConstraint(rag_passages(row['prompt']),vocab) if vocab else None
                generated=generate(model,tok,row['prompt'],device,max_tokens=64,constraint=constraint)['answer']
            s=score(row,generated);skip=False
        for key in ('all',row['origin'],row['task'],'answerable' if row['gold'] else 'unanswerable'):
            c=agg[key];c['total']+=1;c['skipped_context']+=skip
            for m,v in s.items():c[m]+=v
        results.append({'id':row['id'],'task':row['task'],'origin':row['origin'],'answer':row['answer'],'generated':generated,**s,'skipped_context':skip})
    if hashlib.sha256(weights.read_bytes()).hexdigest()!=model_digest:raise ValueError('Model changed')
    summary={k:dict(v) for k,v in sorted(agg.items())}
    report={'run':a.run,'model_sha256':model_digest,'data':a.data,'test_sha256':digest,'evaluated_at_unix':time.time(),
      'model_context_length':model.cfg.context_length,'limit':a.limit or None,'decoding':a.decoding,'span_norm':a.span_norm if a.decoding=='span' else None,'protocol':__doc__.strip(),
      'summary':summary,'results':results}
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:{m:v.get(m,0) for m in ('strict','value','source','abstained','skipped_context','total')} for k,v in summary.items()},ensure_ascii=False,indent=1))

if __name__=='__main__':main()
