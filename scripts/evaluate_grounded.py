"""Actual generation over supplied evidence; no retrieval or answer replacement."""
import argparse, hashlib, json
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone
import torch
from trid.runtime import load_run, generate
from evaluate_institution_demo import make_prompt, normalize


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True)
    p.add_argument('--data',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    out=Path(args.output)
    if out.exists():raise ValueError('Use a new output path.')
    rows=[json.loads(x) for x in Path(args.data).read_text().splitlines()]
    torch.set_num_threads(4)
    extractive=(Path(args.run)/'reader.safetensors').exists()
    if extractive:
        from trid.reader import load_reader,read
        model,tok,device=load_reader(args.run,'auto')
    else:
        model,tok,device=load_run(args.run,'auto')
    results=[];groups=defaultdict(lambda:{'count':0,'correct':0});pairs=defaultdict(list)
    for n,r in enumerate(rows):
        result=read(model,tok,make_prompt(r),device) if extractive else generate(model,tok,make_prompt(r),device,max_tokens=48)
        ok=normalize(result['answer'])==normalize(r['expected'])
        item={**r,'generated':result,'correct':ok};results.append(item)
        groups[r['group']]['count']+=1;groups[r['group']]['correct']+=int(ok)
        if r.get('counterfactual_pair'):pairs[r['counterfactual_pair']].append(item)
        if (n+1)%50==0:print(f'{n+1}/{len(rows)} correct {sum(x["correct"] for x in results)}',flush=True)
    if any(len(v)!=2 for v in pairs.values()):raise ValueError('Counterfactual pairs must have exactly two members.')
    filename='reader.safetensors' if extractive else 'model.safetensors'
    report={'run':args.run,'engine':'learned_extractive_reader' if extractive else 'generative_lm','evaluated_at':datetime.now(timezone.utc).isoformat(),'device':device,'data_sha256':hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),'weights_sha256':hashlib.sha256((Path(args.run)/filename).read_bytes()).hexdigest(),'protocol':'Exact whitespace/punctuation-normalized value matching. Relevant evidence supplied directly; no retrieval, answer substitution, or external API. Synthetic developer-authored suite; not real institutional competence. Extractive engine returns its predicted source span or learned missing/conflict class, not generated prose.','count':len(results),'correct':sum(x['correct'] for x in results),'groups':dict(groups),'counterfactual_pairs':len(pairs),'counterfactual_pairs_both_correct':sum(all(x['correct'] for x in v) for v in pairs.values()),'results':results}
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='results'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
