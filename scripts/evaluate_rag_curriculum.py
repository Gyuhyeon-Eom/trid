"""Exact fact+source diagnostic on short synthetic evidence, not retrieval accuracy."""
import argparse,hashlib,json,time,unicodedata
from collections import Counter
from pathlib import Path
import torch
from trid.runtime import load_run,generate

def normalized(s):
    return ''.join(unicodedata.normalize('NFKC',s).split()).rstrip('.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    p.add_argument('--data',default='data/korean-rag-curriculum-v1');p.add_argument('--split',choices=['valid','test'],default='test');a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise ValueError('Use a new output file')
    root=Path(a.data);test=root/f'{a.split}.jsonl';provenance=json.loads((root/'provenance.json').read_text())
    digest=hashlib.sha256(test.read_bytes()).hexdigest()
    if digest!=provenance['hashes'][a.split]:raise ValueError('Evaluation data changed')
    weights=Path(a.run)/'model.safetensors';model_digest=hashlib.sha256(weights.read_bytes()).hexdigest()
    torch.set_num_threads(4);model,tok,device=load_run(a.run,'cpu')
    results=[];counts=Counter();passes=Counter()
    for r in map(json.loads,test.read_text().splitlines()):
        if r['group']!='rag':continue
        prediction=generate(model,tok,r['prompt'],device,max_tokens=64)
        passed=normalized(prediction['answer'])==normalized(r['answer'])
        counts[r['task']]+=1;passes[r['task']]+=passed
        results.append({**r,'generated':prediction,'exact_answer_and_source':passed})
    if hashlib.sha256(weights.read_bytes()).hexdigest()!=model_digest:raise ValueError('Model changed')
    summary={k:{'pass':passes[k],'total':n} for k,n in sorted(counts.items())}
    report={'run':a.run,'split':a.split,'model_sha256':model_digest,'test_sha256':digest,'evaluated_at_unix':time.time(),
      'protocol':'Greedy CPU generation; provided evidence only; exact fact+source after whitespace/NFKC normalization. No retriever, external model, answer substitution. Shared synthetic templates; test has held-out RAG values and phrasing. Not independent real-document benchmark.',
      'summary':summary,'total_pass':sum(passes.values()),'total':sum(counts.values()),'results':results}
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({k:report[k] for k in ['summary','total_pass','total']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
