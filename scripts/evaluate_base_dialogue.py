"""Generate file-free answers against a frozen diagnostic; never insert reference answers."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
from trid.runtime import load_run,generate


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--data',default='data/base-dialogue-v1');p.add_argument('--suite',choices=['evaluation','base-check'],default='evaluation');p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise ValueError('Evaluation output already exists')
    data=Path(a.data);provenance=json.loads((data/'provenance.json').read_text())
    path=data/f'{a.suite}.jsonl';digest=hashlib.sha256(path.read_bytes()).hexdigest()
    hashes=provenance['file_sha256'] if a.suite=='evaluation' else provenance['hashes']
    if digest!=hashes[a.suite]:raise ValueError('Frozen evaluation changed')
    weights=Path(a.run)/'model.safetensors'
    model_digest=hashlib.sha256(weights.read_bytes()).hexdigest()
    torch.set_num_threads(4);model,tok,device=load_run(a.run,a.device)
    results=[]
    for case in map(json.loads,path.read_text().splitlines()):
        history=[]
        for i,turn in enumerate(case.get('turns',[case])):
            prompt=turn['prompt']
            try:result=generate(model,tok,prompt,device,max_tokens=112,history=history)
            except ValueError as error:result={'answer':'','error':str(error),'history_turns_dropped':0}
            results.append({'id':f"{case['id']}-{i+1}",'group':case.get('group','base_check'),'prompt':prompt,'reference':turn['reference'],'history_supplied':list(history),'generated':result,'semantic_pass':None})
            history=history[result['history_turns_dropped']:]
            if result['answer']:history.append({'user':prompt,'assistant':result['answer']})
            print(f"{results[-1]['id']} {prompt}\n  {result['answer']}",flush=True)
    if hashlib.sha256(weights.read_bytes()).hexdigest()!=model_digest:
        raise ValueError('Model weights changed during evaluation; evaluate a stable checkpoint')
    report={'run':a.run,'model_sha256':model_digest,'evaluation_sha256':digest,'parameters':model.parameter_count(),'device':device,'evaluated_at_unix':time.time(),'protocol':'Raw user question plus actual prior model outputs. No document, retrieval, lookup, canned answer replacement, or reference in the input. Greedy generation. Semantic review pending. Same-author small diagnostic, not independent benchmark.','results':results}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
