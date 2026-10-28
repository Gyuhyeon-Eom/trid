"""Frozen fictional Korean diagnostic suite; no gold in prompts/history."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import torch
from trid.corpus import sha256, write_json
from trid.runtime import load_run, generate
from trid.prompts import korean_grounded_prompt


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--data', default='data/korean-eval-v1')
    p.add_argument('--output', required=True)
    a=p.parse_args(); data=Path(a.data); out=Path(a.output)
    if out.exists():raise ValueError('Use a new output file.')
    manifest=json.loads((data/'manifest.json').read_text())
    if sha256(data/'cases.jsonl')!=manifest['sha256']:raise ValueError('Frozen evaluation changed.')
    rows=[json.loads(x) for x in (data/'cases.jsonl').read_text().splitlines()]
    torch.set_num_threads(4)
    model,tok,device=load_run(a.run)
    results=[]
    for case in rows:
        history=list(case.get('initial_history',[]))
        for i,turn in enumerate(case['turns']):
            prompt=korean_grounded_prompt(case['document'],turn['question'])
            supplied=list(history)
            result=generate(model,tok,prompt,device,max_tokens=64,history=supplied)
            results.append({'id':f"{case['id']}-{i+1}",'document':case['document'],**turn,'history_supplied':supplied,'generated':result,'semantic_pass':None})
            history=history[result['history_turns_dropped']:]
            history.append({'user':turn['question'],'assistant':result['answer']})
            print(f"{case['id']}-{i+1}: {result['answer']}",flush=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    write_json(out,{'run':a.run,'weights_sha256':sha256(Path(a.run)/'model.safetensors'),'data_sha256':manifest['sha256'], 'evaluated_at':datetime.now(timezone.utc).isoformat(),'protocol':manifest,'context_length':model.cfg.context_length,'turns':len(results),'turns_with_history_dropped':sum(r['generated']['history_turns_dropped']>0 for r in results),'scoring':'Pending semantic review; references are not exact-match targets.','results':results})

if __name__=='__main__':main()
