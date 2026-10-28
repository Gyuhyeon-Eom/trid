"""Evaluate conversation chains using the model's actual preceding answers."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

from trid.runtime import load_run, generate
from dialogue_probe import repetition


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='auto',choices=['auto','cpu','mps'])
    args=p.parse_args()
    out=Path(args.output)
    if out.exists():
        raise SystemExit('Use a new report path.')
    torch.set_num_threads(4)
    model,tok,device=load_run(args.run,args.device)
    conversations=[
        ['안녕','네 이름은 뭐야?','오늘 너무 피곤해','응 좀 쉴게','고마워','잘 가'],
        ['안녕하세요','내 이름은 민수야.','내 이름이 뭐야?','오늘 할 일이 많아','하나씩 해 볼게'],
        ['안녕','배고파','점심 뭐 먹지?','비빔밥 먹을래','고마워'],
    ]
    results=[]
    for questions in conversations:
        history=[]
        turns=[]
        for q in questions:
            result=generate(model,tok,q,device,max_tokens=64,history=history)
            result.update(prompt=q,repeated_span=repetition(result['answer']))
            turns.append(result)
            if result['history_turns_dropped']:
                history=history[result['history_turns_dropped']:]
            history.append({'user':q,'assistant':result['answer']})
            print(json.dumps({'prompt':q,'answer':result['answer']},ensure_ascii=False),flush=True)
        results.append(turns)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'run':args.run,'weights_sha256':hashlib.sha256((Path(args.run)/'model.safetensors').read_bytes()).hexdigest(),
        'protocol':'Known curriculum phrasings; actual generated previous answers; diagnostic, not held-out generalization.',
        'device':device,'conversations':results},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
