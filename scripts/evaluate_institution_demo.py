"""Evaluate synthetic institutional questions with evaluator-supplied evidence.

No retrieval, external API, answer substitution, or training in this evaluator.
Choice mode masks next-token logits to three labels; training depends on the run.
This is not an implementation or evaluation of Jev.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import re

import torch
from trid.cli import default_chat_run
from trid.runtime import load_run, generate
from trid.text import prompt_ids


def make_prompt(case, options=None):
    evidence='\n'.join(f"[{e['id']}] {e['text']}" for e in case['evidence'])
    if options is None:
        instruction='자료로 답하세요. 정보가 없으면 자료없음, 모순이면 규정충돌. 값만 짧게 답하세요.'
        return f"{instruction}\n자료: {evidence}\n질문: {case['question']}"
    choices=' / '.join(f'{label}: {value}' for label,value in zip('ABC',options))
    return f"자료로 정답을 고르세요. A, B, C 중 하나만 답하세요.\n자료: {evidence}\n질문: {case['question']}\n선택지: {choices}"


def normalize(value):
    return re.sub(r'\s+','',value).rstrip('.!。')


@torch.inference_mode()
def choose(model,tok,case,options,device):
    labels=[tok.encode(label).ids for label in 'ABC']
    if any(len(ids)!=1 for ids in labels):
        raise ValueError('Choice labels must each be one tokenizer token.')
    ids=prompt_ids(tok,make_prompt(case,options))
    if len(ids)+1>model.cfg.context_length:
        raise ValueError(f"{case['id']}: choice prompt exceeds context limit")
    logits,_=model(torch.tensor([ids],device=device))
    p=logits[0,-1,[ids[0] for ids in labels]].softmax(-1).cpu()
    index=int(p.argmax())
    return {'answer':options[index],'label':'ABC'[index],
            'uncalibrated_label_probabilities':p.tolist(),'input_tokens':len(ids)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',default=default_chat_run())
    parser.add_argument('--data',default='data/institution-demo-v1/eval.jsonl')
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='auto',choices=['auto','cpu','mps'])
    args=parser.parse_args()
    destination=Path(args.output)
    if destination.exists():
        raise ValueError('Use a new report path; previous evaluations are preserved.')
    rows=[json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    torch.set_num_threads(4)
    model,tok,device=load_run(args.run,args.device)
    for row in rows:
        if len(set(row['options']))!=3 or row['expected'] not in row['options']:
            raise ValueError(f"Invalid answer options: {row['id']}")
        if len(prompt_ids(tok,make_prompt(row)))+48>model.cfg.context_length:
            raise ValueError(f"{row['id']}: generation prompt exceeds context limit")
    groups=defaultdict(lambda:{'total':0,'generation_exact':0,'choice_correct':0,'all_orders_correct':0})
    results=[]
    for row in rows:
        generated=generate(model,tok,make_prompt(row),device,max_tokens=48)
        permutations=[]
        for options in itertools.permutations(row['options']):
            result=choose(model,tok,row,options,device)
            permutations.append({'options':list(options),**result,'correct':result['answer']==row['expected']})
        primary=permutations[0]
        item={**row,'generated':generated,'generation_exact':normalize(generated['answer'])==normalize(row['expected']),
              'choice':primary,'permutations':permutations,
              'all_orders_correct':all(p['correct'] for p in permutations),
              'order_invariant':len({p['answer'] for p in permutations})==1}
        results.append(item)
        g=groups[row['group']];g['total']+=1
        for metric,passed in [('generation_exact',item['generation_exact']),('choice_correct',primary['correct']),('all_orders_correct',item['all_orders_correct'])]:
            g[metric]+=int(passed)
        print(json.dumps({'id':row['id'],'expected':row['expected'],'generated':generated['answer'],'choice':primary['answer'],'all_orders_correct':item['all_orders_correct']},ensure_ascii=False),flush=True)
    pairs={}
    for r in results:
        if r.get('counterfactual_pair'):
            pairs.setdefault(r['counterfactual_pair'],[]).append(r)
    report={'evaluated_at':datetime.now(timezone.utc).isoformat(),'run':args.run,'device':device,
        'weights_sha256':hashlib.sha256((Path(args.run)/'model.safetensors').read_bytes()).hexdigest(),
        'test_sha256':hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        'fictional':True,'protocol':'Oracle evidence supplied directly; no retrieval or access-control evaluation. No training on this set.',
        'choice_method':'Next-token probabilities restricted to single-token A/B/C labels. Training depends on the evaluated run; no probability calibration; not Jev. All six option orders tested.',
        'scoring':'Generation exact value match after whitespace/final-punctuation normalization. Semantic review required for non-matches. Gold answers never passed separately to inference.',
        'cases':len(results),'generation_exact':sum(r['generation_exact'] for r in results),
        'choice_correct':sum(r['choice']['correct'] for r in results),
        'all_orders_correct':sum(r['all_orders_correct'] for r in results),
        'order_invariant':sum(r['order_invariant'] for r in results),
        'choice_accuracy_across_all_orders':sum(p['correct'] for r in results for p in r['permutations'])/(6*len(results)),
        'counterfactual_pairs':{key:{'ids':[r['id'] for r in pair],
           'generation_both_correct':all(r['generation_exact'] for r in pair),
           'choice_both_correct':all(r['choice']['correct'] for r in pair)} for key,pair in pairs.items()},
        'groups':dict(groups),'results':results}
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:report[k] for k in ['cases','generation_exact','choice_correct','all_orders_correct','choice_accuracy_across_all_orders','counterfactual_pairs']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
