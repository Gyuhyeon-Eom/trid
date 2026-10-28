"""Greedy reading diagnostics on validation passages, with explicit abstention."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import torch
from trid.corpus import sha256,write_json
from trid.runtime import generate,load_run
from trid.text import prompt_ids


def normalize(s):
    return re.sub(r'\s+','',s).strip('.!?。')


def f1(pred,answer):
    a,b=Counter(normalize(pred)),Counter(normalize(answer))
    common=sum((a&b).values())
    return 2*common/max(1,sum(a.values())+sum(b.values()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True)
    p.add_argument('--data',default='data/korean-mrc-v1/valid.jsonl')
    p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise ValueError('Use a new report path.')
    weights_path=Path(a.run)/"model.safetensors"
    weights_sha=sha256(weights_path)
    torch.set_num_threads(4);model,tok,device=load_run(a.run)
    if sha256(weights_path)!=weights_sha:
        raise ValueError("Checkpoint changed during load; evaluate a completed, stable run.")
    rows=[json.loads(x) for x in Path(a.data).read_text().splitlines()]
    results=[];excluded=0
    abstentions={normalize(x) for x in ['자료에서 확인할 수 없습니다.','자료에 없습니다.','자료없음','알 수 없습니다.','해당 정보는 자료에 없습니다.']}
    for row in rows:
        # Length filter depends only on model, source and question, never the gold span.
        if len(prompt_ids(tok,row['prompt']))+64>model.cfg.context_length:
            excluded+=1;continue
        result=generate(model,tok,row['prompt'],device,max_tokens=64)
        pred=normalize(result['answer']);abstains=pred in abstentions
        gold=row['answer_variants'] or [row['answer']]
        em=any(pred==normalize(g) for g in gold)
        correct=abstains if row['is_impossible'] else em
        results.append({'guid':row['source_guid'],'is_impossible':row['is_impossible'],'prompt':row['prompt'],'references':gold,'generated':result,'exact_match_or_abstention':correct,'abstains':abstains,'character_f1':None if row['is_impossible'] else max(f1(result['answer'],g) for g in gold)})
    positive=[r for r in results if not r['is_impossible']];negative=[r for r in results if r['is_impossible']]
    if sha256(weights_path)!=weights_sha:
        raise ValueError('Checkpoint changed during evaluation; report discarded.')
    report={'run':a.run,'weights_sha256':weights_sha,'data_sha256':sha256(a.data),'evaluated_at':datetime.now(timezone.utc).isoformat(),'eligible':len(results),'excluded_for_generation_budget':excluded,
        'answerable':len(positive),'answerable_exact':sum(r['exact_match_or_abstention'] for r in positive),'answerable_false_abstentions':sum(r['abstains'] for r in positive),'answerable_mean_character_f1':sum(r['character_f1'] for r in positive)/max(1,len(positive)),
        'unanswerable':len(negative),'unanswerable_correct_abstentions':sum(r['abstains'] for r in negative),
        'protocol':'Internal validation subset derived from official KLUE train, not official benchmark score. Full passages supplied. Output is actual LM generation. Exact span matching can penalize correct explanatory prose; unanswerable uses a fixed set of complete abstention phrases, not substring matching. References never enter model inputs.', 'results':results}
    write_json(out,report)
    print(json.dumps({k:v for k,v in report.items() if k!='results'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
