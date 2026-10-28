"""Evaluate learned presence and source-span selection, including a paired LM subset."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import torch
from trid.corpus import sha256, write_json
from trid.korean_reader import load_reader, read, source_parts
from evaluate_korean_mrc import normalize, f1


def metrics(rows):
    positive = [r for r in rows if not r['is_impossible']]
    negative = [r for r in rows if r['is_impossible']]
    return {'eligible':len(rows), 'answerable':len(positive),
            'answerable_exact':sum(r['correct'] for r in positive),
            'answerable_false_abstentions':sum(r['prediction']['abstains'] for r in positive),
            'answerable_span_exact_ignoring_presence':sum(r['span_exact'] for r in positive),
            'answerable_mean_character_f1':sum(r['character_f1'] for r in positive)/max(1,len(positive)),
            'unanswerable':len(negative),
            'unanswerable_correct_abstentions':sum(r['prediction']['abstains'] for r in negative)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='runs/trid-korean-span-v1')
    p.add_argument('--data',default='data/korean-mrc-v1/valid.jsonl')
    p.add_argument('--baseline',default='reports/korean-mrc-v1-reading-only.json')
    p.add_argument('--output',default='reports/korean-span-v1-evaluation.json')
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise ValueError('Use a new report path')
    weights=Path(a.run)/'reader.safetensors'; before=sha256(weights)
    torch.set_num_threads(4);model,tok,device=load_reader(a.run)
    if before!=sha256(weights):raise ValueError('Unstable checkpoint')
    baseline=json.loads(Path(a.baseline).read_text())
    if baseline['data_sha256']!=sha256(a.data):raise ValueError('Different evaluation data')
    paired={r['guid']:r for r in baseline['results']}
    results=[]
    for line in Path(a.data).read_text().splitlines():
        row=json.loads(line);document,question=source_parts(row)
        prediction=read(model,tok,document,question,device)
        gold=row['answer_variants'] or [row['answer']]
        exact=any(normalize(prediction['answer'])==normalize(g) for g in gold)
        span_exact=any(normalize(prediction['best_span_if_answerable'])==normalize(g) for g in gold)
        correct=prediction['abstains'] if row['is_impossible'] else exact and not prediction['abstains']
        if row['source_guid'] in paired:
            prior=paired[row['source_guid']]
            assert prior['prompt']==row['prompt'] and prior['references']==gold
        if not prediction['abstains']:
            start,end=prediction['source_character_span']
            assert 0<=start<end<=len(document) and prediction['answer']==document[start:end]
        results.append({'guid':row['source_guid'],'is_impossible':row['is_impossible'],
                        'document':document,'question':question,'references':gold,'prediction':prediction,
                        'correct':bool(correct),'span_exact':bool(span_exact),
                        'character_f1':None if row['is_impossible'] else max(f1(prediction['answer'],g) for g in gold),
                        'paired_with_generative_baseline':row['source_guid'] in paired})
    subset=[r for r in results if r['paired_with_generative_baseline']]
    assert {r['guid'] for r in subset}==set(paired)
    if before!=sha256(weights):raise ValueError('Checkpoint changed; report discarded')
    report={'run':a.run,'weights_sha256':before,'data_sha256':sha256(a.data),
            'evaluated_at':datetime.now(timezone.utc).isoformat(),'all_validation':metrics(results),
            'paired_generative_subset':metrics(subset),'baseline_report':a.baseline,
            'baseline_paired_metrics':{k:baseline[k] for k in ['eligible','answerable','answerable_exact','answerable_false_abstentions','answerable_mean_character_f1','unanswerable','unanswerable_correct_abstentions']},
            'protocol':'Full passages; question before document; no labels in inference; learned 2-class argmax presence, joint ordered source span at most 64 Unicode characters. Probabilities uncalibrated. Same 197 questions as LM baseline plus remaining internal validation questions. Not official benchmark; no official dev test used.',
            'results':results}
    write_json(out,report)
    print(json.dumps({k:v for k,v in report.items() if k!='results'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
