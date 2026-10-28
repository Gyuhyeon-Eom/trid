"""Frozen narrow synthetic test, historical T01 diagnostic, and optional broad MRC check."""
import argparse,json
from collections import Counter
from pathlib import Path
from datetime import datetime,timezone
import torch
from trid.corpus import sha256,write_json
from trid.short_reader import load_pipeline,read
from trid.korean_reader import load_reader as load_old,read as read_old,source_parts
from evaluate_korean_mrc import normalize,f1


def score(rows):
    pos=[r for r in rows if not r['is_impossible']];neg=[r for r in rows if r['is_impossible']]
    return {'total':len(rows),'answerable':len(pos),'span_exact_ignoring_presence':sum(r['span_exact'] for r in pos),
            'answerable_exact':sum(r['correct'] for r in pos),'false_abstentions':sum(r['prediction']['abstains'] for r in pos),
            'unanswerable':len(neg),'correct_abstentions':sum(r['prediction']['abstains'] for r in neg),
            'presence_correct':sum(not r['prediction']['abstains'] for r in pos)+sum(r['prediction']['abstains'] for r in neg),
            'overall_correct':sum(r['correct'] for r in rows),
            'answerable_character_f1':sum(r['character_f1'] for r in pos)/max(1,len(pos))}


def assess(cases,fn):
    result=[]
    for case in cases:
        pred=fn(case['document'],case['question']);gold=case.get('references') or ([case['answer']] if not case['is_impossible'] else [])
        exact=any(normalize(pred['answer'])==normalize(g) for g in gold)
        span_exact=any(normalize(pred['best_span_if_answerable'])==normalize(g) for g in gold)
        if not pred['abstains']:
            a,b=pred['source_character_span'];assert 0<=a<b<=len(case['document']) and pred['answer']==case['document'][a:b]
        result.append({**case,'prediction':pred,'correct':pred['abstains'] if case['is_impossible'] else exact and not pred['abstains'],
                       'span_exact':span_exact,'character_f1':None if case['is_impossible'] else max(f1(pred['answer'],g) for g in gold)})
    by_field={field:score([r for r in result if r.get('field')==field])
              for field in sorted({r['field'] for r in result if 'field' in r})}
    # Match wrong selected values to other labeled fields in the same document.
    # This is reporting only: gold fields never enter inference or model selection.
    facts={}
    for case in cases:
        if not case['is_impossible'] and case.get('field'):
            facts.setdefault(case['document'],[]).append((case['field'],normalize(case['answer'])))
    confusion=Counter()
    for r in result:
        if r['correct'] or r['prediction']['abstains'] or not r.get('field'):continue
        pred=normalize(r['prediction']['answer'])
        matched=[field for field,value in facts.get(r['document'],[]) if pred==value and field!=r['field']]
        for field in matched:confusion[f'{r["field"]} -> {field}']+=1
    return {'metrics':score(result),'by_field':by_field,'wrong_field_confusions':dict(confusion),'results':result}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',default='runs/trid-korean-short-v1')
    p.add_argument('--output',default='reports/korean-short-v1-evaluation.json')
    p.add_argument('--data',default='data/korean-short-v1');p.add_argument('--baseline-run')
    p.add_argument('--development-regression',action='store_true',help='Test has already informed curriculum revisions; do not call this an untouched test.')
    p.add_argument('--audit-cases',help='Additional frozen JSON with a cases array; never used for selection')
    a=p.parse_args()
    if Path(a.output).exists():raise ValueError('Use new report path')
    paths=[Path(a.run)/c/'model.safetensors' for c in ['span','presence']];hashes=[sha256(p) for p in paths]
    torch.set_num_threads(4);span,presence,tok,device=load_pipeline(a.run)
    fn=lambda d,q:read(span,presence,tok,d,q,device)
    test_path=Path(a.data)/'test.jsonl';cases=[json.loads(x) for x in test_path.read_text().splitlines()]
    synthetic=assess(cases,fn)
    print('development regression:' if a.development_regression else 'held-out short:',json.dumps(synthetic['metrics']),flush=True)
    fixture=json.loads(Path('reports/korean-span-v1-regression-cases.json').read_text())
    regression=assess(fixture['cases'],fn)
    print('T01:',json.dumps(regression['metrics']),flush=True)
    audit=None;audit_result=None;old_audit=None
    if a.audit_cases:
        audit=json.loads(Path(a.audit_cases).read_text())
        audit_result=assess(audit['cases'],fn)
        print('additional audit:',json.dumps(audit_result['metrics']),flush=True)
    # Separate fixed 197-question historical diagnostic, never mixed with narrow test metrics.
    baseline=json.loads(Path('reports/korean-mrc-v1-reading-only.json').read_text())
    ids={r['guid'] for r in baseline['results']};broad=[]
    for line in Path('data/korean-mrc-v1/valid.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r['source_guid'] not in ids:continue
        d,q=source_parts(r);broad.append({'id':r['source_guid'],'document':d,'question':q,'references':r['answer_variants'], 'answer':r['answer'],'is_impossible':r['is_impossible']})
    assert len(broad)==197
    broad_result=assess(broad,fn)
    print('broad MRC:',json.dumps(broad_result['metrics']),flush=True)
    del span,presence
    if device=='mps':torch.mps.empty_cache()
    if a.baseline_run:
        old_paths=[Path(a.baseline_run)/c/'model.safetensors' for c in ['span','presence']]
        old_weights=[sha256(p) for p in old_paths]
        old_span,old_presence,oldtok,olddevice=load_pipeline(a.baseline_run)
        old_result=assess(cases,lambda d,q:read(old_span,old_presence,oldtok,d,q,olddevice))
        if audit:old_audit=assess(audit['cases'],lambda d,q:read(old_span,old_presence,oldtok,d,q,olddevice))
        assert old_weights==[sha256(p) for p in old_paths]
    else:
        old_weights=sha256('runs/trid-korean-span-v1/reader.safetensors')
        old,oldtok,olddevice=load_old('runs/trid-korean-span-v1')
        old_result=assess(cases,lambda d,q:read_old(old,oldtok,d,q,olddevice))
        if audit:old_audit=assess(audit['cases'],lambda d,q:read_old(old,oldtok,d,q,olddevice))
        assert old_weights==sha256('runs/trid-korean-span-v1/reader.safetensors')
    print('old model on same short test:',json.dumps(old_result['metrics']),flush=True)
    assert hashes==[sha256(p) for p in paths]
    report={'run':a.run,'weights_sha256':dict(zip(['span','presence'],hashes)),
            'evaluated_at':datetime.now(timezone.utc).isoformat(),'test_sha256':sha256(test_path),
            'protocol':'Independent component checkpoints selected on validation only (selection recorded in component training metadata). Fixed argmax presence; no threshold tuning. Frozen synthetic set with disjoint templates/values/document groups. Narrow curriculum, not real-world benchmark. Historical T01 and MRC are development diagnostics, not untouched tests. No official dev test used.',
            'test_status':'development regression: already evaluated before curriculum revision' if a.development_regression else 'first evaluation of this frozen synthetic test',
            'data':a.data,'baseline_run':a.baseline_run or 'runs/trid-korean-span-v1',
            'short_test':synthetic,'previous_model_short_test':old_result,'previous_model_weights_sha256':old_weights,
            't01_regression':regression,'historical_broad_mrc_197':broad_result}
    if audit:
        report['additional_audit']={'cases_path':a.audit_cases,'cases_sha256':sha256(a.audit_cases),
            'protocol':audit.get('protocol'),'candidate':audit_result,'previous_model':old_audit}
    write_json(a.output,report)


if __name__=='__main__':main()
