"""Korean evidence-reading curriculum v3 = v2 + every remaining eligible KLUE MRC train QA.

v2 (scripts/prepare_rag_curriculum_v2.py) capped real-text rows at --klue-train 6000; the extractive reader's
validation failures concentrate on rag_klue/rag_klue_missing while synthetic templates are near perfect, so v3 adds
the rest of the eligible train-split KLUE QAs and changes nothing else:
- train.jsonl: v2 train lines verbatim (same order, same bytes) followed by the extra KLUE rows.
- valid.jsonl, base-check.jsonl (and test.jsonl if present): byte-identical copies of v2.
- Extra rows reuse the v2 KLUE pipeline (klue_candidates/klue_emit): dev-overlap exclusion, markup skip, span
  validity, NFC, context-length fit, 2-passage distractors drawn from the train-split pool only, same id/task/
  group/source fields. Candidates already present in v2 (by source_guid, train or valid) are excluded.
- RNG: random.Random(SEED) with SEED=300326, independent of v2's 271030 so v2 rows are untouched and the extra
  rows' distractor/citation choices are reproducible.
- weight: extra rows copy the v2 per-row weight of their task (sampling masses are not renormalised; the
  extractive trainer ignores weight).
"""
import argparse,json,random,shutil
from collections import Counter
from pathlib import Path
from tokenizers import Tokenizer
import prepare_rag_curriculum_v2 as v2
from prepare_rag_curriculum_v2 import klue_candidates,klue_emit,fits,norm,digest,sha256

SEED=300326
V2=Path('data/korean-rag-curriculum-v2')
COPIED=['valid.jsonl','test.jsonl','base-check.jsonl','KLUE-License.md','ATTRIBUTION.txt']

def prompt_key(x):return json.dumps([x.get('history',[]),x['prompt']],ensure_ascii=False,sort_keys=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',default=str(V2));p.add_argument('--output',default='data/korean-rag-curriculum-v3')
    p.add_argument('--eval-source',default='data/korean-rag-eval-v1-source',help='authored held-out items; asserted absent from training')
    p.add_argument('--klue-extra',type=int,default=10**9,help='cap on extra KLUE rows (default: all eligible)')
    a=p.parse_args();base=Path(a.base);out=Path(a.output)
    if out.exists():raise ValueError('Use a new output directory')
    base_prov=json.loads((base/'provenance.json').read_text())
    for name,expected in base_prov['hashes'].items():  # keys: train, valid, base-check (and test if present)
        if sha256(base/f'{name}.jsonl')!=expected:raise ValueError(f'base {name} changed')
    train_bytes=(base/'train.jsonl').read_bytes()
    train=[json.loads(l) for l in train_bytes.split(b'\n') if l];valid=[json.loads(l) for l in (base/'valid.jsonl').read_text().splitlines()]
    seen_guid={x['source_guid'] for x in train+valid if x.get('source_guid')}
    weights={x['task']:x['weight'] for x in train if x['group']=='rag_real'}
    before=Counter(x['task'] for x in train if x['group']=='rag_real')

    r=random.Random(SEED);tok=Tokenizer.from_file(v2.TOKENIZER);counts=Counter()
    if sha256(v2.TOKENIZER)!=base_prov['tokenizer_sha256']:raise ValueError('tokenizer changed since v2')
    items,pool,klue_prov=klue_candidates(counts)
    if klue_prov['revision']!=base_prov['klue']['source']['revision']:raise ValueError('KLUE revision differs from v2')
    candidates=[it for it in items['train'] if it[3]['guid'] not in seen_guid]
    counts['klue_candidates_train']=len(items['train']);counts['klue_already_in_v2']=len(items['train'])-len(candidates)
    extra=klue_emit(tok,r,'train',candidates,pool['train'],a.klue_extra,counts)
    for x in extra:x['weight']=weights[x['task']]

    # Leak checks: no extra prompt/question/passage may touch a valid row; no held-out eval text anywhere.
    valid_keys={prompt_key(x) for x in valid};valid_blob='\n'.join(x['prompt'] for x in valid)
    valid_ctx={digest(c) for _,c in pool['valid']}|{digest(x['prompt']) for x in valid}
    valid_q={norm(qa['question']) for *_,qa in items['valid']};by_guid={it[3]['guid']:it for it in candidates}
    for x in extra:
        if prompt_key(x) in valid_keys:raise ValueError(f'train/valid prompt overlap: {x["id"]}')
        _,ctx,_,qa=by_guid[x['source_guid']];q=norm(qa['question'])
        if ctx in valid_blob or digest(ctx) in valid_ctx:raise ValueError(f'extra passage occurs in valid: {x["id"]}')
        if q in valid_blob or q in valid_q:raise ValueError(f'extra question occurs in valid: {x["id"]}')
    guids=[x['source_guid'] for x in extra]
    if len(set(guids))!=len(guids) or set(guids)&seen_guid:raise ValueError('duplicate source_guid')
    held=[json.loads(l) for f in sorted(Path(a.eval_source).glob('*.jsonl')) for l in f.read_text().splitlines() if l.strip()]
    held_text={norm(e['text']) for h in held for e in h['evidence']}|{norm(h['question']) for h in held}
    for x in extra:
        if any(t in x['prompt'] for t in held_text):raise ValueError(f'held-out text leaked into train: {x["id"]}')
        if not fits(tok,x):raise ValueError(f'train row exceeds context: {x["id"]}')

    out.mkdir(parents=True)
    (out/'train.jsonl').write_bytes(train_bytes+''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in extra).encode())
    for name in COPIED:
        if (base/name).exists():shutil.copyfile(base/name,out/name)
    for name,expected in base_prov['hashes'].items():
        if name!='train' and sha256(out/f'{name}.jsonl')!=expected:raise ValueError(f'{name} copy mismatch')
    rows=train+extra;after=Counter(x['task'] for x in rows if x['group']=='rag_real')
    splits={'train':rows,'valid':valid}
    if (out/'test.jsonl').exists():splits['test']=[json.loads(l) for l in (out/'test.jsonl').read_text().splitlines()]
    provenance={'purpose':'Korean evidence-use curriculum v3: v2 plus all remaining eligible KLUE MRC train QAs (real-text rows were the validation failure mode)',
      'changes_from_v2':__doc__.strip(),'seed':SEED,'base':str(base),'base_hashes':base_prov['hashes'],'base_rows':base_prov['rows'],
      'tokenizer':v2.TOKENIZER,'tokenizer_sha256':sha256(v2.TOKENIZER),'context_length':v2.CONTEXT,'output_reserve_tokens':v2.OUTPUT_RESERVE,
      'sampling_mass':base_prov['sampling_mass'],'weight_policy':'v2 weights kept verbatim; extra rows copy the v2 per-row weight of their task',
      'rows':{s:len(d) for s,d in splits.items()},'extra_rows':len(extra),'extra_tasks':dict(Counter(x['task'] for x in extra)),
      'klue_real_train_before':dict(before),'klue_real_train_after':dict(after),
      'tasks':{s:dict(Counter(x['task'] for x in d)) for s,d in splits.items()},'counts':dict(counts),
      'held_out_check':f'No passage or question text from {a.eval_source} ({len(held)} items) occurs in any extra training prompt; v2 rows already checked.',
      'valid_check':'valid.jsonl byte-identical to v2; no extra prompt, passage or question occurs in any valid row or valid-split KLUE candidate.',
      'klue':{'source':klue_prov,'policy':base_prov['klue']['policy']+' Extra rows: train-split candidates not in v2 by source_guid, distractors from the train-split pool only.'},
      'limitations':base_prov['limitations'],
      'hashes':{s:sha256(out/f'{s}.jsonl') for s in splits}}
    provenance['hashes']['base-check']=sha256(out/'base-check.jsonl')
    (out/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
    print(json.dumps({k:provenance[k] for k in ['rows','extra_rows','extra_tasks','klue_real_train_before','klue_real_train_after','counts','hashes']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
