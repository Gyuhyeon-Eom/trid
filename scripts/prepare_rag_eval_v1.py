"""Held-out Korean evidence-reading evaluation v1. Never used for training or checkpoint selection.

Two parts, same prompt/answer format as the RAG curricula (`trid.prompts.rag_prompt`):
- authored: 160 hand-written items (4 authors x 8 kinds) in data/korean-rag-eval-v1-source,
  written independently of the training generators; fictional institutions and figures.
- klue_dev: real passages from official KLUE MRC v1.1 dev (excluded from all trid training;
  earlier reading diagnostics evaluated on it, so it is not unseen by the developers).
"""
import argparse,hashlib,json,random,re,unicodedata
from collections import Counter
from pathlib import Path
from tokenizers import Tokenizer
from trid.prompts import rag_prompt
from trid.text import encode_example

SOURCE=Path('data/korean-rag-eval-v1-source')
KLUE=Path('data/klue-mrc-source')
TOKENIZER='runs/trid-korean-medium-base-v1/tokenizer.json'
CONTEXT=1024;OUTPUT_RESERVE=64
KIND_COUNTS={'rag_value':8,'rag_name':6,'rag_noise':6,'rag_table':5,'rag_missing':5,'rag_conflict':3,'rag_revision':4,'rag_combine':3}
ABSTAIN={'rag_missing':'확인 불가','rag_conflict':'근거 충돌'}

def sha256(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def norm(t):return unicodedata.normalize('NFC',t).strip()

def check_authored(path,items):
    kinds=Counter(x['task'] for x in items)
    if dict(kinds)!=KIND_COUNTS:raise ValueError(f'{path}: kind counts {dict(kinds)}')
    for x in items:
        if set(x)!={'id','task','domain','evidence','question','answer','values','sources'}:raise ValueError(f'{x.get("id")}: keys')
        ids=[e['id'] for e in x['evidence']]
        if len(set(ids))!=len(ids) or not all(re.fullmatch(r'D[1-9]',i) for i in ids):raise ValueError(f'{x["id"]}: passage ids')
        if any('[D' in e['text'] for e in x['evidence']):raise ValueError(f'{x["id"]}: bracket in passage')
        if sum(len(e['text']) for e in x['evidence'])>550:raise ValueError(f'{x["id"]}: evidence too long')
        text={e['id']:e['text'] for e in x['evidence']}
        if x['task'] in ABSTAIN:
            if x['answer']!=ABSTAIN[x['task']] or x['values'] or x['sources']:raise ValueError(f'{x["id"]}: abstain format')
        else:
            if not x['values'] or len(x['values'])!=len(x['sources']):raise ValueError(f'{x["id"]}: values/sources')
            for v,s in zip(x['values'],x['sources']):
                if v not in text.get(s,''):raise ValueError(f'{x["id"]}: value {v!r} not in {s}')
            if x['answer']!=', '.join(f'{v} [{s}]' for v,s in zip(x['values'],x['sources'])):raise ValueError(f'{x["id"]}: answer format')

def fits(tok,row):
    ids,_=encode_example(tok,row,10**6)
    return len(ids)-len(tok.encode(row['answer']).ids)+OUTPUT_RESERVE<=CONTEXT

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='data/korean-rag-eval-v1')
    p.add_argument('--klue-answerable',type=int,default=100);p.add_argument('--klue-unanswerable',type=int,default=50);a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise ValueError('Use a new output directory')
    tok=Tokenizer.from_file(TOKENIZER);rows=[];files={}
    for path in sorted(SOURCE.glob('authored-*.jsonl')):
        items=[json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        check_authored(path,items);files[path.name]=sha256(path)
        for x in items:
            row={'id':x['id'],'origin':'authored','task':x['task'],'domain':x['domain'],
                 'prompt':rag_prompt([(e['id'],e['text']) for e in x['evidence']],x['question']),
                 'answer':x['answer'],'gold':[[v] for v in x['values']],'sources':x['sources']}
            if not fits(tok,row):raise ValueError(f'{x["id"]} exceeds context')
            rows.append(row)
    if len(files)!=4:raise ValueError('expected authored-A..D')
    prov=json.loads((KLUE/'provenance.json').read_text())
    if sha256(KLUE/'dev.json')!=prov['sha256']['dev.json']:raise ValueError('KLUE dev changed')
    dev=json.loads((KLUE/'dev.json').read_text())['data'];r=random.Random(20261027)
    paragraphs=[(norm(d['title']),norm(p['context']),p['context'],qa) for d in dev for p in d['paragraphs'] for qa in p['qas']]
    r.shuffle(paragraphs);need={'rag_klue':a.klue_answerable,'rag_klue_missing':a.klue_unanswerable};got=Counter();used_ctx=set();skipped=Counter()
    for title,ctx,raw,qa in paragraphs:
        task='rag_klue_missing' if qa['is_impossible'] else 'rag_klue'
        if got[task]>=need[task] or ctx in used_ctx or '[D' in ctx:continue
        gold=[] if qa['is_impossible'] else sorted({norm(x['text']) for x in qa['answers'] if x.get('text') and raw[x['answer_start']:x['answer_start']+len(x['text'])]==x['text'] and norm(x['text']) in ctx})
        if task=='rag_klue' and not gold:skipped['invalid_span']+=1;continue
        sid,did=r.sample([f'D{k}' for k in range(1,10)],2)
        passages=[(sid,ctx)]
        if r.random()<.5:
            other=r.choice(paragraphs)
            if other[0]!=title:passages.append((did,other[1]))
        for ps in ([*passages],passages[:1]):
            r.shuffle(ps)
            row={'id':'klue-'+qa['guid'],'origin':'klue_dev','task':task,'domain':title,'prompt':rag_prompt(ps,norm(qa['question'])),
                 'answer':'확인 불가' if not gold else f'{gold[0]} [{sid}]','gold':[gold] if gold else [],'sources':[sid] if gold else []}
            if fits(tok,row):break
        else:skipped['too_long']+=1;continue
        rows.append(row);got[task]+=1;used_ctx.add(ctx)
    if got!=Counter(need):raise ValueError(f'KLUE sample short: {dict(got)}')
    out.mkdir(parents=True)
    (out/'test.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
    (out/'ATTRIBUTION.txt').write_text('Rows with origin klue_dev derive from KLUE MRC v1.1 dev. KLUE benchmark contributors. https://github.com/KLUE-benchmark/KLUE revision '+prov['revision']+'. CC BY-SA 4.0. Changes: NFC, RAG prompt formatting, optional distractor passage, answer reformatted.\n')
    provenance={'purpose':'Held-out evaluation for RAG SFT comparison. Never train on it or select checkpoints with it.',
      'authored':'160 items hand-written by four separate authoring agents on 2026-10-27 from a fixed schema, without access to the v2 training generator. Fictional institutions/figures. Developer-authored; not an industry benchmark.',
      'klue_dev':'Official KLUE MRC dev; excluded from trid training. Previously used in reading diagnostics, so developers have seen its format.',
      'rows':len(rows),'tasks':dict(Counter(x['task'] for x in rows)),'skipped':dict(skipped),
      'context_length':CONTEXT,'output_reserve_tokens':OUTPUT_RESERVE,'tokenizer_sha256':sha256(TOKENIZER),
      'source_sha256':files,'hashes':{'test':sha256(out/'test.jsonl')}}
    (out/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
    print(json.dumps({k:provenance[k] for k in ['rows','tasks','skipped']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
