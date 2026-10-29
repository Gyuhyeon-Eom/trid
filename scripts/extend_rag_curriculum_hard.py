"""Korean evidence-reading curriculum v4 = v3 train rows + hard TRAIN-only rows; valid.jsonl byte-identical to v2.

Why: v2/v3 synthetic rag_missing rows are solvable by a lexical shortcut (asked label absent from the document <=>
missing), and KLUE distractors are easy other-title paragraphs present in only half the rows. v4 keeps every v3
train line verbatim (same order, same bytes) and appends:
1. rag_missing_hard (synthetic, answer 확인 불가): the question's entity/label IS in the document but the asked
   attribute is absent. `hard_mode` names the construction:
   - qualifier: document states the value for one explicit qualifier (2025년/상반기/1차 ...), the question asks the
     neighbouring one (2026년/하반기/2차); the asked qualifier token never occurs in any passage (string-checked).
   - relation: a person field is present, the question asks that person's 전임/후임/직전 holder; no relational word
     occurs in any passage (string-checked).
   - placeholder: the asked label is present with a non-value ("수강료: 별도 공지"); no value was ever generated for it
     (by construction) and no sibling value of the same kind leaks a candidate.
   - same_type: the asked field is absent while a field of the same value kind (another amount, date ...) is
     present; the asked label does not occur in any passage (string-checked).
   rag_value_hard (answer `<value> [Dn]`) are the contrastive positives of the same constructions (matching
   qualifier, 전임 holder stated, placeholder on a sibling field, asked same-kind field present) so that none of
   the new surface cues (a year in the question, 별도 공지 in the document) becomes a shortcut to 확인 불가.
2. rag_klue_hard: train-split answerable KLUE QAs re-emitted with a HARD distractor (always 2 passages): a SAME-title
   other paragraph when the title has one (KLUE MRC train has one paragraph for most titles), otherwise the other-title
   paragraph that shares the most idf-weighted character bigrams with title+question+passage head (LexicalIndex); the exact
   answer string is absent from the distractor (also whitespace-insensitively). rag_klue_missing_hard: the same
   question paired with that hard distractor alone (optional random other-title passage), answer 확인 불가; a string
   check only, so a paraphrased answer in the distractor is a known label risk. `distractor` records same_title /
   lexical. Ids are source_guid based with a distinct prefix so they stay unique next to v3's train-klue-<guid>.
3. rag_tydi: TyDi QA v1.1 GoldP (secondary_task) Korean TRAIN rows (data/tydiqa-ko-source, Apache-2.0) in the
   same prompt format: residual <b>/<i> tags stripped, NFC, answer must be an exact substring of the passage,
   1 or 2 passages (other-title TyDi Korean passage as distractor 50%, answer string absent from it). The TyDi
   validation split is never read here; it is reserved for a later held-out evaluation.
valid.jsonl, base-check.jsonl (and test.jsonl if present), KLUE-License.md are byte-identical copies of v3 (= v2).
RNG: random.Random(SEED), SEED=411029, independent of v2 (271030) and v3 (300326). Weights: v2 per-row weight of
the task each new task mirrors (rag_missing, rag_value, rag_klue, rag_klue_missing); the extractive trainer
ignores weight.
"""
import argparse,json,math,random,re,shutil
from collections import Counter,defaultdict
from pathlib import Path
from tokenizers import Tokenizer
import prepare_rag_curriculum_v2 as v2
from prepare_rag_curriculum_v2 import DOMAINS,make_doc,label_question,fresh,josa,klue_candidates,fits,norm,digest,sha256
from trid.text import SPECIAL
from trid.prompts import rag_prompt,rag_passages
from trid.extractive import encode as extractive_encode

SEED=411029
V3=Path('data/korean-rag-curriculum-v3')
TYDI=Path('data/tydiqa-ko-source')
COPIED=['valid.jsonl','test.jsonl','base-check.jsonl','KLUE-License.md']
HARD_MODES=['qualifier','relation','placeholder','same_type']
QUALIFIER_PAIRS=[('{y}년','{y2}년'),('{y}년도','{y2}년도'),('상반기','하반기'),('하반기','상반기'),('1차','2차'),('2차','1차'),('1학기','2학기'),('2학기','1학기')]
YEARS=[2024,2025,2026,2027]
RELATION_WORDS=['전임','후임','직전','이전','전 ','현 ','현재','후속','차기','다음']
RELATIONS=[('전임',['전임자','전임 {a}','직전 {a}','이전 {a}']),('후임',['후임자','후임 {a}','다음 {a}','차기 {a}'])]
PLACEHOLDERS=['별도 공지','추후 공지','추후 안내','미정','홈페이지 참조','별도 안내 예정','담당 부서 문의','추후 확정']
TYDI_SOURCE='TyDi QA v1.1 GoldP (secondary_task) train, Korean (Apache-2.0), reformatted'
TAG=re.compile(r'</?[bi](\s[^>]*)?>')

def prompt_key(x):return json.dumps([x.get('history',[]),x['prompt']],ensure_ascii=False,sort_keys=True)
def squash(t):return re.sub(r'\s+','',t)

# ---------------------------------------------------------------- 1. synthetic hard rows
def dedupe(fs):return list({f[0]:f for f in fs}.values())

def hard_synthetic(r,mode,positive,i):
    """One rag_missing_hard (positive=False) or rag_value_hard (positive=True) row, or None when a string check fails
    (callers retry). Every construction keeps the questioned entity/label inside the document."""
    domain=r.choice(list(DOMAINS));_,quals,fields=DOMAINS[domain]
    ids=r.sample([f'D{k}' for k in range(1,10)],3)
    target_q=r.choice(quals)
    n=r.randint(3,min(5,len(fields)));chosen=r.sample(fields,n);target=chosen[0]
    used=set()
    def vals(fs):
        out={}
        for f in fs:out[f[0]]=fresh(r,f[1],used);used.add(out[f[0]])
        return out
    values=vals(chosen)
    def shuffled(xs):xs=list(xs);r.shuffle(xs);return xs
    def ask(field,prefix):return label_question(r,(prefix+' ' if prefix else '')+r.choice(field[2]),field[1])
    passages=[];answer=None;checks=[]
    if mode=='qualifier':
        doc_tpl,ask_tpl=r.choice(QUALIFIER_PAIRS)
        y=r.choice(YEARS);y2=r.choice([v for v in YEARS if v!=y and abs(v-y)<=2])
        doc_t=doc_tpl.format(y=y,y2=y2);ask_t=ask_tpl.format(y=y,y2=y2)
        absent_token=str(y2) if '{y2}' in ask_tpl else ask_t  # year variants: the asked year digits never occur at all
        with_program=r.random()<.7
        passages=[(ids[0],make_doc(r,domain,f'{doc_t} {target_q}' if with_program else doc_t,shuffled(chosen),values))]
        asked=doc_t if positive else ask_t
        q=ask(target,f'{asked} {target_q}' if with_program else asked)
        doc=passages[0][1]
        if positive:answer=f'{values[target[0]]} [{ids[0]}]'
        else:checks.append(absent_token not in doc and squash(absent_token) not in squash(doc))
        checks.append(doc_t in doc and asked in q)
    elif mode=='relation':
        people=[f for f in fields if f[1]=='person']
        if not people:return None
        target=r.choice(people)
        rest=[f for f in chosen if f!=target][:3]
        rel,aliases=r.choice(RELATIONS)
        cur=values.get(target[0]) or fresh(r,'person',used);used.add(cur)
        facts={target[0]:cur,**{f[0]:values[f[0]] for f in rest}}
        doc_fields=[target]+rest
        if positive:
            prev=fresh(r,'person',used);used.add(prev)
            rel_label=r.choice([f'{rel} {target[0]}',f'{rel}자'])
            facts[rel_label]=prev;doc_fields=doc_fields+[(rel_label,'person',[])]
        passages=[(ids[0],make_doc(r,domain,target_q,shuffled(doc_fields),facts))]
        alias=r.choice(aliases).format(a=r.choice(target[2]))
        subject=r.choice([f'{target_q} {target[0]}',target[0],f'{target[0]} {cur}',cur])
        q=r.choice([f'{subject}의 {josa(alias,"은/는")} 누구야?',f'{subject} {josa(alias,"이/가")} 누구인가요?',
                    f'{subject}의 {josa(alias,"을/를")} 알려줘.',f'{subject}의 {alias} 이름을 알려주세요.'])
        doc=passages[0][1]
        if positive:answer=f'{prev} [{ids[0]}]';checks.append(rel in doc and prev in doc)
        else:checks.append(not any(w in doc for w in RELATION_WORDS))
        checks.append(cur in doc and target[0] in doc)
    elif mode=='placeholder':
        ph=r.choice(PLACEHOLDERS)
        blank=chosen[1] if positive else target  # field that carries the placeholder instead of a value
        facts={**values,blank[0]:ph}
        passages=[(ids[0],make_doc(r,domain,target_q,shuffled(chosen),facts))]
        q=ask(target,target_q if r.random()<.6 else '')
        doc=passages[0][1]
        checks.append(ph in doc and blank[0] in doc and values[blank[0]] not in doc)
        if positive:answer=f'{values[target[0]]} [{ids[0]}]';checks.append(values[target[0]] in doc)
        if r.random()<.3:  # unrelated second document keeps the row from being a one-passage template
            d2=r.choice([d for d in DOMAINS if d!=domain]);f2=r.sample(DOMAINS[d2][2],3)
            passages.append((ids[1],make_doc(r,d2,r.choice(DOMAINS[d2][1]),f2,vals(f2))));passages=shuffled(passages)
    elif mode=='same_type':
        kind=target[1].split(':')[0]
        same=[f for f in fields if f!=target and f[1].split(':')[0]==kind]
        if not same:return None
        sibling=r.choice(same)
        doc_fields=dedupe(([target] if positive else [])+[sibling]+chosen[1:])[:5]
        facts={**{k:v for k,v in values.items() if positive or k!=target[0]},**vals([f for f in doc_fields if f[0] not in values])}
        passages=[(ids[0],make_doc(r,domain,target_q,shuffled(doc_fields),facts))]
        asked=r.choice([target,sibling]) if positive else target
        q=ask(asked,target_q if r.random()<.6 else '')
        doc=passages[0][1]
        checks.append(sibling[0] in doc and facts[sibling[0]] in doc)
        if positive:answer=f'{facts[asked[0]]} [{ids[0]}]';checks.append(facts[target[0]] in doc)
        else:checks.append(target[0] not in doc and all(a not in doc for a in target[2]) and values[target[0]] not in doc)
    if not all(checks):return None
    if any('[D' in t or any(s in t for s in SPECIAL) for _,t in passages):return None
    return {'id':f'train-synhard-{"pos" if positive else "neg"}-{mode}-{i:05d}','prompt':rag_prompt(passages,q),
            'answer':answer if positive else '확인 불가','task':'rag_value_hard' if positive else 'rag_missing_hard',
            'group':'rag','domain':domain,'hard_mode':mode,'source':v2.SOURCE+'; hard contrastive constructions (v4)'}

def synthetic_rows(r,tok,n_missing,n_value,seen,counts):
    out=[]
    for positive,total in [(False,n_missing),(True,n_value)]:
        per=[total//len(HARD_MODES)+(1 if k<total%len(HARD_MODES) else 0) for k in range(len(HARD_MODES))]
        for mode,n in zip(HARD_MODES,per):
            made=tries=0
            while made<n:
                tries+=1
                if tries>n*20:raise RuntimeError(f'cannot generate {mode} positive={positive}')
                try:row=hard_synthetic(r,mode,positive,made)
                except RuntimeError:counts['synhard_value_collision']+=1;continue
                if row is None:counts[f'synhard_rejected_{mode}']+=1;continue
                if row['prompt'] in seen:counts['synhard_duplicate']+=1;continue
                if not fits(tok,row):counts['synhard_too_long']+=1;continue
                seen.add(row['prompt']);out.append(row);made+=1
    return out

# ---------------------------------------------------------------- 2. KLUE hard rows
def answer_of(ctx,raw,qa,counts):
    """Normalised answer text when the QA is answerable with a valid span inside the normalised passage, else None."""
    if qa['is_impossible']:return None
    ok=[a for a in qa['answers'] if a.get('text') and raw[a['answer_start']:a['answer_start']+len(a['text'])]==a['text']]
    if not ok:counts['klue_invalid_span']+=1;return None
    a=norm(ok[0]['text'])
    if a not in ctx:counts['klue_answer_not_in_normalized']+=1;return None
    return a

def shingles(t):
    t=squash(t);return {t[i:i+2] for i in range(len(t)-1)}

class LexicalIndex:
    """Character-bigram index over pool paragraphs; top(text) ranks paragraphs by idf-weighted shared bigrams (bigrams in
    more than 2% of the pool are ignored). Used to pick hard distractors when a title has no second paragraph."""
    def __init__(self,pool):
        self.pool=pool;post=defaultdict(list)
        for k,(_,c) in enumerate(pool):
            for g in shingles(c):post[g].append(k)
        n=len(pool);self.post={g:p for g,p in post.items() if len(p)<=n*.02};self.idf={g:math.log(n/len(p)) for g,p in self.post.items()}
    def top(self,text,k):
        score=Counter()
        for g in shingles(text):
            w=self.idf.get(g)
            if w:
                for i in self.post[g]:score[i]+=w
        return [self.pool[i] for i,_ in score.most_common(k)]

def klue_hard_rows(tok,r,items,pool,cap_hard,cap_missing,counts):
    """rag_klue_hard: gold passage + hard distractor (same-title paragraph when the title has one, else the lexically most
    similar other-title paragraph); rag_klue_missing_hard: the hard distractor alone (optional random other-title
    passage) with answer 확인 불가. The exact answer string is absent from every distractor (also whitespace-insensitively)."""
    by_title=defaultdict(list)
    for title,ctx in pool:by_title[title].append(ctx)
    index=LexicalIndex(pool)
    answerable=[]
    for title,ctx,raw,qa in items:
        a=answer_of(ctx,raw,qa,counts)
        if a is not None:answerable.append((title,ctx,qa,a))
    counts['klue_hard_answerable_candidates']=len(answerable)
    r.shuffle(answerable)
    hard=[];missing=[]
    def clean(c,a):return a not in c and squash(a) not in squash(c)
    for title,ctx,qa,a in answerable:
        if len(hard)>=cap_hard and len(missing)>=cap_missing:break
        q=norm(qa['question'])
        same=[c for c in by_title[title] if c!=ctx and clean(c,a)];r.shuffle(same)
        similar=[c for t,c in index.top(title+' '+q+' '+ctx[:120],12) if t!=title and c!=ctx and clean(c,a)]
        cands=[(c,'same_title') for c in same]+[(c,'lexical') for c in similar]
        if not cands:counts['klue_hard_no_distractor']+=1;continue
        src='KLUE MRC v1.1 train (CC BY-SA 4.0), reformatted; hard distractor (v4)'
        if len(hard)<cap_hard:
            for c,kind in cands[:8]:
                ids=r.sample([f'D{k}' for k in range(1,10)],2);ps=[(ids[0],ctx),(ids[1],c)];r.shuffle(ps)
                row={'id':f'train-kluehard-{qa["guid"]}','prompt':rag_prompt(ps,q),'answer':f'{a} [{ids[0]}]','task':'rag_klue_hard','group':'rag_real',
                     'source':src,'source_guid':qa['guid'],'source_title':title,'distractor':kind}
                if fits(tok,row):hard.append(row);counts[f'klue_hard_distractor_{kind}']+=1;break
            else:counts['klue_hard_too_long']+=1
        if len(missing)<cap_missing:
            c,kind=cands[0];ids=r.sample([f'D{k}' for k in range(1,10)],2);ps=[(ids[0],c)]
            if r.random()<.5:
                other=r.choice(pool)
                if other[0]!=title and clean(other[1],a):ps.append((ids[1],other[1]))
            def build(ps):
                ps=list(ps);r.shuffle(ps)
                return {'id':f'train-kluemh-{qa["guid"]}','prompt':rag_prompt(ps,q),'answer':'확인 불가','task':'rag_klue_missing_hard','group':'rag_real',
                        'source':src,'source_guid':qa['guid'],'source_title':title,'distractor':kind}
            row=build(ps)
            if not fits(tok,row):row=build(ps[:1])
            if fits(tok,row):missing.append(row);counts[f'klue_missing_hard_distractor_{kind}']+=1
            else:counts['klue_missing_hard_too_long']+=1
    return hard,missing

# ---------------------------------------------------------------- 3. TyDi QA GoldP Korean
def tydi_rows(tok,r,counts):
    import pyarrow.parquet as pq
    prov=json.loads((TYDI/'provenance.json').read_text())
    name='train-00000-of-00001.parquet'
    if sha256(TYDI/name)!=prov['sha256'][name]:raise ValueError('TyDi source changed')
    raw=[x for x in pq.read_table(TYDI/name).to_pylist() if x['id'].startswith('korean')]
    counts['tydi_korean_train']=len(raw)
    cands=[];seen=set()
    for x in raw:
        ctx=norm(TAG.sub('',x['context']));q=norm(x['question']);title=norm(x['title'])
        a=norm(TAG.sub('',x['answers']['text'][0])) if x['answers']['text'] else ''
        if not a or a not in ctx:counts['tydi_answer_not_substring']+=1;continue
        if '[D' in ctx or any(s in ctx for s in SPECIAL) or '[D' in q:counts['tydi_skipped_markup']+=1;continue
        if (q,ctx) in seen:counts['tydi_duplicate']+=1;continue
        seen.add((q,ctx));cands.append((x['id'],title,ctx,q,a))
    pool=[(t,c) for _,t,c,_,_ in cands]
    out=[]
    for sid,title,ctx,q,a in cands:
        ids=r.sample([f'D{k}' for k in range(1,10)],2);ps=[(ids[0],ctx)]
        if r.random()<.5:
            other=r.choice(pool)
            if other[0]!=title and a not in other[1]:ps.append((ids[1],other[1]))
        def build(ps):
            ps=list(ps);r.shuffle(ps)
            return {'id':f'train-tydi-{sid}','prompt':rag_prompt(ps,q),'answer':f'{a} [{ids[0]}]','task':'rag_tydi','group':'rag_real',
                    'source':TYDI_SOURCE,'source_guid':sid,'source_title':title}
        row=build(ps)
        if not fits(tok,row):row=build(ps[:1])
        if fits(tok,row):out.append(row)
        else:counts['tydi_too_long']+=1
    return out,prov

# ---------------------------------------------------------------- main
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',default=str(V3));p.add_argument('--output',default='data/korean-rag-curriculum-v4')
    p.add_argument('--v2',default='data/korean-rag-curriculum-v2',help='valid.jsonl must stay byte-identical to this')
    p.add_argument('--eval-source',default='data/korean-rag-eval-v1-source',help='authored held-out items; asserted absent from training')
    p.add_argument('--missing-hard',type=int,default=2000);p.add_argument('--value-hard',type=int,default=1000,help='contrastive positives; 0 disables')
    p.add_argument('--klue-hard',type=int,default=3000);p.add_argument('--klue-missing-hard',type=int,default=1500)
    p.add_argument('--no-tydi',action='store_true')
    a=p.parse_args();base=Path(a.base);out=Path(a.output)
    if out.exists():raise ValueError('Use a new output directory')
    base_prov=json.loads((base/'provenance.json').read_text())
    for name,expected in base_prov['hashes'].items():
        if sha256(base/f'{name}.jsonl')!=expected:raise ValueError(f'base {name} changed')
    v2_prov=json.loads((Path(a.v2)/'provenance.json').read_text())
    if base_prov['hashes']['valid']!=v2_prov['hashes']['valid'] or sha256(Path(a.v2)/'valid.jsonl')!=v2_prov['hashes']['valid']:raise ValueError('base valid differs from v2')
    train_bytes=(base/'train.jsonl').read_bytes()
    train=[json.loads(l) for l in train_bytes.split(b'\n') if l];valid=[json.loads(l) for l in (base/'valid.jsonl').read_text().splitlines()]
    weights={x['task']:x['weight'] for x in train}
    mirror={'rag_missing_hard':'rag_missing','rag_value_hard':'rag_value','rag_klue_hard':'rag_klue','rag_klue_missing_hard':'rag_klue_missing','rag_tydi':'rag_klue'}
    before=Counter(x['task'] for x in train if x['group'] in ('rag','rag_real'))
    seen={x['prompt'] for x in train}|{x['prompt'] for x in valid}

    r=random.Random(SEED);tok=Tokenizer.from_file(v2.TOKENIZER);counts=Counter()
    if sha256(v2.TOKENIZER)!=base_prov['tokenizer_sha256']:raise ValueError('tokenizer changed since base')
    synthetic=synthetic_rows(r,tok,a.missing_hard,a.value_hard,seen,counts)
    items,pool,klue_prov=klue_candidates(counts)
    if klue_prov['revision']!=base_prov['klue']['source']['revision']:raise ValueError('KLUE revision differs from base')
    klue_hard,klue_missing=klue_hard_rows(tok,r,items['train'],pool['train'],a.klue_hard,a.klue_missing_hard,counts)
    tydi,tydi_prov=([],None) if a.no_tydi else tydi_rows(tok,r,counts)
    extra=synthetic+klue_hard+klue_missing+tydi
    for x in extra:x['weight']=weights[mirror[x['task']]]

    # Leak checks: no extra prompt/question/passage may touch a valid row; no held-out eval text anywhere; ids unique.
    valid_keys={prompt_key(x) for x in valid};valid_blob='\n'.join(x['prompt'] for x in valid)
    valid_ctx={digest(c) for _,c in pool['valid']}|{digest(x['prompt']) for x in valid}
    valid_q={norm(qa['question']) for *_,qa in items['valid']}
    ids=[x['id'] for x in train+valid+extra if x.get('id')]  # v1 dialogue rows carry no id
    if len(set(ids))!=len(ids):raise ValueError('duplicate id')
    keys=[prompt_key(x) for x in extra]
    if len(set(keys))!=len(keys):raise ValueError('duplicate extra prompt')
    held=[json.loads(l) for f in sorted(Path(a.eval_source).glob('*.jsonl')) for l in f.read_text().splitlines() if l.strip()]
    held_text={norm(e['text']) for h in held for e in h['evidence']}|{norm(h['question']) for h in held}
    for x in extra:
        if prompt_key(x) in valid_keys:raise ValueError(f'train/valid prompt overlap: {x["id"]}')
        if x['group']=='rag_real':
            q=x['prompt'].rsplit('\n질문: ',1)[1];passages=[t for _,t in rag_passages(x['prompt'])]
            for ctx in passages:
                if ctx in valid_blob or digest(ctx) in valid_ctx:raise ValueError(f'extra passage occurs in valid: {x["id"]}')
            if q in valid_blob or q in valid_q:raise ValueError(f'extra question occurs in valid: {x["id"]}')
        if any(t in x['prompt'] for t in held_text):raise ValueError(f'held-out text leaked into train: {x["id"]}')
        if not fits(tok,x):raise ValueError(f'train row exceeds context: {x["id"]}')
    # Diagnostics: what the extractive trainer (scripts/train_extractive.py) will be able to label.
    usable=Counter()
    for x in extra:
        row=extractive_encode(tok,x['prompt'],x['answer'],v2.CONTEXT)
        if row is None:usable[x['task'],'unmapped_answer']+=1
        elif row['kind']==0 and not row['label_decodable']:usable[x['task'],'label_'+(row.get('label_reason') or 'outside_decode_rules')]+=1
        else:usable[x['task'],'usable']+=1

    out.mkdir(parents=True)
    (out/'train.jsonl').write_bytes(train_bytes+''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in extra).encode())
    for name in COPIED:
        if (base/name).exists():shutil.copyfile(base/name,out/name)
    for name,expected in base_prov['hashes'].items():
        if name!='train' and sha256(out/f'{name}.jsonl')!=expected:raise ValueError(f'{name} copy mismatch')
    if sha256(out/'valid.jsonl')!=v2_prov['hashes']['valid']:raise ValueError('valid.jsonl is not byte-identical to v2')
    attribution=(base/'ATTRIBUTION.txt').read_text()
    if tydi:
        shutil.copyfile(TYDI/'LICENSE',out/'TYDIQA-LICENSE.txt')
        attribution+=('\nRows with task rag_tydi derive from TyDi QA v1.1 Gold Passage task (secondary_task) train split, Korean subset. '
                      'Google Research. Source: https://github.com/google-research-datasets/tydiqa (commit '+tydi_prov['repository_commit']+'), '
                      'downloaded as the Hugging Face parquet conversion google-research-datasets/tydiqa@'+tydi_prov['huggingface_revision']+'.\n'
                      'License: Apache-2.0 (TYDIQA-LICENSE.txt). Changes: residual <b>/<i> tags removed, NFC, RAG prompt formatting, optional '
                      'distractor passage from another TyDi Korean title, answer reformatted as `<span> [Dn]`, length filtering. Validation split not used.\n')
    (out/'ATTRIBUTION.txt').write_text(attribution)
    rows=train+extra;after=Counter(x['task'] for x in rows if x['group'] in ('rag','rag_real'))
    splits={'train':rows,'valid':valid}
    if (out/'test.jsonl').exists():splits['test']=[json.loads(l) for l in (out/'test.jsonl').read_text().splitlines()]
    provenance={'purpose':'Korean evidence-use curriculum v4: v3 plus hard contrastive missing rows, same-title KLUE distractors and TyDi QA GoldP Korean passages',
      'changes_from_v3':__doc__.strip(),'seed':SEED,'base':str(base),'base_hashes':base_prov['hashes'],'base_rows':base_prov['rows'],
      'v2_valid_sha256':v2_prov['hashes']['valid'],
      'tokenizer':v2.TOKENIZER,'tokenizer_sha256':sha256(v2.TOKENIZER),'context_length':v2.CONTEXT,'output_reserve_tokens':v2.OUTPUT_RESERVE,
      'sampling_mass':base_prov['sampling_mass'],'weight_policy':'v3 weights kept verbatim; new tasks copy the v2 per-row weight of the task they mirror: '+json.dumps(mirror),
      'arguments':{k:v for k,v in vars(a).items()},
      'rows':{s:len(d) for s,d in splits.items()},'extra_rows':len(extra),'extra_tasks':dict(Counter(x['task'] for x in extra)),
      'extra_hard_modes':{t:dict(Counter(x['hard_mode'] for x in extra if x['task']==t)) for t in ('rag_missing_hard','rag_value_hard')},
      'extra_passage_counts':{t:dict(Counter(len(rag_passages(x['prompt'])) for x in extra if x['task']==t)) for t in Counter(x['task'] for x in extra)},
      'rag_train_before':dict(before),'rag_train_after':dict(after),
      'tasks':{s:dict(Counter(x['task'] for x in d)) for s,d in splits.items()},'counts':dict(counts),
      'extractive_trainer_diagnostic':{f'{t}:{k}':n for (t,k),n in sorted(usable.items())},
      'held_out_check':f'No passage or question text from {a.eval_source} ({len(held)} items) occurs in any extra training prompt; v3 rows already checked.',
      'valid_check':'valid.jsonl byte-identical to v2 and v3 (sha256 asserted); no extra prompt occurs in valid; no extra real-text passage or question occurs in any valid row or valid-split KLUE candidate; ids unique across train+valid.',
      'klue':{'source':klue_prov,'policy':base_prov['klue']['policy']+' v4 hard rows: train-split answerable QAs; distractor/missing paragraphs are same-title train-split paragraphs in which the exact answer string is absent.'},
      'tydi':None if tydi_prov is None else {'source':tydi_prov,'policy':'secondary_task train split, ids starting with korean-; validation split reserved for evaluation and never read by this builder.'},
      'limitations':base_prov['limitations']+' rag_klue_missing_hard labels rest on a string check (the same-title paragraph may paraphrase the answer); TyDi questions were written without seeing the passage, so some answers are loose.',
      'hashes':{s:sha256(out/f'{s}.jsonl') for s in splits}}
    provenance['hashes']['base-check']=sha256(out/'base-check.jsonl')
    (out/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
    print(json.dumps({k:provenance[k] for k in ['rows','extra_rows','extra_tasks','extra_hard_modes','extra_passage_counts','rag_train_before','rag_train_after','counts','extractive_trainer_diagnostic','hashes']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
