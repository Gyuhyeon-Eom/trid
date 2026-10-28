"""Prepare pinned official KLUE MRC as a separate reading curriculum.

Whole contexts only: no gold-answer-based cropping. Public dev is held out.
This is reading supervision, not natural multi-turn conversational training.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import unicodedata
from tokenizers import Tokenizer
from trid.corpus import sha256, write_json
from trid.text import encode_example, SPECIAL
from trid.prompts import korean_mrc_prompt


def norm(text):
    return unicodedata.normalize('NFC', text).strip()


def digest(text):
    return hashlib.sha256(re.sub(r'\s+', '', norm(text)).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source', default='data/klue-mrc-source')
    p.add_argument('--output', default='data/korean-mrc-v1')
    p.add_argument('--tokenizer', default='runs/trid-pretrain-pilot/tokenizer.json')
    p.add_argument('--context-length', type=int, default=512)
    a=p.parse_args(); source=Path(a.source); out=Path(a.output)
    if out.exists():raise ValueError('Use a new output folder.')
    provenance=json.loads((source/'provenance.json').read_text())
    for filename, expected in provenance['sha256'].items():
        if sha256(source/filename)!=expected:raise ValueError('Source changed.')
    train=json.loads((source/'train.json').read_text())['data']
    dev=json.loads((source/'dev.json').read_text())['data']
    dev_titles={norm(r['title']) for r in dev if r.get('title')}
    dev_contexts={digest(p['context']) for r in dev for p in r['paragraphs']}
    tok=Tokenizer.from_file(a.tokenizer)
    counts=Counter(); rows={'train':[], 'valid':[], 'test':[]}; owners={}; seen=set()
    # Union titles sharing identical contexts, so repeated passages cannot cross splits.
    parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    def union(a,b):
        a,b=find(a),find(b)
        parent[max(a,b)]=min(a,b)
    passage_title={}
    eligible=[]
    for record in train:
        title=norm(record.get('title',''))
        if title in dev_titles or any(digest(p['context']) in dev_contexts for p in record['paragraphs']):
            counts['train_documents_excluded_for_official_dev_overlap']+=1
            continue
        key='title:'+title if title else 'context:'+digest(record['paragraphs'][0]['context'])
        find(key)
        for paragraph in record['paragraphs']:
            h=digest(paragraph['context'])
            if h in passage_title:union(key,passage_title[h])
            passage_title[h]=key
        eligible.append((key,record))
    sources=[('local',key,r) for key,r in eligible]+[('official_dev','dev:'+str(i),r) for i,r in enumerate(dev)]
    for origin,key,record in sources:
        split='test' if origin=='official_dev' else ('valid' if int(digest(find(key))[:8],16)%20==0 else 'train')
        for paragraph in record['paragraphs']:
            raw=paragraph['context']; context=norm(raw); context_hash=digest(context)
            if context_hash in owners and owners[context_hash]!=split:raise ValueError('Document leakage.')
            owners[context_hash]=split
            for qa in paragraph['qas']:
                counts[f'{split}_source_questions']+=1
                question=norm(qa['question']); impossible=bool(qa['is_impossible'])
                answers=[x['text'] for x in qa['answers'] if x.get('text')]
                if not impossible:
                    valid_answers=[x for x in qa['answers'] if x.get('text') and raw[x['answer_start']:x['answer_start']+len(x['text'])]==x['text']]
                    if not valid_answers:
                        counts[f'{split}_invalid_answer_span']+=1;continue
                    answer=norm(valid_answers[0]['text'])
                else:answer='자료에서 확인할 수 없습니다.'
                prompt=korean_mrc_prompt(context,question)
                if any(s in prompt or s in answer for s in SPECIAL):
                    counts[f'{split}_reserved_control_token']+=1;continue
                row={'prompt':prompt,'answer':answer,'task':'korean_mrc_unanswerable' if impossible else 'korean_mrc_answerable','weight':1,
                     'source_guid':qa['guid'],'source_title':record['title'],'source':record.get('source'),
                     'context_sha256':context_hash,'original_split':origin,'is_impossible':impossible,
                     'answer_variants':[norm(x) for x in answers] if not impossible else [],
                     'license_reference':'https://github.com/KLUE-benchmark/KLUE/blob/'+provenance['revision']+'/License.md'}
                identity=digest(prompt)
                if identity in seen:
                    counts[f'{split}_duplicate_question']+=1;continue
                try:ids,_=encode_example(tok,row,a.context_length)
                except ValueError:
                    counts[f'{split}_whole_context_too_long']+=1;continue
                row['training_tokens']=len(ids);seen.add(identity);rows[split].append(row)
                counts[f'{split}_{"unanswerable" if impossible else "answerable"}']+=1
    if not rows['train'] or not rows['valid']:raise ValueError('Empty training or validation split.')
    out.mkdir(parents=True)
    for split,items in rows.items():
        with (out/f'{split}.jsonl').open('w') as f:
            for row in items:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    shutil.copy2(source/'License.md',out/'License.md')
    (out/'ATTRIBUTION.txt').write_text('KLUE benchmark contributors. Source: https://github.com/KLUE-benchmark/KLUE\nRevision: '+provenance['revision']+'\nRepository license: CC BY-SA 4.0; original source labels retained per question.\nChanges: NFC, instruction formatting, title/passage grouped split, full-context length filtering and duplicate removal.\nNo commercial-clearance claim for every upstream document or resulting model.\n')
    write_json(out/'provenance.json',{'source':provenance,'counts':dict(counts),'rows':{k:len(v) for k,v in rows.items()},'context_length':a.context_length,'tokenizer_sha256':sha256(a.tokenizer),
        'split_policy':'Official train: 95/5 title and identical-context connected groups. Exclude official dev titles/contexts before split. Official dev retained only as test.',
        'length_policy':'Keep full contexts fitting the budget. No answer-based cropping. This biases toward shorter passages; not the full KLUE benchmark.',
        'limitations':'Reading curriculum, not multi-turn dialogue. Wikipedia-derived passages may overlap historical foundation training; no claim of contamination-free public benchmark performance.',
        'frozen_internal_eval':'data/korean-eval-v1 is never used here.',
        'file_sha256':{s:sha256(out/f'{s}.jsonl') for s in rows}})
    print(json.dumps({'rows':{k:len(v) for k,v in rows.items()},'counts':dict(counts)},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
