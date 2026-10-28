"""Reproduce truthful missing-field augmentation without reading any test rows."""
import argparse,json,random
from pathlib import Path
from trid.corpus import sha256,write_json

NOTES={
    'institution':['기관명은 이 문서에 기재되어 있지 않다.','기관명: 미기재','이 자료에는 기관의 이름이 없다.'],
    'department':['담당 부서의 이름은 명시되지 않았다.','담당부서: 정보 없음'],
    'overseas_food':['해외 출장 식비에 관한 금액은 이 문서에 없다.'],
    'lodging':['국내 숙박비의 지급액은 기재되어 있지 않다.'],
    'overseas_lodging':['해외 숙박비에 대한 금액 정보는 제공되지 않았다.'],
}
LABELS={'institution':'기관명','department':'담당 부서','overseas_food':'해외 식비','lodging':'국내 숙박비','overseas_lodging':'해외 숙박비'}


def build(source,split):
    rows=[json.loads(x) for x in (source/f'{split}.jsonl').read_text().splitlines()]
    rng=random.Random(2940 if split=='train' else 2941);added=[]
    for i in range(0,len(rows),8):
        group=rows[i:i+8]
        assert len(group)==8 and len({r['group'] for r in group})==1
        absent=[r['field'] for r in group if r['is_impossible'] and r['field'] in NOTES]
        field='institution' if 'institution' in absent else rng.choice(absent)
        note=rng.choice(NOTES[field]) if split=='train' else f'이 문서에는 {LABELS[field]}가 나와 있지 않습니다.'
        if split=='valid' and field=='institution':note='이 문서에는 기관명이 나와 있지 않습니다.'
        for row in group:
            added.append({**row,'id':row['id']+'-missing-note','group':row['group']+'-missing-note','document':row['document']+'\n'+note})
    return rows+added


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='data/korean-short-v2');p.add_argument('--output',default='data/korean-presence-v2')
    p.add_argument('--verify',action='store_true',help='Verify existing data byte-for-byte without overwriting it')
    a=p.parse_args();root=Path(a.output);source=Path(a.source)
    if not a.verify:root.mkdir(parents=True,exist_ok=False)
    meta={'source':str(source),'seeds':{'train':2940,'valid':2941},'notes':NOTES,'splits':{},
          'purpose':'Presence-only continuation; retain originals plus true explicit missing-field statements. No source answers or test rows added.'}
    for split in ['train','valid']:
        rows=build(source,split);content=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows);path=root/f'{split}.jsonl'
        if a.verify:assert path.read_text()==content,f'{split} does not reproduce'
        else:path.write_text(content)
        meta['splits'][split]={'rows':len(rows),'sha256':sha256(path)}
    if not a.verify:write_json(root/'provenance.json',meta)
    print(json.dumps(meta['splits'],indent=2))


if __name__=='__main__':main()
