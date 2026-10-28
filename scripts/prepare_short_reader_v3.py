"""Paired field substitutions: identical question, one relevant fact changed."""
import json,random,hashlib,shutil
from pathlib import Path
from tokenizers import Tokenizer
from trid.korean_reader import encode
from trid.corpus import sha256,write_json
from prepare_short_reader_v2 import FIELDS,LABELS,QUESTIONS,VALID_LABELS,VALID_Q

CONFUSABLE=[1,3,3,1,7,6,5,4]
EXTRA_LABELS=['국내 식사비','해외 식사비','국내 숙박 경비','해외 숙박 경비','출장 비용 제출 기한','담당 부서명','규정 소속 기관명','국내 출장 일당']
NOTES=['국내 식비','해외 식비','국내 숙박비','해외 숙박비','정산 기한','담당 부서','기관명','국내 일비']


def main():
    root=Path('data/korean-short-v3');root.mkdir(exist_ok=False)
    parent=Path('data/korean-short-v2');tok=Tokenizer.from_file('runs/trid-korean-short-v2/span/tokenizer.json')
    manifest={'purpose':'Same-question positive/negative pairs; replace queried field with confusable field while preserving other facts. Replay original v2 train/valid. No test rows used in training.',
              'development_tests':'v2 1120 and audit 120 have already been observed; they are regression only.',
              'split_policy':'Disjoint full documents, name/value pools and fact styles for newly authored pairs. Shared eight field concepts.',
              'seeds':{'train':4300,'valid':4301},'splits':{},'confusable_fields':dict(zip(FIELDS,[FIELDS[i] for i in CONFUSABLE]))}
    seen=set();value_sets=[]
    for si,(split,pairs_per_field) in enumerate([('train',300),('valid',40)]):
        rng=random.Random(4300+si)
        rows=[json.loads(line) for line in (parent/f'{split}.jsonl').read_text().splitlines()]
        old_count=len(rows);values_seen=set();maximum=0
        bucket=lambda n:int(hashlib.sha256(str(n).encode()).hexdigest()[:8],16)%10
        numbers=[i for i in range(1,200) if (bucket(i)<7 if split=='train' else bucket(i)==7)]
        stems=['온솔','해담','바른','가온','소담','늘찬','청솔','은빛','새길','다솜','푸른'] if split=='train' else ['별하','아람','윤슬','꽃내']
        departments=['운영지원팀','재무지원팀','회계과','총무팀','인사부','행정지원실','경영지원팀','정산관리과'] if split=='train' else ['기획협력팀','여비관리과','문화지원실','업무지원부']
        for field in range(8):
            for i in range(pairs_per_field):
                nums=rng.sample(numbers,6)
                money=lambda n:f'{n}만원' if rng.random()<.5 else f'{n*1000:,}원'
                values=[money(nums[0]),money(nums[1]),money(nums[2]),money(nums[3]),f'{nums[4]}일',rng.choice(departments),rng.choice(stems)+rng.choice(['문화재단','환경연구원','도시공사','교육진흥원']),money(nums[5])]
                values_seen.update(values)
                rival=CONFUSABLE[field]
                # Zero to four irrelevant facts; neither the target nor its replacement appears twice.
                others=rng.sample([f for f in range(8) if f not in (field,rival)],rng.randrange(5))
                order=others+[field];rng.shuffle(order);facts={}
                for f in set(others+[field,rival]):
                    label=rng.choice(LABELS[f]+[EXTRA_LABELS[f]]) if split=='train' else VALID_LABELS[f]
                    value=values[f]
                    if split=='train':
                        facts[f]=rng.choice([f'{label}: {value}',f'{label} | {value}',f'{label}의 기준은 {value}이다.',f'{label} 항목의 내용은 {value}으로 정한다.',f'{label}\n{value}'])
                    else:facts[f]=f'{label} 항목을 확인하면 {value}이라고 되어 있다.'
                prefix=rng.choice(['출장 규정\n','기관 내부 문서\n','가상기관 규정\n','']) if split=='train' else '비용 처리 참고자료\n'
                positive=prefix+'\n'.join(facts[f] for f in order)
                negative=prefix+'\n'.join(facts[rival] if f==field else facts[f] for f in order)
                if rng.random()<.25:
                    note=f'{NOTES[field]} 항목은 이 자료에 기재되어 있지 않다.' if split=='train' else f'참고: {NOTES[field]} 관련 정보는 제시되지 않았다.'
                    negative=negative+'\n'+note if rng.random()<.5 else note+'\n'+negative
                question=rng.choice(QUESTIONS[field]+[f'{rng.choice(LABELS[field])} 내용을 알려줘.']) if split=='train' else VALID_Q[field]
                group=f'{split}-pair-{field}-{i:04d}'
                for missing,document in [(False,positive),(True,negative)]:
                    # Repeated documents within one split are allowed; across splits are not.
                    key=(split,document)
                    assert all((other,document) not in seen for other in ['train','valid'] if other!=split)
                    seen.add(key)
                    answer=None if missing else values[field]
                    encoded=encode(tok,document,question,256,[] if missing else [answer],missing)
                    maximum=max(maximum,len(encoded['ids']))
                    rows.append({'id':group+('-absent' if missing else '-present'),'group':group,'document':document,'question':question,'answer':answer,'is_impossible':missing,'field':FIELDS[field]})
        for previous in value_sets:assert not values_seen&previous
        value_sets.append(values_seen)
        path=root/f'{split}.jsonl';path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        manifest['splits'][split]={'rows':len(rows),'replay_rows':old_count,'new_pairs':8*pairs_per_field,'positive':sum(not r['is_impossible'] for r in rows),'negative':sum(r['is_impossible'] for r in rows),'max_new_input_tokens':maximum,'sha256':sha256(path)}
    # Convenience path for the evaluator; explicitly NOT a fresh test.
    shutil.copy2(parent/'test.jsonl',root/'test.jsonl')
    manifest['regression_test_sha256']=sha256(root/'test.jsonl')
    write_json(root/'provenance.json',manifest);print(json.dumps(manifest['splits'],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
