"""New contrastive curriculum; old evaluated test rows are not recycled."""
import hashlib,itertools,json,random
from pathlib import Path
from tokenizers import Tokenizer
from trid.korean_reader import encode
from trid.corpus import sha256,write_json

FIELDS=['domestic_food','overseas_food','lodging','overseas_lodging','deadline','department','institution','daily_allowance']
LABELS=[
 ['국내 출장 식비','국내여비 식대','국내 출장 하루 식사비'],
 ['해외 출장 식비','국외 출장 식대','외국 출장 하루 식사비'],
 ['국내 출장 숙박비','국내여비 숙소비','국내 출장 1박 숙박비'],
 ['해외 출장 숙박비','국외 출장 숙소비','외국 출장 1박 숙박비'],
 ['출장 정산 기한','복귀 후 정산 신청 기간','출장비 정산 신청 기한'],
 ['담당 부서','출장 정산 담당부서','정산 업무 담당 팀'],
 ['기관명','규정 적용 기관 이름','공공기관 이름'],
 ['국내 출장 일비','국내여비 일일 수당','국내 출장 하루 일당']]
VALID_LABELS=['국내 출장의 일일 식비','해외 출장의 일일 식비','국내 출장 숙박료','해외 출장 숙박료','출장 복귀 뒤 정산 기한','비용 정산 담당 부서','이 규정의 기관명','국내 출장의 일비']
TEST_LABELS=['국내 출장 식사 경비','국외 출장 식사 경비','국내 출장 숙소 경비','국외 출장 숙소 경비','출장 정산 접수 기간','정산 담당 조직','규정 운영기관 명칭','국내 출장 일일 수당액']
QUESTIONS=[
 ['국내 출장 식비는 얼마야?','국내 식대 지급액을 알려줘.','국내 출장 하루 식사비가 얼마인가요?','국내 출장 중 먹는 데 쓸 수 있는 돈은?'],
 ['해외 출장 식비는 얼마야?','국외 출장 식대 지급액을 알려줘.','해외에서 하루 식사비가 얼마인가요?','외국 출장 중 먹는 데 쓸 수 있는 돈은?'],
 ['국내 숙박비는 얼마야?','국내 출장 숙소 비용을 알려줘.','국내에서 하룻밤 묵는 비용은 얼마인가요?','국내 출장의 숙박비 상한액은?'],
 ['해외 숙박비는 얼마야?','국외 출장 숙소 비용을 알려줘.','해외에서 하룻밤 묵는 비용은 얼마인가요?','외국 출장의 숙박비 상한액은?'],
 ['출장 정산 기한은 며칠이야?','복귀 후 정산을 언제까지 해야 해?','출장비 정산 신청 기간을 알려줘.','출장을 마친 뒤 비용을 청구할 수 있는 기간은?'],
 ['담당 부서가 어디야?','정산 담당 팀의 이름을 알려줘.','출장비 처리는 어떤 부서에서 맡아?','정산 문의는 어디로 하면 돼?'],
 ['기관 이름이 뭐야?','어느 기관의 규정이야?','공공기관 이름이 뭔데?','회사나 기관의 이름을 알려줘.'],
 ['국내 출장 일비는 얼마야?','국내 출장 하루 수당을 알려줘.','국내 출장의 일당은 얼마인가요?','국내여비 중 일비 금액은?']]
VALID_Q=['국내 출장 하루에 지급되는 식비를 찾아줘.','해외 출장 하루에 지급되는 식비를 찾아줘.','국내 숙소에서 1박 할 때의 지급 기준은?','해외 숙소에서 1박 할 때의 지급 기준은?','출장에서 돌아온 후 정산 가능 기간을 찾아줘.','비용 정산을 담당하는 부서명을 찾아줘.','이 규정을 적용하는 기관 명칭을 찾아줘.','국내 출장 일비 지급 기준을 찾아줘.']
TEST_Q=['국내 출장 때 식사 경비로 얼마를 받을 수 있지?','국외 출장 때 식사 경비로 얼마를 받을 수 있지?','국내 출장 때 숙소 경비로 얼마를 받을 수 있지?','국외 출장 때 숙소 경비로 얼마를 받을 수 있지?','출장 정산 접수를 마쳐야 하는 기간이 어떻게 돼?','정산 담당 조직의 정확한 이름이 어떻게 돼?','규정을 운영하는 기관의 정식 명칭이 어떻게 돼?','국내 출장 중 일비로 받는 액수가 어떻게 돼?']


def topic(s):
    x=ord(s[-1]);return '은' if 0xAC00<=x<=0xD7A3 and (x-0xAC00)%28 else '는'


def main():
    root=Path('data/korean-short-v2')
    if root.exists():raise ValueError('Use a new curriculum version')
    tok=Tokenizer.from_file('runs/trid-korean-short-v1/span/tokenizer.json');root.mkdir()
    combos=list(itertools.combinations(range(8),4));seen=set();value_sets=[]
    manifest={'origin':'Authored fictional contrastive policy data, not actual institutional rules',
       'focus':'Separate domestic/overseas food and lodging; department vs institution name; unnamed institutions with departments present',
       'split_policy':'Full documents, full fact/question styles and values/name pools separated. Shared eight concepts. Numeric digit lengths represented in every split.',
       'limits':'Narrow synthetic curriculum. Existing user regression wording may be included in training; legacy evaluations are development diagnostics, not unseen tests.',
       'no_recycling':'Does not read prior test rows; v2 test frozen before training. Old test performance is regression only.',
       'templates':{'train_labels':LABELS,'valid_labels':VALID_LABELS,'test_labels':TEST_LABELS,'train_questions':QUESTIONS,'valid_questions':VALID_Q,'test_questions':TEST_Q},
       'splits':{},'tokenizer_sha256':sha256('runs/trid-korean-short-v1/span/tokenizer.json')}
    for si,(split,n) in enumerate([('train',840),('valid',70),('test',140)]):
        rng=random.Random(2700+si);rows=[];vals=set();maximum=0
        bucket=lambda x:int(hashlib.sha256(str(x).encode()).hexdigest()[:8],16)%10
        numbers=[i for i in range(1,200) if (bucket(i)<7 if split=='train' else bucket(i)==7 if split=='valid' else bucket(i)>=8)]
        stems=[['온솔','해담','바른','가온','소담','늘찬','마루','청솔','은빛','새길','다솜','푸른'],['별하','아람','윤슬','꽃내'],['나봄','미리내','라움','솔비','고운','모아']][si]
        departments=[['운영지원팀','재무지원팀','회계과','총무팀','인사부','행정지원실','경영지원팀','정산관리과'],['기획협력팀','여비관리과','문화지원실','업무지원부'],['사업관리팀','회계협력과','행정운영실','재정지원부']][si]
        for i in range(n):
            nums=rng.sample(numbers,6)
            money=lambda k,foreign=False:f'{nums[k]}달러' if foreign and rng.random()<.4 else (f'{nums[k]}만원' if rng.random()<.6 else f'{nums[k]*1000:,}원')
            values=[money(0),money(1,True),money(2),money(3,True),f'{nums[4]}일',rng.choice(departments),rng.choice(stems)+rng.choice(['문화재단','환경연구원','도시공사','기술센터','교육진흥원']),money(5)]
            present=combos[i%len(combos)];facts=[]
            for f in present:
                label=rng.choice(LABELS[f]) if split=='train' else (VALID_LABELS if split=='valid' else TEST_LABELS)[f]
                v=values[f]
                if split=='train':
                    style=rng.randrange(5)
                    fact=[f'{label}{topic(label)} {v}이다.',f'{label}: {v}',f'{label} | {v}',f'이 문서에 적힌 {label}{topic(label)} {v}이다.',f'{label} 항목에는 {v}이라고 명시되어 있다.'][style]
                elif split=='valid':fact=f'{label}의 명시된 내용은 {v}이다.'
                else:fact=f'{label}에 해당하는 값은 {v}으로 기록되어 있다.'
                facts.append(fact)
            rng.shuffle(facts)
            prefix=rng.choice(['','가상기관 규정\n','기관 내부 참고 문서\n','출장 업무 안내\n']) if split=='train' else ('기관 업무 기준\n' if split=='valid' else '기관 출장 처리 문서\n')
            document=prefix+('\n' if rng.random()<.5 else ' ').join(facts)
            if document in seen:raise ValueError('Duplicate document')
            seen.add(document);vals.update(values)
            for f in range(8):
                q=rng.choice(QUESTIONS[f]+[f'{rng.choice(LABELS[f])} 내용을 알려줘.']) if split=='train' else (VALID_Q if split=='valid' else TEST_Q)[f]
                answer=values[f] if f in present else None
                row=encode(tok,document,q,256,[answer] if answer else [],answer is None);maximum=max(maximum,len(row['ids']))
                rows.append({'id':f'{split}-{i:04d}-{f}','group':f'{split}-{i:04d}','document':document,'question':q,'answer':answer,'is_impossible':answer is None,'field':FIELDS[f]})
        for earlier in value_sets:assert not vals&earlier
        value_sets.append(vals)
        p=root/f'{split}.jsonl';p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        manifest['splits'][split]={'documents':n,'rows':len(rows),'positive':len(rows)//2,'negative':len(rows)//2,'max_input_tokens':maximum,'sha256':sha256(p)}
    write_json(root/'provenance.json',manifest);print(json.dumps(manifest['splits'],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
