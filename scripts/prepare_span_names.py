"""Span-only name-boundary diversity; never reads evaluation cases."""
import itertools,json,random
from pathlib import Path
from trid.corpus import sha256,write_json
from tokenizers import Tokenizer
from trid.korean_reader import encode


def main():
    root=Path('data/korean-span-names-v1');root.mkdir(exist_ok=False)
    tok=Tokenizer.from_file('runs/trid-korean-short-v3/span/tokenizer.json')
    all_docs=set();all_names=set();meta={'purpose':'New fictional name combinations and boundaries, with replay of v3 answerable data; span head training only. No evaluation examples read.','seeds':{'train':5100,'valid':5101},'splits':{}}
    for si,(split,count) in enumerate([('train',1200),('valid',120)]):
        rng=random.Random(5100+si)
        prefixes=[['교육','문화','환경','기술','지역','경영','인사','회계','총무','교통','시설','안전','행정','재무','정산','산업','기획','대학'],['복지','경제','정보','청년','홍보','자료','감사','학술','농업','혁신']][si]
        middles=[['지원','운영','관리','협력','정책','연구','사업','평가','분석','개발'],['추진','통합','서비스','조직']][si]
        suffixes=['팀','과','부','실','센터']
        departments=[''.join(parts) for parts in itertools.product(prefixes,middles,suffixes)]
        stems=[['가온','다솜','늘찬','푸른','새길','온누리','해담','바른','소담','은빛','청솔','마루'],['별하','아람','윤슬','꽃내','미르','하늬']][si]
        institutions=[''.join(parts) for parts in itertools.product(stems,prefixes,['재단','연구원','공사','진흥원','연수원'])]
        names=set(departments+institutions);assert not names&all_names;all_names.update(names)
        rows=[r for line in Path(f'data/korean-short-v3/{split}.jsonl').read_text().splitlines() if not (r:=json.loads(line))['is_impossible']]
        replay=len(rows);max_tokens=0
        for i in range(count):
            department=rng.choice(departments);institution=rng.choice(institutions)
            numbers=rng.sample(range(1,100) if split=='train' else range(101,190),2)
            vals=[department,institution,f'{numbers[0]}만원',f'{numbers[1]}만원']
            labels=['담당 부서','기관명','해외 출장 식비','해외 출장 숙박비'] if split=='train' else ['정산 업무 담당 부서','규정 적용 기관명','국외 출장의 식비','국외 출장의 숙박비']
            facts=[]
            for label,value in zip(labels,vals):
                facts.append(rng.choice([f'{label}: {value}',f'{label} | {value}',f'{label}의 값은 {value}이다.',f'{label}\n{value}']) if split=='train' else f'{label} 항목을 읽으면 {value}이라고 쓰여 있다.')
            rng.shuffle(facts);doc=('기관 내부 참고자료\n' if split=='train' else '출장 규정 안내자료\n')+'\n'.join(facts)
            assert doc not in all_docs;all_docs.add(doc)
            qs=[rng.choice(['담당 부서는 어디야?','정산 담당 팀의 이름을 알려줘.','출장비 처리는 어떤 부서에서 맡아?']),rng.choice(['기관 이름이 뭐야?','어느 기관의 규정이야?','회사나 기관의 이름을 알려줘.']),rng.choice(['해외 출장 식비는 얼마야?','국외 출장 식대 지급액을 알려줘.']),rng.choice(['해외 숙박비는 얼마야?','국외 출장 숙소 비용을 알려줘.'])] if split=='train' else ['정산 업무를 담당하는 부서의 이름을 찾아줘.','이 규정을 적용하는 기관 명칭을 찾아줘.','해외 출장 하루에 지급되는 식비를 찾아줘.','해외 숙소에서 1박 할 때의 지급 기준은?']
            for f,(q,value) in enumerate(zip(qs,vals)):
                encoded=encode(tok,doc,q,256,[value],False,character_encoding='hangul_v1');max_tokens=max(max_tokens,len(encoded['ids']))
                rows.append({'id':f'{split}-names-{i:04d}-{f}','group':f'{split}-names-{i:04d}','field':['department','institution','overseas_food','overseas_lodging'][f],'document':doc,'question':q,'answer':value,'is_impossible':False})
        path=root/f'{split}.jsonl';path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        meta['splits'][split]={'rows':len(rows),'replay_rows':replay,'new_documents':count,'department_pool':len(departments),'institution_pool':len(institutions),'max_new_input_tokens':max_tokens,'sha256':sha256(path)}
    write_json(root/'provenance.json',meta);print(json.dumps(meta['splits'],indent=2))


if __name__=='__main__':main()
