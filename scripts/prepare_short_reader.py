"""Authored short-document curriculum. No real institutional policy claims."""
import json
import random
from pathlib import Path
from itertools import combinations
from trid.corpus import sha256, write_json
from trid.korean_reader import encode
from tokenizers import Tokenizer

FIELDS = ['domestic_food','overseas_food','lodging','deadline','department','institution']
FACTS = {
 'train': [
 ['국내 출장 식비는 하루 {}이다.','해외 출장 식비는 하루 {}이다.','국내 출장 숙박비 한도는 1박에 {}이다.','출장비 정산은 복귀 후 {} 이내에 신청한다.','출장비 정산 담당 부서는 {}이다.','이 규정을 사용하는 기관의 이름은 {}이다.'],
 ['국내 출장 시 일일 식대로 {}을 지급한다.','해외 출장 시 일일 식대로 {}을 지급한다.','국내 출장의 1박 숙박비 상한은 {}이다.','출장에서 돌아오면 {} 안에 정산해야 한다.','정산 업무는 {}에서 맡는다.','규정 적용 기관명은 {}이다.'],
 ['국내 출장의 하루 식사비는 {}으로 정한다.','해외 출장의 하루 식사비는 {}으로 정한다.','국내 숙박에 대해 하룻밤 최대 {}을 지원한다.','출장 종료 뒤 정산 신청 기한은 {}이다.','정산 문의는 {}으로 한다.','이 지침의 기관은 {}이다.']],
 'valid': [['국내 출장자의 하루 식비 지급액: {}.','해외 출장자의 하루 식비 지급액: {}.','국내 출장 시 숙소 비용은 1박 {}까지 인정한다.','복귀일로부터 정산 신청에 허용되는 기간은 {}이다.','출장 정산 업무의 담당 조직은 {}이다.','문서에 명시된 기관 명칭은 {}이다.']],
 'test': [['국내 출장 중 식사에 쓸 수 있는 일일 금액은 {}이다.','해외 출장 중 식사에 쓸 수 있는 일일 금액은 {}이다.','국내 출장 숙소의 하룻밤 비용 한도를 {}으로 둔다.','출장을 마치고 돌아온 뒤 {} 안으로 비용을 정산한다.','출장비 정산을 처리하는 부서는 {}이다.','이 문서의 규정을 운영하는 기관은 {}이다.']]
}
QUESTIONS = {
 'train': [
 ['국내 출장 식비는 하루 얼마야?','해외 출장 식비는 하루 얼마야?','국내 출장 숙박비는 1박에 얼마야?','복귀 후 정산 신청 기한은 며칠이야?','출장비 정산 담당 부서는 어디야?','규정을 사용하는 기관 이름은 뭐야?'],
 ['국내 출장 하루 식대를 알려줘.','해외 출장 하루 식대를 알려줘.','국내 출장의 숙박비 상한은 얼마인가?','출장이 끝나면 며칠 안에 정산해야 하나?','정산 문의를 어느 부서로 해야 해?','어느 기관의 지침인가?'],
 ['국내 출장 일일 식사비 금액은?','해외 출장 일일 식사비 금액은?','국내 숙박에 지원되는 최대 금액은?','출장 종료 뒤 비용 정산에 허용된 기간은?','정산 업무를 맡는 팀 이름은?','문서의 기관 명칭을 답해줘.']],
 'valid': [['국내 출장자의 일일 식비 지급액은 얼마인가요?','해외 출장자의 일일 식비 지급액은 얼마인가요?','국내 출장 숙소 비용을 하루에 얼마까지 인정하나요?','돌아온 날부터 정산을 신청할 수 있는 기간은?','출장 정산 담당 조직의 명칭은?','문서에 적힌 기관명은 무엇인가요?']],
 'test': [['국내 출장 중 하루 식사에 쓸 수 있는 돈이 얼마인지 알려줘.','해외 출장 중 하루 식사에 쓸 수 있는 돈이 얼마인지 알려줘.','국내 출장에서 하룻밤 묵는 비용의 한도를 알려줘.','출장을 마치고 돌아와서 언제까지 비용을 정산해야 하지?','출장비 정산을 처리하는 부서가 어디인지 알려줘.','이 규정을 운영하는 기관의 이름을 알려줘.']]
}


def main():
    root=Path('data/korean-short-v1')
    if root.exists():raise ValueError('Use a new data version')
    tok=Tokenizer.from_file('runs/trid-korean-base-v1/tokenizer.json')
    root.mkdir();man={'origin':'Codex-authored fictional policy facts; not real regulations',
      'split_policy':'Different document groups, complete sentence/question templates, numeric ranges and name pools per split; common six field concepts intentionally shared',
      'inference':'Templates and field labels are only data preparation; runtime does not use them to answer',
      'limits':'Narrow synthetic curriculum, not natural-language benchmark or public-institution readiness',
      'max_training_input_tokens':192,'tokenizer_sha256':sha256('runs/trid-korean-base-v1/tokenizer.json'),
      'templates':{'facts':FACTS,'questions':QUESTIONS},'splits':{}}
    all_documents=set();all_values={}
    for offset,(split,n) in enumerate([('train',400),('valid',80),('test',100)]):
        rng=random.Random(900+offset);rows=[];values_seen=set();max_tokens=0
        names=[['가상해솔','가상누리','가상새봄','가상별빛','가상한결','가상온빛'],['가상초록','가상여울','가상청명'],['가상노을','가상다온','가상보람']][offset]
        for i in range(n):
            numbers=rng.sample(range(1+offset*100,81+offset*100),4)
            stem=rng.choice(names)
            values=[f'{numbers[0]}만원',f'{numbers[1]}만원',f'{numbers[2]}만원',f'{numbers[3]}일',stem+'지원팀',stem+'연구원']
            included=list(combinations(range(6),3))[i%20]
            facts=[rng.choice(FACTS[split])[f].format(values[f]) for f in included]
            rng.shuffle(facts);document=' '.join(facts)
            if document in all_documents:raise ValueError('Repeated document')
            all_documents.add(document);values_seen.update(values)
            for f in range(6):
                question=rng.choice(QUESTIONS[split])[f];present=f in included
                answer=values[f] if present else None
                encoded=encode(tok,document,question,192,[answer] if present else [],not present)
                max_tokens=max(max_tokens,len(encoded['ids']))
                rows.append({'id':f'{split}-{i:04d}-{f}','group':f'{split}-{i:04d}', 'document':document,'question':question,
                             'answer':answer,'is_impossible':not present,'field':FIELDS[f]})
        for other in all_values.values():assert not values_seen&other
        all_values[split]=values_seen
        p=root/f'{split}.jsonl';p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        man['splits'][split]={'rows':len(rows),'documents':n,'positive':len(rows)//2,'negative':len(rows)//2,'max_input_tokens':max_tokens,'sha256':sha256(p)}
    write_json(root/'provenance.json',man)
    print(json.dumps(man['splits'],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
