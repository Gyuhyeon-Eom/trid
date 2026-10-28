"""Synthetic grounded QA curriculum; separate from the frozen demo evaluation.

Values, question wording, and option positions vary. Same-topic synthetic tests
remain limited evidence of generalization, even with exact question exclusion.
"""
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

from tokenizers import Tokenizer
from trid.text import encode_example
from evaluate_institution_demo import make_prompt

ROOT=Path(__file__).resolve().parents[1]


def norm(s):
    return re.sub(r'\W+','',unicodedata.normalize('NFC',s))


def main():
    out=ROOT/'data/institution-training-v1'
    out.mkdir(exist_ok=False)
    tok=Tokenizer.from_file(str(ROOT/'runs/trid-v0.2/tokenizer.json'))
    excluded={norm(json.loads(l)['question']) for l in (ROOT/'data/institution-demo-v1/eval.jsonl').read_text().splitlines()}
    # Each final wording below is reserved for validation; no evaluation-suite wording is inserted.
    fields=[
      ('국내 출장 식비','하루', ['1만원','2만원','3만원','4만원','5만원','6만원','7만원','8만원'],
       ['출장 하루 식대 금액이 궁금해','국내 출장의 일일 식비를 알려줘','출장 가면 하루에 받는 식사비는 얼마지','국내 출장 하루 식대는?','출장 중 하루 식사 비용 기준을 확인해 줘']),
      ('출장비 정산 기한','복귀 후', ['3일','5일','7일','10일','14일','20일','30일'],
       ['출장 정산 신청 기한은?','출장을 마친 뒤 언제까지 정산하지','복귀하고 정산할 수 있는 기간을 알려 줘','출장비를 청구하는 기한이 궁금해','출장비 정산은 복귀 후 언제까지야']),
      ('출장비 담당 부서','', ['운영지원팀','경영지원팀','총무팀','재무팀','회계팀','행정지원팀'],
       ['출장비를 담당하는 팀은 어디야','출장 경비는 어느 팀에 물어봐야 하지','출장 정산 담당 부서를 알려줘','출장비 관련 문의를 보낼 곳은?','출장 경비를 문의할 부서가 궁금해']),
      ('구매 요청 담당 부서','', ['구매지원팀','총무팀','자산관리팀','재무팀','운영팀'],
       ['구매 신청 담당 팀을 알려줘','물품을 사려면 어느 부서에 요청하지','구매 요청서를 낼 부서는?','물건 구매를 담당하는 조직은 어디야','구매 신청은 어느 팀에서 처리해']),
      ('연차 승인자','', ['팀장','부서장','센터장','실장','담당자'],
       ['휴가 신청서를 누가 승인하지','연차 사용을 승인하는 사람은?','연차 결재권자를 알려줘','연차 신청의 결재 담당자가 누구지','누구에게 연차 허가를 받으면 돼']),
      ('연차 신청 접수처','', ['인사시스템','업무포털','그룹웨어','근태관리시스템'],
       ['휴가를 신청할 시스템을 알려줘','연차 신청서를 어디에 등록하지','연차 접수처가 궁금해','휴가 신청은 어디에서 하지','연차를 신청하는 창구를 알려 줘']),
      ('업무용 노트북 고장 담당 부서','', ['정보지원팀','전산팀','IT지원팀','장비관리팀'],
       ['노트북 수리는 어느 팀에 요청하지','업무용 노트북에 문제가 생겼는데 담당 부서는?','회사 노트북 고장 담당 팀을 알려줘','컴퓨터가 고장 나면 어느 부서에 문의하지','업무용 노트북 수리 담당 조직은 어디야']),
      ('장비 고장 문의 번호','', ['1357','2468','3579','4826','5831','6240','7315','8642'],
       ['장비 수리 문의 전화번호를 알려줘','노트북 고장은 내선 몇 번으로 전화하지','장비 장애 접수 번호는?','고장 신고를 할 내부 전화는 몇 번이야','장비 고장 관련 연락 번호가 궁금해']),
      ('회의록 보관 기간','', ['1년','2년','3년','4년','5년','7년','10년'],
       ['회의 기록의 보존 연한을 알려줘','회의록을 보관하는 기간은?','회의 문서는 얼마나 오래 보관하지','회의록 보존 기간이 궁금해','회의록 보관 의무 기간을 확인해 줘']),
      ('회의록 등록 시스템','', ['기록관리시스템','문서관리시스템','업무포털','전자문서시스템'],
       ['회의 기록을 등록할 시스템을 알려줘','회의록은 어느 시스템에 넣지','회의 문서 등록 장소는?','회의록을 올릴 곳이 어디야','회의록 등록용 시스템이 궁금해']),
      ('방문 예약 기한','방문일', ['1일 전','2일 전','3일 전','5일 전','7일 전'],
       ['방문 예약 신청 마감이 언제야','방문 신청은 언제까지 접수하지','방문하려면 며칠 전에 예약하지','방문 신청 기한을 알려 줘','방문 예약은 방문일로부터 얼마나 먼저 해야 해']),
      ('방문 예약 담당 부서','', ['안내지원팀','운영지원팀','총무팀','안내데스크'],
       ['방문 예약은 어느 팀에 물어보지','방문 신청 담당 부서를 알려줘','방문 예약을 처리하는 곳이 어디야','방문 예약 관련 문의 부서는?','방문 신청은 누가 담당해']),
    ]
    rows={'train':[],'valid':[]};seen=set();skipped=0
    for split,count,seed in [('train',7200,811),('valid',480,812)]:
        rng=random.Random(seed)
        for n in range(count):
            i=rng.randrange(len(fields));field,qualifier,values,questions=fields[i]
            q=rng.choice(questions[:-1]) if split=='train' else questions[-1]
            value,other=rng.sample(values,2)
            branch=rng.choices(['direct','missing','conflict','revision','condition'],[6,2,1,1,2])[0]
            if branch=='condition':
                limit=rng.choice([20,30,50,70,100]);price=rng.choice([limit-10,limit,limit+10])
                lower,upper=rng.sample(['팀장','부서장','센터장','실장'],2)
                text=f'가상기관 규정: {limit}만원 미만의 물품 구매 승인자는 {lower}이다. {limit}만원 이상 구매 승인자는 {upper}이다.'
                q=rng.choice([f'{price}만원 구매의 결재권자를 알려줘',f'물품 가격이 {price}만원이면 승인 담당자는 누구지',f'{price}만원짜리를 구매할 때 승인할 사람은?']) if split=='train' else f'구매 금액이 {price}만원일 때 누가 승인하지?'
                value=lower if price<limit else upper
                options=[lower,upper,'자료없음']
            elif branch=='missing':
                distractor=rng.choice([j for j in range(len(fields)) if j!=i])
                df,dq,dv,_=fields[distractor]
                text=f'가상기관 규정: {df}은 {dq} {rng.choice(dv)}이다.'
                if rng.random()<.5:text+=f' {field}은 이 문서에 정하지 않았다.'
                options=[value,other,'자료없음'];value='자료없음'
            elif branch=='conflict':
                text=f'가상기관 안내 A: {field}은 {qualifier} {value}이다. 안내 B: {field}은 {qualifier} {other}이다. 두 안내의 시행일과 우선순위는 알려지지 않았다.'
                options=[value,other,'규정충돌'];value='규정충돌'
            elif branch=='revision':
                text=f'가상기관 규정: 폐지된 구규정의 {field}은 {qualifier} {other}이다. 구규정을 대체한 현행 규정의 {field}은 {qualifier} {value}이다.'
                options=[value,other,'규정충돌']
            else:
                statements=[f'{field}은 {qualifier} {value}이다.']
                for j in rng.sample([j for j in range(len(fields)) if j!=i],rng.choice([0,1,2])):
                    f,qual,vals,_=fields[j];statements.append(f'{f}은 {qual} {rng.choice(vals)}이다.')
                rng.shuffle(statements)
                text='가상기관 규정: '+' '.join(statements)
                options=[value,other,'자료없음']
            if norm(q) in excluded:continue
            rng.shuffle(options)
            case={'id':f'{split}-{n}','question':q,'evidence':[{'id':'D01','text':text}]}
            for mode in ['generate','choice']:
                prompt=make_prompt(case,None if mode=='generate' else options)
                answer=value if mode=='generate' else 'ABC'[options.index(value)]
                if prompt in seen:continue
                row={'prompt':prompt,'answer':answer,'task':f'{mode}-{branch}','source':'authored synthetic institutional curriculum','weight':1}
                try:encode_example(tok,row,256)
                except ValueError:skipped+=1;continue
                seen.add(prompt);rows[split].append(row)
        rng.shuffle(rows[split])
    for split,rs in rows.items():
        (out/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs))
    manifest={'purpose':'Synthetic grounded QA and separately taught A/B/C output format; no actual institution data',
      'eval_question_exclusion':'All normalized standalone questions in institution-demo-v1 excluded.',
      'split':'Separate question phrasings and random seeds. Same subjects/answer pools/templates; near-duplicates remain possible.',
      'limits':'Cannot establish real institutional competence. Developer has seen baseline evaluation failures. Numeric conditions are toy tasks, not production policy enforcement.',
      'counts':{s:len(r) for s,r in rows.items()},'overlength_skipped':skipped,
      'sha256':{s:hashlib.sha256((out/f'{s}.jsonl').read_bytes()).hexdigest() for s in rows}}
    (out/'provenance.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
