"""Deterministic varied-value reading curriculum; no external weights or data.

Fresh tests reserve question phrasing, document template and answer values.
These are still developer-authored synthetic tests, not an independent audit.
"""
import json, random, re, hashlib
from pathlib import Path
from tokenizers import Tokenizer
from trid.text import encode_example, prompt_ids
from evaluate_institution_demo import make_prompt

ROOT=Path(__file__).resolve().parents[1]
# First six phrasings train, next validation, last two fresh test.
FIELDS=[
('food',['출장 하루 식대는?','출장 중 일일 식비 지원액을 알려줘','국내 출장 식사 비용은 하루 얼마까지 지원하지','출장 하루 식사비 기준이 궁금해','하루에 지급하는 국내 출장 식대가 얼마지','출장 때 하루 식비 지원 한도를 확인해 줘','출장 식비의 하루 기준액을 알려 줄래','국내 출장에서 하루치 밥값으로 인정되는 금액은?','출장 하루의 식사 지원금 액수를 확인하고 싶어'],
 ['국내 출장 식비는 하루 {v}이다.','국내 출장 시 하루 식대로 {v}을 지원한다.','일일 국내 출장 식비 지원 한도: {v}.','출장 식사 비용은 하루당 {v}까지 지급한다.','출장자의 하루 식대 기준액을 {v}으로 정한다.','국내 출장에 적용하는 식비의 일일 상한은 {v}이다.','국내 출장 식사비 안내 — 하루 지원금액 [{v}].']),
('deadline',['출장 정산을 언제까지 해야 하지','출장비 정산 마감까지 며칠 있어','돌아오고 며칠 안에 출장비를 신청해야 해','출장 경비 정산 기한을 알려줘','출장 복귀 후 정산 신청 기간은?','출장비 청구가 가능한 복귀 후 기간을 확인해 줘','출장을 다녀온 뒤 정산의 마감 기한이 궁금해','출장 복귀일부터 정산서 제출까지 허용되는 일수는?','출장을 끝낸 후 경비 청구를 마쳐야 하는 기간이 얼마지'],
 ['출장비 정산은 복귀 후 {v} 이내에 신청한다.','출장에서 돌아온 뒤 {v} 안에 경비를 정산해야 한다.','출장비 정산 기한: 복귀 후 {v}.','출장 경비 청구서는 복귀일 기준 {v} 이내 제출한다.','출장 정산 마감일은 복귀 후 {v}이다.','복귀일부터 {v} 내에 출장 경비 정산을 마친다.','출장 경비의 청구 가능 기간은 복귀일부터 세어 [{v}]이다.']),
('travelteam',['출장 경비는 어느 팀에 물어보지','출장비 정산 담당 부서가 어디지','출장비 문제를 담당하는 곳을 알려줘','출장 경비 관련 문의를 보낼 팀은?','출장비 업무를 맡은 조직이 궁금해','출장 정산 문의 담당자는 어느 팀 소속이지','출장 경비 문의의 담당 부서를 확인해 줘','출장 비용에 관한 문의를 전달할 부서 명칭은?','출장 정산에 관해 도움받을 부서 이름을 적어 줘'],
 ['출장비 담당 부서는 {v}이다.','출장 경비 문의는 {v}에서 처리한다.','출장 정산 담당: {v}.','출장비 관련 문의를 {v}에 접수한다.','출장비 업무는 {v}에서 맡는다.','출장 경비에 관한 문의 창구는 {v}이다.','출장비 문의처의 부서명은 [{v}]으로 안내되어 있다.']),
('purchase',['구매 신청 담당 팀을 알려줘','물품을 사려면 어느 부서에 요청하지','구매 요청서를 낼 부서는?','물건 구매를 담당하는 조직은 어디야','구매 업무 담당 부서 이름을 말해줘','물품 구매 요청을 어디에 제출하면 돼','물품 구매 신청서의 접수 부서가 궁금해','사무용 물품 구매 요청의 접수처는 어느 부서인가요','물품을 구입하려는데 요청서를 받을 팀 이름이 필요해'],
 ['물품 구매 요청은 {v}에 제출한다.','물품 구매 담당 부서는 {v}이다.','구매 요청 접수처: {v}.','물품 구매 신청서는 {v}에서 받는다.','구매를 원하는 직원은 {v}에 요청한다.','구매 신청을 접수하는 부서는 {v}이다.','물품 구입 요청서를 받는 조직의 명칭은 [{v}]이다.']),
('approver',['연차 사용을 허락할 사람은 누구지','휴가 신청을 누가 승인하지','연차 결재권자를 알려줘','휴가 사용 승인을 맡은 직책은?','연차 신청서에 결재할 사람은 누구야','연차 신청 승인 담당자의 직책을 확인해 줘','연차 승인 권한이 있는 직책은 무엇인가요','연차 결재를 올려야 할 상대의 직책은?','내 연차 사용 여부를 승인하는 직책을 알려 주세요'],
 ['연차 신청의 승인자는 {v}이다.','연차 사용은 {v}의 승인을 받는다.','연차 결재 담당: {v}.','휴가 신청서는 {v}이 승인한다.','연차를 사용하려면 {v}에게 결재를 요청한다.','연차 신청에 대한 승인 권한은 {v}에게 있다.','연차 사용 결재권자 항목에는 [{v}]이라고 되어 있다.']),
('leavechannel',['휴가를 신청할 시스템을 알려줘','연차 신청서를 어디에 등록하지','연차 접수처가 궁금해','휴가 신청은 어디에서 하지','연차 신청서 제출 시스템은?','연차를 신청하려면 어떤 시스템에 접속하지','연차 사용 신청서를 넣을 전산 창구는 어디지','연차 신청서를 제출하도록 정해진 시스템 명칭은?','휴가를 접수하려고 해. 어떤 시스템 이름을 찾으면 돼'],
 ['연차 신청은 {v}에서 접수한다.','휴가를 신청하려면 {v}을 이용한다.','연차 신청 접수 시스템: {v}.','연차 신청서를 {v}에 등록한다.','휴가 사용 신청은 {v}을 통해 제출한다.','연차 신청서를 접수받는 전산 시스템은 {v}이다.','휴가 신청의 전산 접수처 이름은 [{v}]이다.']),
('device',['노트북 수리는 어느 팀에 요청하지','업무용 노트북에 문제가 생겼는데 담당 부서는?','회사 노트북 고장 담당 팀을 알려줘','컴퓨터가 고장 나면 어느 부서에 문의하지','업무 노트북이 작동하지 않아 어디로 연락해','노트북 고장을 신고할 부서 이름은?','회사에서 쓰는 노트북 수리를 요청할 팀은 어디지','업무 노트북의 장애를 접수받는 팀 명칭이 궁금해','회사 노트북이 켜지지 않을 때 신고를 받을 조직은?'],
 ['업무용 노트북 고장은 {v}에 신고한다.','노트북 고장 담당 부서는 {v}이다.','업무 노트북 장애 접수: {v}.','회사 노트북 수리 요청은 {v}에서 받는다.','업무용 컴퓨터가 고장 나면 {v}에 연락한다.','업무 노트북의 고장 신고를 받는 곳은 {v}이다.','업무용 노트북의 장애 접수 부서명은 [{v}]이다.']),
('phone',['장비 수리 문의 전화번호를 알려줘','노트북 고장은 내선 몇 번으로 전화하지','장비 장애 접수 번호는?','고장 신고를 할 내부 전화는 몇 번이야','장비 문의 전화 번호가 궁금해','노트북 장애 접수 내선 번호를 확인해 줘','장비가 고장 났을 때 사용할 내선을 알려줘','장비 장애를 신고하려고 눌러야 하는 내선 번호는?','노트북 수리 요청 전화를 걸 번호를 적어 줘'],
 ['장비 고장 내부 문의 번호는 {v}이다.','노트북 고장 문의는 내선 {v}로 연락한다.','장비 수리 문의 전화: {v}.','장비 장애 신고를 위한 내선은 {v}이다.','고장 난 노트북은 내선 {v}에 문의한다.','장비 장애 접수 담당자의 내선 번호는 {v}이다.','노트북 수리 안내에 적힌 연락 내선은 [{v}]이다.']),
('retention',['회의 기록의 보존 연한을 알려줘','회의록을 보관하는 기간은?','회의 문서는 얼마나 오래 보관하지','회의록 보존 기간이 궁금해','회의 기록을 몇 년이나 보관해야 하지','회의록을 남겨 둬야 하는 기간은?','회의 기록의 보관 의무 기간을 확인하고 싶어','회의 기록을 폐기하지 않고 보존해야 하는 연수는?','회의록의 보존 연한 항목에는 어떤 기간이 나와 있어'],
 ['회의록 보관 기간은 {v}이다.','회의 기록은 {v} 동안 보존한다.','회의록 보존 연한: {v}.','회의록을 폐기하지 않고 {v}간 보관한다.','회의 기록의 보존 기간을 {v}으로 정한다.','회의록에는 {v}의 보관 의무가 있다.','회의 기록의 보존 기간 항목은 [{v}]으로 정해져 있다.']),
('recordsystem',['회의 기록을 등록할 시스템을 알려줘','회의록은 어느 시스템에 넣지','회의 문서 등록 장소는?','회의록을 올릴 곳이 어디야','회의록 등록용 시스템 이름은?','회의 기록을 작성하면 어느 시스템에 등록해야 해','회의록을 접수할 전산 시스템이 궁금해','회의가 끝난 뒤 회의록을 올리도록 지정된 시스템은?','회의 기록의 등록처로 안내된 시스템 명칭을 적어 줘'],
 ['회의록은 {v}에 등록한다.','회의 기록 등록 시스템은 {v}이다.','회의록 등록처: {v}.','회의를 마치면 {v}에 회의록을 올린다.','회의 기록은 {v}을 통해 등록한다.','회의록의 전산 등록은 {v}에서 진행한다.','회의 기록의 등록 시스템 명칭은 [{v}]이다.']),
('visitdeadline',['방문 예약 신청 마감이 언제야','방문 신청은 며칠 전에 접수하지','방문하려면 며칠 앞서 예약하지','방문 신청 기한을 알려 줘','방문 예약을 사전에 해야 하는 기간은?','방문일 며칠 전에 신청을 마쳐야 하지','방문 신청은 예정일보다 며칠 앞서 해야 해','방문 예정일부터 거꾸로 며칠 전에 예약해야 하나요','방문 예약을 위해 확보해야 할 사전 신청 일수는?'],
 ['방문 예약은 방문일 {v} 전까지 신청한다.','방문 신청은 방문 예정일보다 {v} 앞서 접수한다.','방문 사전 예약 기한: {v} 전.','방문을 원하면 {v} 전에 예약해야 한다.','방문 예약 신청은 예정일 {v} 전에 마친다.','방문하려는 날짜보다 {v} 먼저 예약을 접수한다.','방문 예약의 사전 신청 기간 항목에는 [{v}]이라고 적혀 있다.']),
('visitteam',['방문 예약은 어느 팀에 물어보지','방문 신청 담당 부서를 알려줘','방문 예약을 처리하는 곳이 어디야','방문 예약 관련 문의 부서는?','방문 신청을 접수하는 부서 이름은?','기관 방문 예약은 어디에 요청해야 해','방문 신청 문의 담당 조직을 확인해 줘','기관을 방문하고 싶을 때 예약 신청을 받을 부서명은?','방문 예약 접수 업무를 맡은 팀 명칭이 필요해'],
 ['방문 예약 담당 부서는 {v}이다.','방문 신청은 {v}에 문의한다.','방문 예약 접수처: {v}.','방문을 예약하려면 {v}으로 연락한다.','방문 신청 업무는 {v}에서 처리한다.','방문 예약을 접수받는 담당 조직은 {v}이다.','방문 예약 업무 담당 부서 항목은 [{v}]이다.'])]


def values(key,split,rng):
    numbers=list(range(1,19)) if split=='train' else [19,20] if split=='valid' else list(range(21,46))
    if key=='food':return f'{rng.choice(numbers)}만원'
    if key in ('deadline','visitdeadline'):return f'{rng.choice(numbers)}일'
    if key=='retention':return f'{rng.choice(numbers)}년'
    if key=='phone':return str(rng.randrange(1000,6000) if split=='train' else rng.randrange(6000,7000) if split=='valid' else rng.randrange(7000,10000))
    if key=='approver':return rng.choice(['팀장','부서장','센터장','실장','본부장','담당자'])
    if key in ('leavechannel','recordsystem'):
        pool=['인사시스템','업무포털','그룹웨어','근태관리시스템','기록관리시스템','문서관리시스템','전자문서시스템'] if split=='train' else ['행정포털','지원시스템'] if split=='valid' else ['통합관리시스템','사내업무포털','기록지원시스템']
        return rng.choice(pool)
    pool=['운영지원팀','경영지원팀','총무팀','재무팀','회계팀','행정지원팀','구매지원팀','자산관리팀','운영팀','정보지원팀','전산팀','IT지원팀','장비관리팀','안내지원팀','기록관리팀'] if split=='train' else ['시설관리팀','업무지원팀'] if split=='valid' else ['시설지원팀','고객지원팀','행정관리팀','기술지원팀']
    return rng.choice(pool)


def build():
    out=ROOT/'data/grounded-v2';out.mkdir(exist_ok=False)
    tok=Tokenizer.from_file(str(ROOT/'runs/trid-institution-v1/tokenizer.json'))
    normalize=lambda s:re.sub(r'\W','',s)
    excluded={normalize(json.loads(x)['question']) for x in (ROOT/'data/institution-demo-v1/eval.jsonl').read_text().splitlines()}
    data={};seen=set();skipped=0
    for split,count in [('train',14000),('valid',600),('test',240)]:
        rng=random.Random({'train':541,'valid':542,'test':543}[split]);rows=[]
        for n in range(count):
            key,qs,templates=FIELDS[n%len(FIELDS)]
            q=rng.choice(qs[:6] if split=='train' else qs[6:7] if split=='valid' else qs[7:])
            if normalize(q) in excluded:continue
            value=values(key,split,rng);other=values(key,split,rng)
            while other==value:other=values(key,split,rng)
            template=rng.choice(templates[:5]) if split=='train' else templates[5] if split=='valid' else templates[6]
            branch=rng.choices(['direct','missing','conflict','revision'],[75,8,7,10])[0]
            if split=='test':branch='direct'  # paired value changes; absence separately below
            statements=[template.format(v=value)]
            answer=value
            if branch=='missing':statements=[];answer='자료없음'
            elif branch=='conflict':
                statements=[f'자료 A: {template.format(v=value)} 자료 B: {template.format(v=other)} 어느 자료가 우선인지 알 수 없다.'];answer='규정충돌'
            elif branch=='revision':
                statements=[f'폐지된 옛 규정: {template.format(v=other)} 이를 대체한 현행 규정: {template.format(v=value)}']
            distractors=[]
            for f,_,ts in rng.sample([f for f in FIELDS if f[0]!=key],rng.choice([1,2])):
                t=rng.choice(ts[:5]) if split=='train' else ts[5] if split=='valid' else ts[6]
                distractors.append(t.format(v=values(f,split,rng)))
            statements+=distractors;rng.shuffle(statements)
            text=rng.choice(['가상기관 규정: ','업무 안내: ','내부 업무 매뉴얼: ',''])+' '.join(statements)
            case={'id':f'{split}-{n}','group':branch,'question':q,'evidence':[{'id':rng.choice(['D01','T01','A12','M03']),'text':text}],'expected':answer}
            prompt=make_prompt(case)
            row={'prompt':prompt,'answer':answer,'task':f'grounded-{branch}','source':'authored varied-document curriculum','weight':1}
            if prompt in seen:continue
            try:
                encode_example(tok,row,256)
                if len(prompt_ids(tok,prompt))+48>256:raise ValueError('output reserve')
            except ValueError:skipped+=1;continue
            seen.add(prompt)
            if split!='test':rows.append(row)
            else:
                case['counterfactual_pair']=case['id'];rows.append(case)
                othercase={**case,'id':case['id']+'-changed','expected':other,'evidence':[{'id':case['evidence'][0]['id'],'text':text.replace(template.format(v=value),template.format(v=other))}]}
                rows.append(othercase)
        data[split]=rows
    # Retain diverse familiar basic dialogue at low sampling weight. No new claims of general conversation skill.
    old=[json.loads(x) for x in (ROOT/'data/dialogue-v4/train.jsonl').read_text().splitlines()]
    rng=random.Random(544)
    for r in rng.sample(old,min(2500,len(old))):
        if r['prompt'] in seen:continue
        r={**r,'weight':.3,'source':'local dialogue-v4 replay'}
        data['train'].append(r);seen.add(r['prompt'])
    for split,rows in data.items():
        (out/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    info={'counts':{k:len(v) for k,v in data.items()},'overlength_skipped':skipped,'sources':'Authored synthetic documents and local dialogue-v4 replay. No new downloads or external model weights.', 'split':'Question phrasing and document sentence templates disjoint; held-out number ranges, phone ranges, teams and system names for valid/test. Approver vocabulary overlaps. All tests synthetic and developer-authored; not independent auditing. Existing 24 demo questions excluded exactly.', 'test_note':'240 base cases and 240 paired value changes; final number may be smaller if token limits exceeded. Generated before training. Test prompts not used for checkpoint selection.', 'sha256':{s:hashlib.sha256((out/f'{s}.jsonl').read_bytes()).hexdigest() for s in data}}
    (out/'provenance.json').write_text(json.dumps(info,ensure_ascii=False,indent=2))
    print(json.dumps(info,ensure_ascii=False,indent=2))

if __name__=='__main__':build()
