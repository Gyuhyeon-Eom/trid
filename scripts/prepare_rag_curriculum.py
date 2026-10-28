"""Local synthetic Korean evidence-reading curriculum; no downloaded model or API."""
import argparse,hashlib,json,random,shutil
from collections import Counter
from pathlib import Path
from tokenizers import Tokenizer
from trid.text import encode_example

ROOT=Path('data/korean-rag-curriculum-v1')
BASE=Path('data/base-dialogue-v1')
from trid.prompts import RAG_PREFIX as PREFIX
KINDS=['amount','name','noise','table','missing','conflict','revision','combine']

def example(rng,kind,split,i):
    # Train/valid/test values differ; combinations, wording, IDs and evidence order vary.
    amounts={'train':[2,5,8,11,14,17,20,23,26,29], 'valid':[3,6,9,12,15], 'test':[4,7,10,13,16]}[split]
    names={'train':['김민수','이서연','박지훈','정하린','최도윤','강서준','한지우','송예린'],
           'valid':['조은서','임건우','문소율'], 'test':['오지수','신나래','백시온']}[split]
    a,b=rng.sample(amounts,2);name=rng.choice(names)
    ids=rng.sample(['D1','D2','D3','D4','D5','D6','D7','D8'],3);s,t,u=ids
    field=rng.choice(['국내 식비','해외 식비','국내 숙박비','해외 숙박비'])
    other={'국내 식비':'해외 식비','해외 식비':'해외 숙박비','국내 숙박비':'국내 식비','해외 숙박비':'해외 식비'}[field]
    ask={'train':[f'{field}는 얼마인가요?',f'{field} 금액을 알려줘.'],
         'valid':[f'자료에서 {field} 금액을 찾아줘.'],
         'test':[f'{field}에 해당하는 금액과 출처는?']}[split]
    q=rng.choice(ask);answer=f'{a}만원 [{s}]'
    doc=f'[{s}] 출장 규정: {field}는 {a}만원이다.'
    if kind=='name':
        doc=f'[{s}] 업무 안내: 정산 담당자는 {name}이다.'
        q={'train':['정산 담당자는 누구야?','담당자 이름을 알려줘.'],'valid':['정산을 담당하는 사람은 누구인가요?'],'test':['정산 담당자 성명과 출처를 알려줄래?']}[split][i%len({'train':[0,1],'valid':[0],'test':[0]}[split])]
        answer=f'{name} [{s}]'
    elif kind=='noise':
        lines=[doc,f'[{t}] 출장 규정: {other}는 {b}만원이다.',f'[{u}] 행사 안내: 접수 기간은 {b}일이다.']
        rng.shuffle(lines);doc='\n'.join(lines)
    elif kind=='table':
        lines=[f'{field} | {a}만원',f'{other} | {b}만원'];rng.shuffle(lines)
        doc=f'[{s}] 출장비 표\n항목 | 금액\n'+'\n'.join(lines)
    elif kind=='missing':
        doc=f'[{s}] 출장 규정: {other}는 {b}만원이다.';answer='확인 불가'
    elif kind=='conflict':
        lines=[doc,f'[{t}] 출장 규정: {field}는 {b}만원이다.'];rng.shuffle(lines)
        doc='\n'.join(lines);answer='근거 충돌'
    elif kind=='revision':
        lines=[f'[{s}] 현행 규정: {field}는 {a}만원이다.',f'[{t}] 폐지된 규정: {field}는 {b}만원이다.'];rng.shuffle(lines)
        doc='\n'.join(lines)
    elif kind=='combine':
        lines=[doc,f'[{t}] 업무 안내: 정산 담당자는 {name}이다.'];rng.shuffle(lines)
        doc='\n'.join(lines);q=f'{field}와 정산 담당자를 함께 알려줘.'
        answer=f'{a}만원 [{s}], {name} [{t}]'
    prompt=PREFIX+doc+'\n질문: '+q
    return {'id':f'{split}-{kind}-{i:04d}','prompt':prompt,'answer':answer,'task':'rag_'+kind,'group':'rag','source':'authored synthetic evidence; fictional policy and people; not actual institution facts'}

def main():
    global ROOT
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT));a=p.parse_args();ROOT=Path(a.output)
    if ROOT.exists():raise ValueError('Use a new output directory')
    rng=random.Random(271027);rows={}
    for split,n in [('train',400),('valid',12),('test',12)]:
        rows[split]=[example(rng,k,split,i) for k in KINDS for i in range(n)]
        unique={r['prompt']:r for r in rows[split]};rows[split]=list(unique.values())
    # Keep base dialogue: reading is an added capability, not the only response mode.
    for split in ['train','valid']:
        rows[split]+=[json.loads(x) for x in (BASE/f'{split}.jsonl').read_text().splitlines()]
    # Broader basic phrasing; these are development additions after the v1 diagnostic.
    additions={
      'greeting':['처음 뵙겠습니다','잠깐 얘기 나눠요','여기서 대화 시작해도 될까요','반갑습니다 트리드','좋은 아침이에요','이야기하러 왔어'],
      'thanks':['도움을 받아서 고마워','답변 덕분에 이해했어요','덕분에 해결됐어요','친절하게 알려줘서 고맙다','많이 도움이 됐네요'],
      'farewell':['나중에 다시 올게요','이제 종료할게요','다음에 얘기해요','오늘은 이만 갈게요','여기까지 얘기하자'],
      'tired':['일하느라 너무 피곤하네요','온종일 바빠서 지쳤어요','퇴근했더니 힘이 없네','오늘 일이 힘들었어','좀 쉬고 싶어요'],
      'error':['방금 답은 잘못된 내용이야','그 설명에는 오류가 있어요','틀렸으니 확인해줘','다시 검토해 줄래','내 질문을 오해한 것 같아'],
      'clarify':['그거 해줘','그 건을 도와줘','알아서 해주겠니','이것 좀 부탁해','처리 부탁합니다'],
      'human':['너도 마음이 있어','정말 감정을 갖고 있니','인간처럼 느끼는 거야','기계야 사람이야','사람과 같은 경험을 해봤니'],
      'identity':['트리드라고 부르면 되니','너를 뭐라고 부를까','프로그램 이름 좀 알려줘','자기 이름을 소개해줘','너 이름이 trid 맞니'],
      'lonely':['오늘 외로워서 얘기하고 싶어','내 이야기를 들어줄래','마음이 허전해','대화 상대가 필요해','혼자 있으니 적적하네요'],
      'plan':['할 일을 정리하는 법이 궁금해','업무를 시작하기 전에 순서를 정하고 싶어','막막한 업무를 나눠보자','뭐부터 해야 할지 모르겠네요','일의 순서를 잡는 방법을 알려줘']}
    answers={r['task']:r['answer'] for r in rows['train'] if r['group']=='basic'}
    existing={r['prompt'] for r in rows['train']+rows['valid']}
    for key,questions in additions.items():
        for q in questions:
            if q not in existing:
                rows['train'].append({'prompt':q,'answer':answers['basic_'+key],'task':'basic_'+key,'group':'basic','source':'authored extra development phrasing v2'})
                existing.add(q)
    masses={'basic':.35,'public_work':.20,'limits':.10,'instruction':.05,'multi':.05,'rag':.25}
    counts=Counter((r['group'],r['task']) for r in rows['train'])
    groups=Counter(g for g,t in counts)
    for r in rows['train']:
        r['weight']=masses[r['group']]*len(rows['train'])/(groups[r['group']]*counts[(r['group'],r['task'])])
    tok=Tokenizer.from_file('runs/trid-korean-base-v1/tokenizer.json')
    keys={}
    for split,data in rows.items():
        keys[split]={json.dumps([r.get('history',[]),r['prompt']],ensure_ascii=False,sort_keys=True) for r in data}
        for r in data:encode_example(tok,r,512)
    assert not keys['train'] & keys['valid']
    assert not keys['train'] & keys['test']
    assert not keys['valid'] & keys['test']
    ROOT.mkdir(parents=True)
    for split,data in rows.items():
        rng.shuffle(data);(ROOT/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data))
    provenance={'purpose':'Korean evidence-use curriculum mixed with file-free base dialogue; no retriever involved in these tests',
      'data':'Authored fictional documents. No external weights, APIs, or chatbot outputs.',
      'split':'Within newly authored RAG examples, disjoint numeric values and names, varied question phrasings except combine, disjoint whole prompts. Values may appear in earlier generic training or pretraining. Shared synthetic templates; not independent real-document benchmark.',
      'v1_diagnostic':'Previously inspected 56-turn evaluation is now development/regression only; do not claim a new unseen test result on it.',
      'sampling_mass':masses,'rows':{k:len(v) for k,v in rows.items()},
      'hashes':{k:hashlib.sha256((ROOT/f'{k}.jsonl').read_bytes()).hexdigest() for k in rows},
      'scope':'Short evidence only, <=512-token context. Does not establish long PDF or end-to-end retrieval quality.',
      'research_reference':'https://arxiv.org/abs/2403.10131 ; inspired by relevant evidence plus distractors, not a full reproduction of RAFT'}
    shutil.copyfile('data/korean-rag-curriculum-v1-source/base-check.jsonl',ROOT/'base-check.jsonl')
    provenance['hashes']['base-check']=hashlib.sha256((ROOT/'base-check.jsonl').read_bytes()).hexdigest()
    (ROOT/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
    print(json.dumps(provenance,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
