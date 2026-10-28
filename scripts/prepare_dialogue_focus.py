"""A deliberately narrow, authored dialogue curriculum. No inference lookup."""
import hashlib
import json
import random
from pathlib import Path

from tokenizers import Tokenizer
from trid.text import encode_example
from prepare_dialogue import normalize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/dialogue-v4'
rng = random.Random(121)
tok = Tokenizer.from_file(str(ROOT/'data/wiki-ko-tokens/tokenizer.json'))
heldout = {normalize(json.loads(line)['prompt'])
           for name in ['dialogue-eval-dev.jsonl','dialogue-eval-final.jsonl']
           for line in (ROOT/'data'/name).read_text().splitlines()}
rows = {split: [json.loads(line) for line in (ROOT/'data/dialogue-v3'/f'{split}.jsonl').read_text().splitlines()
                if json.loads(line)['task'] != 'public_daily'] for split in ['train','valid']}
seen = {json.dumps([r.get('history',[]),normalize(r['prompt'])],ensure_ascii=False,sort_keys=True)
        for split in rows.values() for r in split}

def add(q,a,topic,history=None):
    if normalize(q) in heldout:
        return
    row = {'prompt':q,'answer':a,'task':topic,'weight':1,'source':'authored focus curriculum'}
    if history:
        row['history']=history
    key=json.dumps([row.get('history',[]),normalize(q)],ensure_ascii=False,sort_keys=True)
    if key in seen:
        return
    try:
        encode_example(tok,row,256)
    except ValueError:
        return
    seen.add(key)
    rows['train'].append(row)

intents={r['topic']:r for r in json.loads((ROOT/'data/dialogue-intents.json').read_text())}
extras={
 'repeat':['같은 표현이 또 나오는데 왜 그래','전에 쓴 말을 다시 쓰네','비슷한 말만 여러 차례 하는구나','한 말을 또 하고 있어','대답이 자꾸 되풀이돼','문장이 중복되는 것 같은데','또 똑같은 표현을 쓰네'],
 'cold':['창문 사이로 찬바람이 불어와','문을 여니까 방이 추워지네','차가운 바람이 들어와서 쌀쌀하다','방 안에 냉기가 도네','쌀쌀한 공기가 창으로 들어와','밖에 바람이 차갑구나'],
 'tired':['바쁘게 일하고 나니 기운이 빠져','일하느라 지쳐서 휴식이 필요해','여러 시간 일하고 나니 피곤하다','계속 일해서 힘이 하나도 없어','업무를 마치니 몸이 무겁고 피곤해'],
 'weather':['지금 대전에 비가 오는 중인지 궁금해','부산에는 현재 비가 오나요','오늘 강릉 날씨를 확인할 수 있니','지금 눈이 내리는지 확인해 줘','우리 지역 기온이 몇 도인지 알려 줘'],
 'identity':['이야기하고 있는 상대를 뭐라고 불러야 하지','네 이름은 어떻게 부르면 되는 거야','인공지능 이름이 뭐였더라','지금 말하고 있는 너를 소개해 줘'],
 'thanks':['설명이 도움이 되어서 고맙네','가르쳐 준 내용을 잘 이해했어 고마워','네가 알려준 덕에 도움이 됐어','도와준 덕분에 한결 나아졌어 고마워']}
for topic,questions in extras.items():
    for q in questions:
        for prefix in ['', '있잖아, ', '한 가지 말하자면 ']:
            add(prefix+q,rng.choice(intents[topic]['answers']),topic)

# Separate weather observations from requests for information the model cannot access.
for city in ['서울','부산','제주','대전','광주','대구','인천','수원','강릉','울산']:
    for condition in ['비','눈']:
        particle='가' if condition=='비' else '이'
        for q in [f'{city}에 지금 {condition}{particle} 오는지 알려줘',f'지금 {city}에서 {condition}{particle} 내리는지 알 수 있어?',f'{city}의 오늘 날씨는 어때?']:
            add(q,rng.choice(intents['weather']['answers']),'weather')

# Contextual copying with many names, varied introductions and recall wording.
names=['가람','나래','다온','라온','마루','민수','지훈','지민','하늘','수빈','유진','지원','도윤','예린','수현','지우',
       '민준','서준','준서','준우','현우','도현','건우','선우','은우','시우','정우','지호','승우','승현',
       '민지','수진','혜진','소연','지연','은지','은서','예은','서윤','하윤','유나','예나','아린','채원','채린','소윤','민아','수아','수영','영희','철수','동현','태민','성민']
recalls=['내 이름이 뭐야?','아까 말한 이름을 알려줘','내 이름 기억해?','내가 알려준 이름이 뭐였어?',
         '나를 뭐라고 부르기로 했지?','조금 전에 내가 누구라고 했지?','내가 말해준 내 이름을 기억하니?',
         '내 이름을 기억하고 있으면 말해 줘','방금 내 이름을 뭐라고 말했어?','앞에서 알려준 이름이 궁금해']
for name in names:
    for intro in [f'내 이름은 {name}야.',f'나는 {name}라고 해.',f'{name}라고 불러 줘.']:
        add(intro,f'반가워요, {name}님. 저는 trid입니다.','name_introduction')
        for question in recalls:
            greeting=rng.choice([f'반가워요, {name}님.','반가워요! 저는 trid입니다.'])
            add(question,f'이름은 {name}라고 말씀하셨어요.','name_memory',[{'user':intro,'assistant':greeting}])

# Rephrase request framing without inserting held-out evaluation prompts.
singles=[r.copy() for r in rows['train'] if not r.get('history')]
for row in singles:
    q=row['prompt'].rstrip('.!?')
    if q.startswith(('저기','음,','그런데','있잖아')):
        continue
    for prefix,suffix in [('','.'),('','!'),('있지, ',''),('내가 하고 싶은 말은, ',''),('',' 어떻게 생각해?')]:
        add(prefix+q+suffix,row['answer'],row['task'])

# Sample 1–3 earlier turns, while keeping the current answer tied to the current question.
singles=[r.copy() for r in rows['train'] if not r.get('history')]
for row in singles:
    for count in [1,2,3]:
        prior=rng.sample(singles,count)
        history=[{'user':p['prompt'],'assistant':p['answer']} for p in prior]
        add(row['prompt'],row['answer'],row['task'],history)

OUT.mkdir(exist_ok=False)
for split,examples in rows.items():
    for row in examples:
        row['weight']=1
    rng.shuffle(examples)
    (OUT/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in examples))
provenance={'purpose':'Narrow basic dialogue tuning after the mixed public-data stage. Not a general conversation corpus.',
 'source':'authored dialogue from v3, plus this deterministic builder; no new downloads or external model outputs',
 'counts':{s:len(r) for s,r in rows.items()},'seed':121,
 'evaluation':'Exact normalized dev/final questions excluded; dev errors informed curriculum, so only final remains held out.',
 'validation':'Existing authored v3 validation; same topic classes as training; no public daily validation.',
 'sha256':{s:hashlib.sha256((OUT/f'{s}.jsonl').read_bytes()).hexdigest() for s in rows}}
(OUT/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
print(json.dumps(provenance,ensure_ascii=False,indent=2))
