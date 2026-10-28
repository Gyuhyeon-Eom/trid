"""Train contrasts between model identity, user name, and short follow-ups."""
import hashlib
import json
from pathlib import Path
import random
from tokenizers import Tokenizer
from trid.text import encode_example
from prepare_dialogue import normalize

root=Path(__file__).resolve().parents[1]
out=root/'data/dialogue-v5'
out.mkdir(exist_ok=False)
rng=random.Random(53)
tok=Tokenizer.from_file(str(root/'data/wiki-ko-tokens/tokenizer.json'))
rows=[json.loads(l) for l in (root/'data/dialogue-v4/train.jsonl').read_text().splitlines()]
valid=[json.loads(l) for l in (root/'data/dialogue-v4/valid.jsonl').read_text().splitlines()]
held={normalize(json.loads(l)['prompt']) for fn in ['dialogue-eval-dev.jsonl','dialogue-eval-final.jsonl'] for l in (root/'data'/fn).read_text().splitlines()}
seen={json.dumps([r.get('history',[]),normalize(r['prompt'])],sort_keys=True,ensure_ascii=False) for r in rows+valid}

def add(q,a,topic,history=None,weight=6):
    r={'prompt':q,'answer':a,'task':topic,'weight':weight,'source':'authored conversational contrasts'}
    if history:r['history']=history
    k=json.dumps([r.get('history',[]),normalize(q)],sort_keys=True,ensure_ascii=False)
    if k in seen or normalize(q) in held:return
    try:encode_example(tok,r,256)
    except ValueError:return
    rows.append(r);seen.add(k)

identity=['네 이름은 뭐야?','너 이름 뭐야?','너의 이름은 뭐니?','너는 이름이 뭐니?','너 이름이 뭐였지?',
          '네 이름을 말해 줘','너는 뭐라고 불러?','넌 이름이 뭐야?','네 이름은 무엇이야?','너 자신을 소개해봐']
answers=['제 이름은 trid예요. 이 컴퓨터에서 작동하는 작은 언어모델입니다.',
         '저는 trid입니다. 한국어 대화를 배우고 있는 작은 언어모델이에요.']
greetings=[{'user':'안녕','assistant':'안녕하세요! 저는 trid입니다. 편하게 이야기해 주세요.'},
           {'user':'안녕하세요','assistant':'반가워요! 오늘은 어떤 이야기를 나눌까요?'},
           {'user':'반가워','assistant':'저도 반가워요. 무엇을 이야기할까요?'}]
contexts=[[],*[ [g] for g in greetings]]
for name in ['민수','수빈','지훈','하늘','가람','예린','나래','도윤','유진','지민','민지','철수']:
    h=[greetings[0],{'user':f'내 이름은 {name}야.','assistant':f'반가워요, {name}님. 저는 trid입니다.'}]
    contexts.append(h)
    for q in ['내 이름은 뭐야?','내 이름이 뭐야?','아까 내가 누구라고 했어?','나를 뭐라고 부르기로 했어?','내가 누구라고 소개했지?']:
        add(q,f'이름은 {name}라고 말씀하셨어요.','name_memory',h)
for q in identity:
    for h in contexts:
        for prefix in ['','그런데 ','그럼 ']:add(prefix+q,rng.choice(answers),'identity',h)
for q in ['내 이름이 뭐야?','내 이름은 뭐야?','내 이름을 알고 있니?','내가 누구인지 알아?']:
    for h in [[],[greetings[0]]]:add(q,'아직 이름을 알려주지 않으셨어요. 어떻게 불러 드리면 될까요?','name_memory',h)

for q in ['오늘 할 일이 많아','할 일이 너무 많네','오늘 일이 많아서 바빠','업무가 많이 밀렸네','해야 하는 게 너무 많아','오늘 업무를 다 할 수 있을까','쌓인 일부터 처리해야겠어']:
    for h in contexts:
        add(q,'해야 할 일을 적고 가장 급한 것부터 하나씩 해 보세요.','plan',h)
followups=[
 ('오늘 너무 피곤해','오늘 고생 많으셨어요. 잠깐 쉬면서 숨을 돌려 보세요.',
  ['응 좀 쉴게','그래 조금 쉬어야겠어','잠깐 쉬고 올게','응 쉬었다 다시 할게','한숨 돌리고 올게','그래 좀 쉬자'], '좋아요. 편하게 쉬고 오세요.'),
 ('오늘 할 일이 많아','해야 할 일을 적고 가장 급한 것부터 하나씩 해 보세요.',
  ['하나씩 해 볼게','응 먼저 급한 것부터 할게','그럼 할 일을 적어 볼게','좋아 차근차근 해야겠어','알았어 시작할게','그래 목록부터 써 볼게'], '좋아요. 하나씩 차근차근 해보세요.'),
 ('점심 뭐 먹지?','비빔밥이나 샌드위치는 어떠세요?',
  ['비빔밥 먹을래','샌드위치로 정했어','그걸 먹어야겠다','좋아 그 메뉴로 할게'], '좋아요. 맛있게 드세요!')]
for q,a,replies,response in followups:
    for prior in contexts:
        for reply in replies:
            add(reply,response,'followup',prior+[{'user':q,'assistant':a}])

rng.shuffle(rows)
for s,rs in [('train',rows),('valid',valid)]:
    (out/f'{s}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs))
manifest={'source':'v4 and authored conversational contrast curriculum',
 'evaluation':'Development conversations informed contrasts; fixed final prompts excluded and still untested.',
 'counts':{'train':len(rows),'valid':len(valid)},'seed':53,
 'sha256':{s:hashlib.sha256((out/f'{s}.jsonl').read_bytes()).hexdigest() for s in ['train','valid']}}
(out/'provenance.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print(json.dumps(manifest,ensure_ascii=False,indent=2))
