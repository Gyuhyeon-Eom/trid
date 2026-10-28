"""Authored, file-free Korean/public-work curriculum. No retrieval or pretrained imports."""
from collections import Counter
import hashlib,json,random,re,unicodedata
from pathlib import Path
from tokenizers import Tokenizer
from trid.text import encode_example

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/base-dialogue-v1-source'
OUT=ROOT/'data/base-dialogue-v1'

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(s):return re.sub(r'[^가-힣a-z0-9]','',unicodedata.normalize('NFC',s).lower())

def main():
    if OUT.exists():raise ValueError('Use a fresh data directory')
    rng=random.Random(271026)
    evaluation=[json.loads(x) for x in (SOURCE/'evaluation.jsonl').read_text().splitlines()]
    heldout={norm(t['prompt']) for r in evaluation for t in r.get('turns',[r])}
    tok=Tokenizer.from_file(str(ROOT/'runs/trid-korean-base-v1/tokenizer.json'))
    parts={'train':[],'valid':[]};seen=set();counts=Counter()
    def add(split,q,a,group,task,history=None):
        if norm(q) in heldout or any(norm(h['user']) in heldout for h in history or []):
            counts['excluded_frozen_prompt']+=1;return
        key=json.dumps([history or [],norm(q)],ensure_ascii=False,sort_keys=True)
        if key in seen:return
        row={'prompt':q,'answer':a,'task':task,'group':group,'source':'authored file-free curriculum v1'}
        if history:row['history']=history
        try:encode_example(tok,row,512)
        except ValueError:counts['over_length']+=1;return
        seen.add(key);parts[split].append(row)
    topics=[]
    expansions=json.loads((ROOT/'data/dialogue-expansion.json').read_text())
    for t in json.loads((ROOT/'data/dialogue-intents.json').read_text()):
        if t['topic'] in {'meeting','email','short','formal','unknown','understand','travel'}:continue
        qs=list(dict.fromkeys(t['questions']+expansions.get(t['topic'],[])))
        answers=t['answers']
        replacements={
          'identity':['저는 한국어로 대화하고 업무 설명을 돕도록 학습 중인 작은 언어모델 trid입니다.'],
          'capability':['기본적인 한국어 대화와 업무 설명, 짧은 문장 작성을 돕습니다. 특정 기관의 규정이나 최신 정보는 확인이 필요합니다.'],
          'internet':['현재 인터넷 검색 기능은 없습니다. 실시간 정보는 공식 안내나 해당 서비스에서 확인해 주세요.'],
          'privacy':['개인정보나 비밀 자료는 필요 이상으로 입력하지 않는 것이 좋습니다. 서비스의 실제 저장 방식과 접근 권한은 운영 설정을 확인해야 합니다.'],
          'memory':['현재 대화에서 알려주신 내용은 짧은 대화 이력 안에서 참고할 수 있습니다. 대화가 길어지거나 새로 시작하면 이전 정보를 기억하지 못할 수 있어요.'],
          'learn':['학습된 내용을 바탕으로 답을 생성합니다. 지금 대화하는 것만으로 모델의 가중치가 자동으로 바뀌지는 않습니다.'],
          'limitations':['작은 실험 모델이라 질문을 잘못 이해하거나 틀린 답을 할 수 있습니다. 중요한 업무 내용은 공식 자료와 함께 확인해 주세요.'],
        }
        topics.append(('basic','basic_'+t['topic'],qs,replacements.get(t['topic'],answers)))
    for line in (SOURCE/'topics.tsv').read_text().splitlines():
        group,key,questions,answer=line.split('\t')
        topics.append((group,'work_'+key,questions.split(';'),[answer.replace('\\n','\n')]))
    # Split base phrasings before augmentation; no prefix siblings cross the split.
    wrappers=[('{}','{}'),('{}, 간단히 알려줘.','{}'),('궁금한 게 있어. {}','{}'),('업무 중인데, {}','{}'),('질문이 있어요. {}','{}'),('도와줄래? {}','{}')]
    for group,key,qs,answers in topics:
        valid_q=qs[-1]
        for q in qs[:-1]:
            for prefix,_ in wrappers:
                # Do not attach work-specific filler to everyday greetings.
                if group=='basic' and prefix.startswith('업무 중'):continue
                add('train',prefix.format(q),rng.choice(answers),group,key)
        add('valid',valid_q,answers[0],group,key)
        if group in {'public_work','instruction'}:
            short=answers[0].split('. ')[0].rstrip('.')+'.'
            for q in qs[:-1]:
                add('train',q+' 짧게 말해 줘.',short,group,key)
                add('train','짧게 다시 말해 줘.',short,'multi','followup_summary',[{'user':q,'assistant':answers[0]}])
                add('train',q,answers[0],group,key,[{'user':'안녕하세요.','assistant':'안녕하세요! 저는 trid입니다. 무엇을 도와드릴까요?'}])
            add('valid','한 문장으로 다시 설명해 줄래?',short,'multi','followup_summary',[{'user':valid_q,'assistant':answers[0]}])
    # Composition/copy tasks use different values in validation, and evaluation values are separate.
    values={
      'train':{'names':['김서연','박민준','이수빈','최하린','정도윤','강지우','서유진','오현수','배은서','한지민'], 'rooms':['다솜실','누리실','새봄실','한빛실','소나무실'], 'amounts':['2만원','5만원','6만원','9만원','11만원']},
      'valid':{'names':['문세아','임준우'], 'rooms':['바다실','솔빛실'], 'amounts':['7만원','13만원']}}
    for split,v in values.items():
        for name in v['names']:
            for q in [f'담당자는 {name}입니다. 담당자 이름은?',f'담당자: {name}. 담당자가 누구야?',f'담당자 이름만 답해 줘. 담당자는 {name}야.']:
                add(split,q,name,'instruction','copy_name')
            for intro in [f'내 이름은 {name}이야.',f'나는 {name}이라고 해.']:
                answer=f'반가워요, {name}님. 어떤 이야기를 나눌까요?'
                add(split,intro,answer,'basic','name_intro')
                for q in ['내 이름이 뭐라고 했지?','아까 말한 내 이름을 알려줘.']:
                    add(split,q,name+'님이라고 하셨어요.','multi','remember_name',[{'user':intro,'assistant':answer}])
            intro=f'담당자는 {name}이야.';answer=f'담당자는 {name}님이군요.'
            for q in ['담당자가 누구라고 했지?','방금 말한 담당자 이름은?']:
                add(split,q,name,'multi','remember_name',[{'user':intro,'assistant':answer}])
        for room in v['rooms']:
            for q in [f'회의 장소는 {room}입니다. 장소는 어디야?',f'회의는 {room}에서 열립니다. 회의 장소를 알려줘.',f'장소: {room}. 회의 장소만 답해.']:
                add(split,q,room,'instruction','copy_room')
        for amount in v['amounts']:
            intro=f'우리 기관의 하루 출장 식비는 {amount}이야.';answer=f'알려주신 기준은 하루 식비 {amount}입니다.'
            add(split,intro,answer,'multi','supplied_policy')
            for q in ['하루 식비가 얼마라고 했지?','방금 알려준 식비 기준을 말해 줘.']:
                add(split,q,amount+'이라고 알려주셨어요.','multi','remember_policy',[{'user':intro,'assistant':answer}])
        numbers=range(1,9) if split=='train' else range(9,11)
        for a in numbers:
            b=a+3
            for q in [f'교육일이 {a}일에서 {b}일로 변경됐어. 안내 문장을 써 줘.',f'기존 교육은 {a}일이고 변경된 교육은 {b}일이야. 변경 안내를 작성해 줘.']:
                add(split,q,f'교육 일정이 {a}일에서 {b}일로 변경되었으니 확인해 주시기 바랍니다.','instruction','changed_schedule')
    dialogues=[
      ('출장 정산 자료를 준비하고 있어.','출장 일정과 영수증 등 증빙을 정리하고 기관의 정산 규정을 확인해 보세요.','기한이 7일로 정해진 거야?','기관마다 정산 기한이 다를 수 있습니다. 소속 기관의 규정을 확인해야 합니다.'),
      ('팀 회의를 준비하려고 해.','목적과 안건, 참석자, 일시와 장소를 정하고 필요한 자료를 준비해 보세요.','미리 뭘 보내면 돼?','회의 안건, 일시와 장소, 검토할 자료를 참석자에게 미리 공유하면 좋습니다.'),
      ('공문을 작성하려고 해.','목적, 주요 내용, 요청 사항을 정리해 기관 서식에 맞춰 작성해 보세요.','어떤 순서로 적을까?','제목, 작성 목적, 주요 내용, 요청 사항 순서로 정리하면 좋습니다.'),
      ('사업 신청서를 접수했어.','접수한 서류와 접수 확인 내용을 보관하고 이후 안내를 확인해 보세요.','그럼 선정된 거지?','접수와 선정은 다릅니다. 접수했다고 선정이 확정된 것은 아니에요.'),
      ('민원인에게 답변해야 해.','문의 내용과 확인된 사실, 검토 결과를 구분해 정리해 보세요.','아직 결과를 몰라.','현재 확인 중인 사항과 이후 안내 방법을 설명하고, 확정되지 않은 결과를 약속하지 마세요.'),
      ('너의 답이 이상해.','제가 질문을 잘못 이해했을 수 있어요. 어떤 부분인지 알려주시면 다시 답해 볼게요.','식비를 물었는데 숙박비로 답했어.','제가 비용 항목을 혼동했네요. 식비는 식사 비용이고 숙박비는 숙소 비용입니다.'),
    ]
    for i,(q,a,q2,a2) in enumerate(dialogues):
        for prefix in ['','그런데 ','그러면 ','한 가지 더, ']:
            add('train',prefix+q2,a2,'multi','coherent_followup',[{'user':q,'assistant':a}])
        add('valid','추가로 궁금해. '+q2,a2,'multi','coherent_followup',[{'user':q,'assistant':a}])
    mass={'basic':.30,'public_work':.42,'limits':.13,'instruction':.09,'multi':.06}
    # Each task receives equal mass inside its group, independent of paraphrase count.
    task_counts=Counter((r['group'],r['task']) for r in parts['train'])
    tasks_per_group=Counter(g for g,t in task_counts)
    for r in parts['train']:
        r['weight']=mass[r['group']]/tasks_per_group[r['group']]/task_counts[r['group'],r['task']]*len(parts['train'])
    OUT.mkdir()
    for split,rows in parts.items():
        rng.shuffle(rows)
        (OUT/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    (OUT/'evaluation.jsonl').write_bytes((SOURCE/'evaluation.jsonl').read_bytes())
    provenance={'goal':'File-free Korean dialogue and generic public-institution work; not institution-specific factual authority',
      'initialization':'own randomly initialized model lineage only; no external pretrained weights',
      'source_files':{str(p.relative_to(ROOT)):digest(p) for p in [SOURCE/'topics.tsv',SOURCE/'evaluation.jsonl',ROOT/'data/dialogue-intents.json',ROOT/'data/dialogue-expansion.json']},
      'official_reference_notes':[
        {'url':'https://alio.go.kr/guide/publicAgencyList.do','checked':'2026-10-27','use':'ALIO is public-institution management disclosure; do not train a fixed current institution count'},
        {'url':'https://www.epeople.go.kr/inst/mois','checked':'2026-10-27','use':'General role of epeople: complaints/proposals; no legal deadlines or individual case claims'}],
      'content':'Authored synthetic Q&A and generic workflow suggestions, not actual employee messages or binding agency policy',
      'rows':{s:len(r) for s,r in parts.items()},'tasks':len(task_counts),'group_rows':{s:dict(Counter(r['group'] for r in rows)) for s,rows in parts.items()},
      'training_sampling_mass':mass,'context_length':512,'tokenizer_sha256':digest(ROOT/'runs/trid-korean-base-v1/tokenizer.json'),
      'split_policy':'Base phrasing split before prefix augmentation; explicit exact normalized frozen evaluation exclusion. Shared concepts and answer wording remain. New values in validation and test.',
      'evaluation_warning':'Small same-author held-out phrasing diagnostic, not an independent public benchmark; prior legacy dialogue exposure not fully excluded',
      'excluded':dict(counts),'file_sha256':{s:digest(OUT/f'{s}.jsonl') for s in ['train','valid','evaluation']}}
    (OUT/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
    print(json.dumps({k:provenance[k] for k in ['rows','tasks','group_rows','excluded']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
