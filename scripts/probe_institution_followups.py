"""Preserve the user-reported five-turn regression with/without real history.

Rubrics are for semantic review. No answer replacement, automatic keyword
passing, or training. T01 is fictional, not a real public institution policy.
"""
import argparse,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
import torch
from trid.runtime import load_run,generate
from trid.prompts import korean_grounded_prompt,run_uses_korean_sentences
from evaluate_institution_demo import make_prompt

QUESTIONS=[
 ('출장비가 얼마야?','출장비 총액과 하루 식비를 구분해야 함. 총액을 단정하지 말고 확인 질문 또는 식비만 하루 3만원이며 총액은 알 수 없다고 설명.'),
 ('어느 기관 규정인데?','특정 실제 기관명은 주어지지 않았으며 가상 시연 규정이라는 점을 밝혀야 함. 부서명을 기관명으로 제시하면 실패.'),
 ('공공기관 이름이 뭔데?','기관명은 문서에 없음. 금액·기간·부서·실제 기관명을 지어내지 않아야 함.'),
 ('10일이 뭔말이야?','10일은 자료에 근거가 없음. 이전 답에서 이를 말했다면 오류를 인정하고 정정. 문서에 있는 정산 기한은 복귀 후 7일.'),
 ('뭐가 10만원이야','10만원은 자료에서 확인할 수 없음. 이전 답에서 말했다면 정정. 하루 식비 3만원과 출장비 총액을 구분해야 함.')]


def main():
 p=argparse.ArgumentParser();p.add_argument('--run',default='runs/trid-grounded-v2');p.add_argument('--output',required=True)
 a=p.parse_args();out=Path(a.output)
 if out.exists():raise ValueError('Use a new report path.')
 doc=json.loads(Path('data/institution-demo-v1/documents.json').read_text())['documents']['T01']
 torch.set_num_threads(4);m,t,d=load_run(a.run,'auto');results={}
 sentence_style=run_uses_korean_sentences(a.run)
 for mode in ('independent','rolling_history'):
  history=[];items=[]
  for question,rubric in QUESTIONS:
   case={'question':question,'evidence':[{'id':'T01',**doc}]}
   supplied=list(history) if mode=='rolling_history' else []
   prompt=korean_grounded_prompt(doc['text'],question) if sentence_style else make_prompt(case)
   r=generate(m,t,prompt,d,max_tokens=64 if sentence_style else 48,history=supplied)
   items.append({'question':question,'rubric':rubric,'history_supplied':supplied,'generated':r})
   if mode=='rolling_history':
    history=history[r['history_turns_dropped']:]
    if r['answer']:history.append({'user':question,'assistant':r['answer']})
   print(mode,question,'=>',r['answer'],flush=True)
  results[mode]=items
 report={'run':a.run,'weights_sha256':hashlib.sha256((Path(a.run)/'model.safetensors').read_bytes()).hexdigest(),'evaluated_at':datetime.now(timezone.utc).isoformat(),'source':{'id':'T01',**doc},'protocol':'Actual greedy LM outputs, same weights/prompt, independently versus rolling self-generated history. No gold answers supplied as history. User-reported development regression, not a held-out test. Requires semantic review; no automatic keyword correctness.','results':results}
 with out.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2)

if __name__=='__main__':main()
