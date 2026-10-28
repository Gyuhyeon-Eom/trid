"""Run real local inference over operator-selected fictional policy excerpts."""
import argparse
import json
from pathlib import Path
import torch

from trid.runtime import load_run, generate
from trid.terminal_input import classify_input
from trid.prompts import korean_grounded_prompt, run_uses_korean_sentences
from evaluate_institution_demo import make_prompt, choose

ROOT=Path(__file__).resolve().parents[1]


def default_institution_run():
    selection=ROOT/'active-institution-model.json'
    relative=json.loads(selection.read_text())['run'] if selection.exists() else 'runs/trid-institution-v1'
    path=(ROOT/relative).resolve()
    if not path.is_relative_to((ROOT/'runs').resolve()):
        raise ValueError('기관 모델은 프로젝트 runs 폴더 안에 있어야 합니다.')
    return str(path)


def main():
    p=argparse.ArgumentParser(description='trid 가상 기관 규정 시연 — 직접 선택한 자료로 답변 생성')
    p.add_argument('question',nargs='?')
    p.add_argument('--chat',action='store_true',help='선택한 자료로 질문을 반복하는 터미널 실행')
    p.add_argument('--source',required=True,nargs='+',help='가상 문서 ID: T01 P01 L01 I01 R01 V01')
    p.add_argument('--run',default=default_institution_run())
    p.add_argument('--documents',default=str(ROOT/'data/institution-demo-v1/documents.json'),help='가상 문서 JSON; 원본 평가 자료는 보존하고 복사본으로 실험하세요')
    p.add_argument('--choices',nargs=3,help='선택지 세 개를 직접 지정하면 선택 모드로 실행')
    p.add_argument('--device',default='auto',choices=['auto','cpu','mps'])
    args=p.parse_args()
    if not args.chat and not args.question:
        p.error('질문을 입력하거나 --chat을 지정하세요.')
    if args.chat and (args.question or args.choices):
        p.error('--chat은 질문 인자 및 --choices와 함께 사용할 수 없습니다.')
    extractive=(Path(args.run)/'reader.safetensors').exists()
    sentence_style=not extractive and run_uses_korean_sentences(args.run)
    if extractive and args.choices:
        p.error('문서 구간 선택 모델은 A/B/C 모드를 지원하지 않습니다.')
    corpus=json.loads(Path(args.documents).read_text())
    if not corpus.get('fictional'):
        raise ValueError('This demo requires explicitly fictional documents.')
    documents=corpus['documents']
    if not all(key in documents for key in args.source):
        p.error('알 수 없는 문서 ID입니다. T01 P01 L01 I01 R01 V01 중 선택하세요.')
    if args.choices and len(set(args.choices))!=3:
        p.error('선택지는 서로 달라야 합니다.')
    print('시연용 가상 규정입니다. 실제 기관의 규정이나 법령이 아닙니다.')
    print('기관 규정 실험 모델입니다. 자료가 있어도 틀릴 수 있으므로 아래 자료와 답변을 비교하세요.')
    def show_sources():
        print('사용자가 선택한 자료 (자동 검색 아님):')
        for key in args.source:
            print(f"[{key}] {documents[key]['title']}\n{documents[key]['text']}")
    show_sources()
    torch.set_num_threads(4)
    if extractive:
        from trid.reader import load_reader,read
        model,tok,device=load_reader(args.run,args.device)
        print('문서 구간 선택 모드: 모델이 고른 원문 구간을 표시합니다. 자유 문장 생성이 아닙니다.')
    else:
        model,tok,device=load_run(args.run,args.device)
        print('문장 생성 모드: 모델이 다음 토큰을 순서대로 생성합니다.')
    print(f'모델: {Path(args.run).resolve()} | {device}')
    if args.chat:
        print('각 질문은 독립적으로 처리합니다.' if extractive else '문맥 한도 안에서 최근 대화를 전달합니다. 이전 오답을 정정하는 능력은 아직 부족합니다.')
        print('대화는 메모리에서만 유지하며 파일에 저장하지 않습니다.')
        print('/sources 문서 목록 · /source T01 자료 선택 · /evidence 선택 자료 보기 · /reset 대화 초기화 · /exit 종료')
    history=[]
    while True:
        try:
            question=input('\n나 [기관 규정] > ').strip() if args.chat else args.question
        except (KeyboardInterrupt,EOFError):
            print('\n종료했습니다.')
            break
        if args.chat:
            kind,value=classify_input(question)
            if kind=='empty':continue
            if kind=='command':
                if value=='exit':break
                if value=='reset':
                    history.clear()
                    print('대화 이력을 초기화했습니다.')
                    continue
                print('명령: /sources · /source 문서ID · /evidence · /reset · /exit')
                continue
            if kind=='notice' and not question.startswith('/'):
                print(value)
                continue
            if question=='/exit':break
            if not question:continue
            if question=='/sources':
                for key,value in documents.items():print(f"{key}: {value['title']}")
                continue
            if question=='/evidence':
                show_sources()
                continue
            if question.split()[0]=='/source':
                selected=question.split()[1:]
                if not selected or any(key not in documents for key in selected):
                    print('올바른 문서 ID를 지정하세요. /sources로 목록을 확인할 수 있습니다.')
                    continue
                args.source=list(dict.fromkeys(selected))
                history.clear()
                print('자료 선택 변경으로 이전 대화 이력을 초기화했습니다.')
                show_sources()
                continue
            if question.startswith('/'):
                print('명령: /sources · /source 문서ID · /evidence · /exit')
                continue
        case={'id':'interactive','question':question,
              'evidence':[{'id':key,**documents[key]} for key in args.source]}
        print(f"\n질문: {question} | 자료: {', '.join(args.source)}")
        try:
            if extractive:
                result=read(model,tok,make_prompt(case),device)
                print(f"trid [문서 구간 선택] > {result['answer']}")
                print(f"[판정: {result['kind']} · 전체 {result['total_seconds']:.3f}초]")
            elif args.choices:
                result=choose(model,tok,case,args.choices,device)
                print(f"trid 선택: {result['label']} — {result['answer']}")
                print('세 선택지로 출력을 제한한 자체 모델 실험입니다. 확률 보정은 적용하지 않았습니다.')
            else:
                evidence='\n'.join(e['text'] for e in case['evidence'])
                prompt=korean_grounded_prompt(evidence,question) if sentence_style else make_prompt(case)
                result=generate(model,tok,prompt,device,max_tokens=64 if sentence_style else 48,history=history)
                print(f"trid [기관 규정] > {result['answer']}")
                print(f"첫 토큰 {result['first_token_seconds']:.3f}초 · 전체 {result['total_seconds']:.3f}초")
                if args.chat:
                    dropped=result['history_turns_dropped']
                    if dropped:
                        history=history[dropped:]
                        print(f'[문맥 한도로 이전 대화 {dropped}턴을 제외했습니다.]')
                    print(f'[모델에 전달한 이전 대화: {len(history)}턴]')
                    if result['answer']:
                        history.append({'user':question,'assistant':result['answer']})
        except ValueError as error:
            if not args.chat:raise
            print(f'입력 오류: {error}')
        if not args.chat:break


if __name__=='__main__':
    main()
