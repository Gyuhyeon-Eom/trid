"""Local source-span reader; each question is independent."""
import argparse
from pathlib import Path
import torch
from trid.korean_reader import load_reader, read
from trid.terminal_input import classify_input


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='runs/trid-korean-span-v1')
    p.add_argument('--document',required=True,help='UTF-8 원문 파일')
    p.add_argument('--question')
    p.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    a=p.parse_args();document=Path(a.document).read_text(encoding='utf-8')
    if not document.strip():p.error('빈 문서입니다.')
    torch.set_num_threads(4);model,tok,device=load_reader(a.run,a.device)
    print(f'trid 구간 추출 실험 | {a.run} | {device}')
    print(f'문서: {Path(a.document).resolve()}')
    print('질문은 서로 독립적입니다. 문서 안의 잘못된 구간을 고를 수 있습니다. /exit 종료')
    def answer(question):
        r=read(model,tok,document,question,device)
        print('trid > '+r['answer'])
        if r['source_character_span'] is not None:
            start,end=r['source_character_span']
            print(f'원문 문자 위치 [{start}, {end}) · {r["total_seconds"]:.3f}초')
            print('근거 주변: '+document[max(0,start-30):start]+'【'+document[start:end]+'】'+document[end:end+30])
        else:print(f'자료 없음으로 판단 · {r["total_seconds"]:.3f}초')
    if a.question is not None:
        answer(a.question);return
    while True:
        try:
            kind,value=classify_input(input('나 [구간 추출] > '))
            if kind=='empty':continue
            if kind=='command':
                if value=='exit':break
                print('매 질문마다 같은 문서를 독립적으로 읽습니다. /exit 종료')
            elif kind=='notice':print(value)
            else:
                try:answer(value)
                except ValueError as error:print(f'입력 오류: {error}')
        except (EOFError,KeyboardInterrupt):
            print();break


if __name__=='__main__':main()
