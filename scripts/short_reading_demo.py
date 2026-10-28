"""Independent span and presence models, local short-document experiment."""
import argparse
from pathlib import Path
import torch
from trid.short_reader import load_pipeline,read
from trid.terminal_input import classify_input


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',default='runs/trid-korean-short-v1')
    p.add_argument('--document',required=True);p.add_argument('--question');p.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    a=p.parse_args();document=Path(a.document).read_text(encoding='utf-8')
    if not document.strip():p.error('빈 문서입니다.')
    torch.set_num_threads(4);span,presence,tok,device=load_pipeline(a.run,a.device)
    print(f'trid 짧은 문서 실험 | 분리된 구간·존재 여부 모델 | {device} | {Path(a.run).name}')
    print(f'문서: {Path(a.document).resolve()}')
    print('출장 규정 중심의 가상 문서로 학습한 실험입니다. 질문마다 독립적으로 읽습니다. /exit 종료')
    def answer(q):
        r=read(span,presence,tok,document,q,device);print('trid > '+r['answer'])
        if r['source_character_span']:
            start,end=r['source_character_span'];print(f'원문 위치 [{start}, {end}) · {r["total_seconds"]:.3f}초')
            print('근거 주변: '+document[max(0,start-25):start]+'【'+document[start:end]+'】'+document[end:end+25])
        else:print(f'자료 없음으로 판단 · {r["total_seconds"]:.3f}초')
    if a.question is not None:answer(a.question);return
    while True:
        try:
            kind,value=classify_input(input('나 [짧은 문서] > '))
            if kind=='empty':continue
            if kind=='command':
                if value=='exit':break
                print('이전 대화는 사용하지 않습니다. /exit 종료')
            elif kind=='notice':print(value)
            else:
                try:answer(value)
                except ValueError as e:print(f'입력 오류: {e}')
        except (EOFError,KeyboardInterrupt):print();break


if __name__=='__main__':main()
