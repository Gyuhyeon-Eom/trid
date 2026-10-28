"""Independent document questions using the exact reading-only training format."""
import argparse
from pathlib import Path
import torch
from trid.prompts import korean_mrc_prompt
from trid.runtime import generate, load_run
from trid.terminal_input import classify_input


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='runs/trid-korean-reading-v1')
    parser.add_argument('--document', required=True, help='UTF-8 텍스트 문서 경로')
    parser.add_argument('--question', help='생략하면 질문 입력창 실행')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'mps'])
    args = parser.parse_args()
    document = Path(args.document).read_text(encoding='utf-8').strip()
    if not document:
        parser.error('빈 문서입니다.')
    torch.set_num_threads(4)
    model, tok, device = load_run(args.run, args.device)
    print(f'trid 문서 독해 실험 | {args.run} | {model.parameter_count():,} parameters | {device}')
    print(f'문서: {Path(args.document).resolve()}')
    print('각 질문은 독립적입니다. 대화 이력·검색 기능 없이 위 문서만 제공합니다. 오답이 가능한 실험 모델입니다.')

    def answer(question):
        result = generate(model, tok, korean_mrc_prompt(document, question), device, max_tokens=64)
        print('trid > ' + result['answer'])
        print(f"[첫 토큰 {result['first_token_seconds']:.3f}초 · 전체 {result['total_seconds']:.3f}초]")

    if args.question is not None:
        answer(args.question)
        return
    print('/exit 종료 · 대화는 파일에 저장하지 않습니다.')
    while True:
        try:
            kind, value = classify_input(input('나 [독해] > '))
            if kind == 'empty':
                continue
            if kind == 'command':
                if value == 'exit':
                    break
                print('질문마다 독립적으로 문서를 읽습니다. /exit 종료')
            elif kind == 'notice':
                print(value)
            else:
                try:
                    answer(value)
                except ValueError as error:
                    print(f'입력 오류: {error}')
        except (EOFError, KeyboardInterrupt):
            print()
            break


if __name__ == '__main__':
    main()
