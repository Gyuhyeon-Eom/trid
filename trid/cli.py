import argparse
import os
from pathlib import Path
import json
import sys
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def default_chat_run():
    selection = PROJECT_ROOT / "active-model.json"
    relative = json.loads(selection.read_text())["run"] if selection.exists() else "runs/trid-v0.1"
    path = (PROJECT_ROOT / relative).resolve()
    if not path.is_relative_to((PROJECT_ROOT / "runs").resolve()):
        raise ValueError("활성 모델은 프로젝트 runs 폴더 안에 있어야 합니다.")
    return str(path)

# Inference/training stay local. Only explicit fetch-wiki downloads public text.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    parser = argparse.ArgumentParser(description="trid — 처음부터 학습하는 초소형 언어모델")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo-data", help="가상 한국어 업무 데이터 생성")
    demo.add_argument("--output", default="data/demo")
    train = sub.add_parser("train", help="질문·답변 학습; --init-run으로 자체 사전학습 결과 계승")
    train.add_argument("--data", default="data/demo")
    train.add_argument("--output", default="runs/trid-v0.1")
    train.add_argument("--steps", type=int, default=300)
    train.add_argument("--batch-size", type=int, default=24)
    train.add_argument("--lr", type=float, default=.0005)
    train.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--init-run", help="직접 사전학습한 로컬 체크포인트에서 대화 학습")
    train.add_argument("--eval-every", type=int, default=100)
    train.add_argument("--validation-limit", type=int, default=256, help="0이면 검증 split 전체 사용")
    train.add_argument("--intent-weight", type=float, default=0.0, help="질문 유형 구분 보조 손실의 가중치; 추론에는 사용하지 않음")
    train.add_argument("--context-length", type=int, help="별도 학습 실행의 문맥 길이; 기존 가중치와 토크나이저 유지")
    train.add_argument("--loss-normalization", choices=["token", "example"], default="token", help="example: 답변 길이에 관계없이 질문별 평균 손실을 동일 비중으로 학습")
    train.add_argument("--micro-batch-size", type=int, default=4, help="한 배치를 나누어 역전파하는 크기; 메모리만 제한하고 학습 결과는 동일 (batch-size 이하로 제한)")
    fetch = sub.add_parser("fetch-wiki", help="공개 한국어 Wikipedia 문서 다운로드 (명시적 네트워크 작업)")
    fetch.add_argument("--output", default="data/wiki-ko-pilot")
    fetch.add_argument("--max-documents", type=int, default=12000)
    prepare = sub.add_parser("prepare-corpus", help="문서 분리 검증·새 토크나이저·토큰 데이터 생성")
    prepare.add_argument("--source", default="data/wiki-ko-pilot")
    prepare.add_argument("--output", default="data/wiki-ko-tokens")
    prepare.add_argument("--vocab-size", type=int, default=8192)
    prepare.add_argument("--tokenizer", help="기존 자체 토크나이저를 변경 없이 재사용")
    pretrain = sub.add_parser("pretrain", help="일반 텍스트로 사전학습; 실행마다 지정한 단계만 수행")
    pretrain.add_argument("--data", default="data/wiki-ko-tokens")
    pretrain.add_argument("--output", default="runs/trid-pretrain-pilot")
    pretrain.add_argument("--config", help="무작위 초기화 시 기본값: configs/pretrain-small.json")
    pretrain.add_argument("--steps", type=int, default=2000, help="새 실행의 전체 스케줄; 재개 시 저장값 사용")
    pretrain.add_argument("--stop-after", type=int, default=200, help="이번 실행에서 수행할 최대 추가 단계")
    pretrain.add_argument("--batch-size", type=int, default=8)
    pretrain.add_argument("--lr", type=float, default=.0003)
    pretrain.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    pretrain.add_argument("--seed", type=int, default=42)
    start = pretrain.add_mutually_exclusive_group()
    start.add_argument("--resume", action="store_true")
    start.add_argument("--init-run", help="자체 사전학습 최적 가중치에서 새 학습 일정 시작")
    pretrain.add_argument("--eval-every", type=int, default=50)
    pretrain.add_argument("--validation-batches", type=int, default=8)
    evaluate = sub.add_parser("evaluate", help="로컬 평가; 지정한 보고서에 질문·정답·출력을 저장")
    evaluate.add_argument("--run", default="runs/trid-v0.1")
    evaluate.add_argument("--data", default="data/demo/test.jsonl")
    evaluate.add_argument("--output", default="reports/evaluation.json")
    evaluate.add_argument("--limit", type=int, default=0)
    evaluate.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    for name in ["ask", "chat", "complete"]:
        p = sub.add_parser(name, help={"ask": "한 번 질문; --index를 주면 색인 문서를 근거로 답하고 감사 로그를 남김", "chat": "터미널 대화", "complete": "사전학습 모델의 문장 이어쓰기"}[name])
        p.add_argument("--run", default=str(PROJECT_ROOT / "runs/trid-pretrain-pilot") if name == "complete" else None)
        p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
        p.add_argument("--max-tokens", type=int, default=48 if name == 'complete' else 96)
        if name == 'chat':
            p.add_argument('--mode',choices=['select','dialogue','institution'],default='dialogue',help='기본은 파일 없는 대화; select는 모드 선택, institution은 가상 문서 실험')
            p.add_argument('--source',nargs='+',default=['T01'],help='기관 모드에서 제공할 가상 문서 ID')
        if name != "chat":
            p.add_argument("prompt")
        if name == "ask":
            from .service.cli import add_parsers as add_service_parsers
            add_service_parsers(sub, p)
    args = parser.parse_args()
    # Private-by-default permissions for new files made by this process.
    os.umask(0o077)
    try:
        if args.command == "demo-data":
            from .demo import make_demo
            make_demo(args.output)
        elif args.command == "fetch-wiki":
            from .corpus import fetch_wiki
            fetch_wiki(args.output, args.max_documents)
        elif args.command == "prepare-corpus":
            from .corpus import prepare
            prepare(args.source, args.output, args.vocab_size, args.tokenizer)
        elif args.command == "pretrain":
            from .pretraining import pretrain as run_pretrain
            config = args.config or (None if args.resume or args.init_run else "configs/pretrain-small.json")
            run_pretrain(args.data, args.output, config, args.steps, args.stop_after,
                         args.batch_size, args.lr, args.device, args.seed, args.resume,
                         args.eval_every, args.validation_batches, args.init_run)
        elif args.command == "train":
            from .training import train as train_model
            train_model(args.data, args.output, args.steps, args.batch_size, args.lr, args.device, args.seed, args.init_run,
                        args.eval_every, args.validation_limit, args.intent_weight, args.context_length, args.loss_normalization,
                        args.micro_batch_size)
        elif args.command == "evaluate":
            from .evaluation import evaluate as run_evaluation
            run_evaluation(args.run, args.data, args.output, args.device, args.limit)
        elif args.command in ("ingest", "serve", "token") or (args.command == "ask" and args.index):
            from .service import cli as service_cli
            getattr(service_cli, args.command)(args)
        else:
            if args.command == 'chat':
                mode=args.mode
                if mode=='select':
                    if sys.stdin.isatty() and args.run is None:
                        print('1. 기관 규정 테스트 — 선택한 가상 문서를 읽기')
                        print('2. 일반 대화 실험 — 기관 문서 없음, 정확도 낮음')
                        while True:
                            selection=input('실행 모드 [1/2, 기본 1] > ').strip()
                            if selection in ('','1','2'):
                                mode='dialogue' if selection=='2' else 'institution'
                                break
                            print('1 또는 2를 입력하세요.')
                    else:
                        mode='dialogue'
                if mode=='institution':
                    command=[sys.executable,str(PROJECT_ROOT/'scripts/institution_demo.py'),'--chat','--device',args.device,'--source',*args.source]
                    if args.run:command.extend(['--run',args.run])
                    result=subprocess.run(command)
                    parser.exit(result.returncode)
            from .runtime import load_run, generate
            import torch
            if args.run is None:
                args.run = default_chat_run()
            metadata_path = Path(args.run) / "training.json"
            metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
            if metadata.get("stage") == "pretrain" and args.command != "complete":
                raise ValueError("이 모델은 아직 대화 학습 전입니다. complete 명령으로 문장 이어쓰기를 확인하세요.")
            torch.set_num_threads(4)
            model, tok, device = load_run(args.run, args.device)
            print(f"trid | {model.parameter_count():,} parameters | {device} | {Path(args.run).name}")
            print("초기 사전학습 모델의 문장 이어쓰기입니다. 대화 모델이 아닙니다." if args.command == "complete" else "초기 실험 모델입니다. 일반 대화·기업 실무 성능은 검증되지 않았습니다.")
            if args.command == "chat":
                print("파일 없는 기본 대화 모드입니다. 기관별 최신 규정·금액은 확인이 필요합니다.")
                print("짧은 이전 대화를 참고합니다. /reset 대화 초기화 · /exit 종료 · 대화는 저장하지 않습니다.")
            history = []
            while True:
                question = args.prompt if args.command != "chat" else input("\n나 > ").strip()
                if args.command == 'chat':
                    from .terminal_input import classify_input
                    kind, value = classify_input(question)
                    if kind == 'empty':
                        continue
                    if kind == 'notice':
                        print(value)
                        continue
                    if kind == 'command':
                        question = '/' + value
                        if value == 'help':
                            print('/exit 종료 · /reset 초기화 · /example 예제. 터미널 명령은 채팅 종료 후 실행하세요.')
                            continue
                if args.command == 'chat' and question == "/exit":
                    break
                if args.command == 'chat' and question == "/reset":
                    history.clear()
                    print("대화 이력을 초기화했습니다.")
                    continue
                if args.command == 'chat' and question == "/example":
                    question = "안녕"
                    print(question)
                if not question:
                    if args.command != 'chat':
                        raise ValueError('질문을 입력해 주세요.')
                    continue
                try:
                    result = generate(model, tok, question, device, args.max_tokens, raw_text=args.command == "complete", history=history)
                    print(f"trid > {result['answer']}")
                    print(f"[첫 토큰 {result['first_token_seconds']:.3f}초 · 전체 {result['total_seconds']:.3f}초]")
                    if result["history_turns_dropped"]:
                        history = history[result["history_turns_dropped"]:]
                        print("[문맥 한도로 오래된 대화를 제외했습니다.]")
                    if args.command == "chat" and result["answer"]:
                        history.append({"user": question, "assistant": result["answer"]})
                except ValueError as e:
                    print(f"입력 오류: {e}")
                    if args.command != "chat":
                        raise
                if args.command != "chat":
                    break
    except (ValueError, FileNotFoundError) as e:
        parser.exit(2, f"오류: {e}\n")
    except (KeyboardInterrupt, EOFError):
        print("\n종료했습니다.")


if __name__ == "__main__":
    main()
