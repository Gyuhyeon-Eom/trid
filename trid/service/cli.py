"""Subcommands wired into trid.cli: `trid ingest <dir> --index <dir>`, `trid serve --index <dir> --run <model>`,
`trid ask --index <dir> --run <model> --user <u> '<q>'`."""
import json
from pathlib import Path


def add_parsers(sub, ask):
    ingest = sub.add_parser("ingest", help="문서 폴더(txt/md/hwp/hwpx/pdf)를 색인; 권한은 <파일>.acl.json 또는 acl.json")
    ingest.add_argument("source")
    ingest.add_argument("--index", required=True, help="색인을 쓸 폴더")
    ingest.add_argument("--no-redact", action="store_true", help="개인정보 치환(trid.korean_quality.redact) 생략")
    serve = sub.add_parser("serve", help="폐쇄망 JSON API (기본 127.0.0.1:8321): POST /ask, GET /health")
    for p in (serve, ask):
        p.add_argument("--index", required=p is serve, help="trid ingest가 만든 색인 폴더")
        p.add_argument("--users", help="사용자별 권한 태그 JSON (기본 <index>/users.json)")
        p.add_argument("--audit", help="감사 로그 JSONL (기본 <index>/audit.jsonl)")
        p.add_argument("--k", type=int, default=3, help="BM25 상위 문단 수; 참조·상위 조 확장은 별도")
        p.add_argument("--hops", type=int, default=1, help="참조 그래프 확장 홉 수 (0이면 BM25만)")
        p.add_argument("--threads", type=int, default=4)
    serve.add_argument("--run", required=True, help="모델 폴더 (추출 헤드 run이면 추출, 아니면 구간 점수화)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8321)
    token = sub.add_parser("token", help="사용자에게 HTTP 접속 토큰 발급 (users.json에는 sha256만 저장)")
    token.add_argument("--users", required=True)
    token.add_argument("--user", required=True)
    ask.add_argument("--user", help="--index와 함께: 권한 태그를 찾을 사용자 이름")


def token(args):
    from .acl import issue_token
    print(issue_token(args.users, args.user))


def ingest(args):
    from .index import Index
    from .ingest import ingest_dir
    docs = ingest_dir(args.source, redact=not args.no_redact, log=print)
    index = Index.build(docs)
    index.save(args.index)
    print(f"색인 완료: 문서 {len(docs)} · 문단 {len(index.passages)} · 참조 {len(index.edges)} → {args.index}")


def engine_for(args):
    import torch
    from .acl import load_users
    from .audit import AuditLog
    from .engine import Engine, load_backend
    from .index import Index
    if not args.run:
        raise ValueError("--run 모델 폴더를 지정하세요.")
    torch.set_num_threads(args.threads)
    root = Path(args.index)
    index = Index.load(root)
    users = load_users(args.users or root / "users.json")
    audit = AuditLog(args.audit or root / "audit.jsonl")
    return Engine(index, load_backend(args.run, "cpu"), audit, users, k=args.k, hops=args.hops)


def serve(args):
    from .server import serve as run_server
    run_server(engine_for(args), args.host, args.port)


def ask(args):
    import time
    if not args.user:
        raise ValueError("--user 사용자 이름을 지정하세요.")
    engine = engine_for(args)
    started = time.perf_counter()
    answer = engine.ask(args.prompt, args.user)
    seconds = time.perf_counter() - started
    print(f"trid > {answer.text}")
    if answer.source:
        print(f"[출처 {answer.source.doc_id} {' > '.join(answer.source.path[1:])} · 문자 {answer.char_offset}]")
    print(f"[근거 {len(answer.candidates)}문단 · {seconds:.3f}초]")
    print(json.dumps({k: v for k, v in answer.to_dict().items() if k != "source"}, ensure_ascii=False))
