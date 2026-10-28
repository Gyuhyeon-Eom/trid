from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

from .runtime import load_run, generate
from .text import read_rows


def evaluate(run, data, output, device="auto", limit=0):
    metadata_file = Path(run) / "training.json"
    if metadata_file.exists() and json.loads(metadata_file.read_text()).get("stage") == "pretrain":
        raise ValueError("사전학습 모델은 문서 검증 손실로 평가하세요. 대화 학습 전 정답 비교는 지원하지 않습니다.")
    rows = read_rows(data)
    if limit < 0:
        raise ValueError("limit은 0 이상이어야 합니다.")
    if limit:
        rows = rows[:limit]
    model, tok, device = load_run(run, device)
    # Warm-up is excluded from the per-request latency statistics.
    generate(model, tok, "이름이 뭐야?", device, max_tokens=4)
    results, groups = [], defaultdict(lambda: {"correct": 0, "total": 0})
    for index, row in enumerate(rows, 1):
        result = generate(model, tok, row["prompt"], device, max_tokens=48, history=row.get("history"))
        correct = result["answer"] == row["answer"].strip()
        task = row.get("task", "unspecified")
        groups[task]["total"] += 1
        groups[task]["correct"] += int(correct)
        results.append({"prompt": row["prompt"], "expected": row["answer"], "task": task, "exact_match": correct, **result})
        if index % 25 == 0:
            print(f"평가 {index}/{len(rows)}", flush=True)
    report = {"evaluated_at": datetime.now(timezone.utc).isoformat(), "device": device,
              "parameters": model.parameter_count(), "examples": len(results),
              "exact_match": sum(r["exact_match"] for r in results) / len(results),
              "median_first_token_seconds": statistics.median(r["first_token_seconds"] for r in results),
              "median_total_seconds": statistics.median(r["total_seconds"] for r in results),
              "by_task": dict(groups),
              "test_sha256": hashlib.sha256(Path(data).read_bytes()).hexdigest(),
              "weights_sha256": hashlib.sha256((Path(run) / "model.safetensors").read_bytes()).hexdigest(),
              "limitation": "Synthetic same-template held-out test; not evidence of general language skill, security or real company performance. M5 timings do not predict Raspberry Pi timings.",
              "results": results}
    destination = Path(output)
    if destination.exists():
        raise ValueError("기존 평가 보고서를 덮어쓰지 않습니다. 새 출력 경로를 사용하세요.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, ensure_ascii=False, indent=2))
