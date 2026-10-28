"""Author-generated fictional data, not company data or an external corpus."""
import hashlib
import json
import random
from pathlib import Path


def make_demo(destination):
    target = Path(destination)
    if target.exists() and any(target.iterdir()):
        raise ValueError("데이터 폴더가 비어 있지 않습니다. 새 경로를 사용하세요.")
    target.mkdir(parents=True, exist_ok=True)
    rows = []
    names = ["가람", "나래", "다온", "라온", "마루", "바다", "새봄", "아람", "여울", "하람"]
    teams = ["개발팀", "기획팀", "영업팀", "운영팀", "인사팀", "재무팀"]
    for name in names:
        for team in teams:
            for site in ["서울", "부산", "대전", "광주"]:
                rows.append(dict(task="extract", prompt=f"자료: {name}의 소속은 {team}이고 근무지는 {site}입니다.\n질문: {name}의 소속 팀은? 팀 이름만 답하세요.", answer=team))
                rows.append(dict(task="extract", prompt=f"자료: {name}은 {site}에서 근무하며 {team} 소속입니다.\n질문: 근무 도시는? 도시 이름만 답하세요.", answer=site))
    for qty in range(0, 50):
        for threshold in [5, 10, 15, 20, 25, 30]:
            rows.append(dict(task="rule", prompt=f"규칙: 재고가 {threshold}개 미만이면 발주, 그렇지 않으면 유지합니다.\n자료: 현재 재고는 {qty}개입니다.\n질문: 발주 또는 유지 중 하나만 답하세요.", answer="발주" if qty < threshold else "유지"))
    for tag in ["긴급", "일반", "보류"]:
        for ticket in range(1, 81):
            rows.append(dict(task="priority", prompt=f"규칙: 긴급은 우선 처리, 일반은 순차 처리, 보류는 대기입니다.\n자료: 요청 {ticket}번의 등급은 {tag}입니다.\n질문: 처리 방식을 답하세요.", answer={"긴급": "우선 처리", "일반": "순차 처리", "보류": "대기"}[tag]))
    for name in names:
        for team in teams:
            for missing in ["연봉", "입사일", "전화번호", "나이"]:
                rows.append(dict(task="missing", prompt=f"자료: {name}은 {team} 소속입니다.\n질문: {name}의 {missing} 정보는 무엇인가요? 자료에 없으면 모른다고 답하세요.", answer="제공된 정보로는 알 수 없습니다."))
    for question in ["이름이 뭐야?", "너는 누구야?", "자기소개를 해줘.", "모델 이름을 알려줘."]:
        rows.append(dict(task="identity", prompt=question, answer="저는 처음부터 학습 중인 초소형 언어모델 trid입니다."))
    random.Random(42).shuffle(rows)
    n = len(rows)
    splits = {"train": rows[:int(n * .8)], "valid": rows[int(n * .8):int(n * .9)], "test": rows[int(n * .9):]}
    for split, values in splits.items():
        content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in values)
        (target / f"{split}.jsonl").write_text(content, encoding="utf-8")
    info = {"origin": "locally authored synthetic Korean examples", "real_company_data": False,
            "counts": {s: len(v) for s, v in splits.items()},
            "limitation": "Held-out rows share templates and vocabulary with training. Not a general Korean or enterprise reasoning benchmark.",
            "sha256": {s: hashlib.sha256((target / f"{s}.jsonl").read_bytes()).hexdigest() for s in splits}}
    (target / "provenance.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(info, indent=2, ensure_ascii=False))
