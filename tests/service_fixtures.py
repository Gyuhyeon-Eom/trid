"""Synthetic documents shared by tests/test_service_*.py."""
import json
from pathlib import Path

REG = """출장여비규정
제정 2019. 3. 1.

제1장 총칙

제1조(목적) 이 규정은 한국공공데이터원 임직원의 출장여비 지급 기준을 정함을 목적으로 한다.
제2조(정의) 이 규정에서 사용하는 용어의 뜻은 다음과 같다.
1. "국내출장"이란 근무지 외의 국내 지역으로 가는 출장을 말한다.
2. "숙박비"란 출장지에서 숙박에 드는 비용을 말한다.

제2장 국내출장

제12조(국내출장여비) ① 국내출장 여비는 운임, 일비, 숙박비, 식비로 구분하여 지급한다.
② 숙박비의 상한은 1박당 8만 원으로 한다. 다만, 제13조에 따른 특별시 출장은 별표 2의 상한을 적용한다.
③ 일비는 1일 2만 5천 원으로 한다.
제13조(특별시 출장) 서울특별시로 가는 출장은 제12조제2항의 상한을 1박당 10만 원으로 한다.
제14조(정산) 출장자는 귀임 후 7일 이내에 별지 제1호서식에 따라 정산서를 제출하여야 한다. 「공공기관 회계규칙」 제5조를 준용한다.

부칙
이 규정은 2019년 3월 1일부터 시행한다.

[별표 2] 특별시 숙박비 상한
구분 상한액
특별시 1박당 10만 원
광역시 1박당 9만 원

[별지 제1호서식] 출장 정산서
성명 부서 출장기간 금액
"""

HR = """급여규정
제1조(목적) 이 규정은 직원의 급여 지급 기준을 정한다.
제2조(지급일) 급여는 매월 25일에 지급한다. 담당자 연락처 010-1234-5678.
제3조(출장) 출장 시 여비는 「출장여비규정」 제12조에 따른다.
"""

USERS = {"kim": ["staff"], "hr_lee": ["staff", "hr"], "notags": []}


def make_corpus(root):
    """<root>/docs with 출장여비규정 (staff via acl.json default) and 급여규정 (hr via sidecar); users.json."""
    root = Path(root)
    docs = root / "docs"
    docs.mkdir(parents=True)
    (docs / "출장여비규정.txt").write_text(REG, encoding="utf-8")
    (docs / "급여규정.txt").write_text(HR, encoding="utf-8")
    (docs / "급여규정.txt.acl.json").write_text(json.dumps({"acl": ["hr"]}), encoding="utf-8")
    (docs / "acl.json").write_text(json.dumps({"default": ["staff"]}), encoding="utf-8")
    (root / "users.json").write_text(json.dumps(USERS), encoding="utf-8")
    return docs
