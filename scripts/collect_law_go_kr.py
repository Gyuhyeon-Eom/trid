"""Collect Korean statutes, rules and rulings from the law.go.kr DRF Open API.

Targets: current laws (law), current admin rules (admrul, nw=1), current local ordinances
(ordin, nw=1), court rulings (prec), Constitutional Court decisions (detc), legislative
interpretations (expc) and administrative tribunal decisions (decc).

Polite by construction: one connection, at least 0.1 s between requests, honest User-Agent,
robots.txt checked before the first request. Every list page and body response is cached under
raw/ and never fetched again, so a re-run resumes where it stopped and a fully cached re-run
makes no requests. docs.jsonl, provenance.json and pilot-stats.json are rebuilt from the cache.

    .venv/bin/python scripts/collect_law_go_kr.py --output data/law-ko-source --limit 100
    .venv/bin/python scripts/collect_law_go_kr.py --output data/law-ko-source --status

Text is copyright-free under 저작권법 제7조; cite 법제처 국가법령정보센터 as the source.
"""
import argparse
from collections import Counter
import hashlib
import json
import http.client
import math
import os
from pathlib import Path
import random
import re
import statistics
import sys
import time
import threading
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trid.corpus import sha256 as file_sha256, write_json, prefetch

HOST = "http://www.law.go.kr"
API = HOST + "/DRF/"
USER_AGENT = ("trid-corpus-collector/0.1 (Korean LM pretraining research project 'trid'; "
              "law.go.kr DRF Open API; <= 10 requests/s, single connection)")
INTERVAL = 0.1
RETRY_WAITS = (10, 30, 60, 120, 300, 300, 300, 300)  # ~25 min of outage before a request is given up
DISPLAY = 100
LICENSE = "KR-Copyright-Act-Art7 (law.go.kr)"
ATTRIBUTION = "출처: 법제처 국가법령정보센터"
MIN_CHARS = 200
TOKENIZER = ROOT / "runs/trid-korean-medium-base-v1/tokenizer.json"
NEAR_SAMPLE_CAP = 5000  # docs per target kept in memory for near-duplicate estimates
ORDIN_WITHIN_CAP = 1000  # ordinances in the quadratic cross-municipality check

# target -> list params, item id (body key), body parameter, stable group id, title, date field
TARGETS = {
    "law": dict(params={}, id="법령일련번호", body="MST", group="법령ID", title="법령명한글",
                date="공포일자", link="법령상세링크", name="현행 법령"),
    "admrul": dict(params={"nw": "1"}, id="행정규칙일련번호", body="ID", group="행정규칙ID",
                   title="행정규칙명", date="발령일자", link="행정규칙상세링크", name="현행 행정규칙"),
    "ordin": dict(params={"nw": "1"}, id="자치법규ID", body="ID", group="자치법규ID", title="자치법규명",
                  date="공포일자", link="자치법규상세링크", name="현행 자치법규"),
    "prec": dict(params={}, id="판례일련번호", body="ID", group="판례일련번호", title="사건명",
                 date="선고일자", link="판례상세링크", name="판례"),
    "detc": dict(params={}, id="헌재결정례일련번호", body="ID", group="헌재결정례일련번호", title="사건명",
                 date="종국일자", link="헌재결정례상세링크", name="헌재결정례"),
    "expc": dict(params={}, id="법령해석례일련번호", body="ID", group="법령해석례일련번호", title="안건명",
                 date="회신일자", link="법령해석례상세링크", name="법령해석례"),
    "decc": dict(params={}, id="행정심판재결례일련번호", body="ID", group="행정심판재결례일련번호",
                 title="사건명", date="의결일자", link="행정심판례상세링크", name="행정심판재결례"),
}
STATUTES = {"law", "admrul", "ordin"}
# Ruling fields in reading order; label None means the field already carries its own headings.
RULING_FIELDS = {
    "prec": [("판시사항", "판시사항"), ("판결요지", "판결요지"), ("판례내용", None)],
    "detc": [("판시사항", "판시사항"), ("결정요지", "결정요지"), ("전문", None)],
    "expc": [("질의요지", "질의요지"), ("회답", "회답"), ("이유", "이유")],
    "decc": [("재결요지", None), ("주문", "주문"), ("청구취지", "청구취지"), ("이유", "이유")],
}
UNIT_TEXT = {"조문내용", "항내용", "호내용", "목내용", "조내용"}

BLOCK_TAG = re.compile(r"<\s*(?:br|/p|/div|/tr|/li)\b[^<>]*>", re.I)
TAG = re.compile(r"<[a-zA-Z/!|][^<>]{0,300}>")
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
BOX = re.compile(r"[\u2500-\u257f]+")
SPACES = re.compile(r"[ \t\u3000\xa0\u200b]+")
AMEND = re.compile(r"<\s*(?:개정|신설|전문개정|제목개정|본조신설|타법개정|일부개정)[^<>]*>")
NOTE = re.compile(r"\[(?:본조신설|전문개정|제목개정|본조개정|종전|시행일)[^\[\]]*\]"
                  r"|\[[^\[\]]{0,80}(?:에서|으로|로)\s*이동[^\[\]]{0,40}\]")
DELETED = re.compile(r"^(?:제\d+(?:조|장|절|관|편)(?:의\d+)?(?:\([^)]*\))?\s*|[①-⑳㉑-㉟]\s*"
                     r"|\d+(?:의\d+)?\.\s*|[가-힣](?:의\d+)?\.\s*)?삭\s*제\s*(?:<[^<>]*>|\[[^\[\]]*\])?\s*$")
CONTACT = re.compile(r"^(?:☎.*|(?:전화(?:번호)?|연락처|담당부서|담당자|문의처?|팩스|FAX|E-?mail|이메일)\s*[:：].*)$",
                     re.I)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE = re.compile(r"(?<![\d.-])(?:\(0\d{1,2}\)\s?|0\d{1,2}[-)]\s?)\d{3,4}-\d{4}(?![\d-])"
                   r"|(?<![\d-])(?:15|16|18)\d{2}-\d{4}(?![\d-]|\s*년)")


class AuthError(RuntimeError):
    pass


class FetchError(RuntimeError):
    pass


def clean(text):
    """Markup, stub, metadata and PII removal on one text; NFC, whitespace-normalized."""
    text = unicodedata.normalize("NFC", text or "")
    text = BLOCK_TAG.sub("\n", text)
    text = TAG.sub("", text)
    text = CONTROL.sub("", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = BOX.sub(" ", text)
    text = AMEND.sub("", text)
    text = NOTE.sub("", text)
    text = EMAIL.sub("", text)
    text = PHONE.sub("", text)
    lines, blank = [], False
    for line in text.split("\n"):
        line = SPACES.sub(" ", line).strip()
        if not line:
            blank = bool(lines)
            continue
        if DELETED.match(line) or CONTACT.match(line):
            continue
        if blank:
            lines.append("")
        lines.append(line)
        blank = False
    return "\n".join(lines).strip()


def norm_date(value):
    digits = re.sub(r"\D", "", value or "")
    if len(digits) != 8:
        return None
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:]}"


def latest_buchik(root):
    """부칙 text of the latest promulgation (ties: last in document order)."""
    best = None
    for index, parent in enumerate(root.iter()):
        bodies = [e.text for e in parent.findall("부칙내용") if e.text and e.text.strip()]
        if bodies:
            key = (re.sub(r"\D", "", parent.findtext("부칙공포일자") or ""), index)
            if best is None or key > best[0]:
                best = (key, "\n".join(bodies))
    return best[1] if best else ""


def statute_text(root, title):
    """Name, 조/항/호/목 in document order, latest 부칙, 별표 text (서식 forms excluded)."""
    parts = [title]
    parts += [e.text for e in root.iter() if e.tag in UNIT_TEXT and e.text and e.text.strip()]
    buchik = latest_buchik(root)
    if buchik:
        parts += ["", buchik]
    for table in root.iter("별표단위"):
        if (table.findtext("별표구분") or "").strip() == "서식":
            continue
        content = table.findtext("별표내용") or ""
        if content.strip():
            parts += ["", content]
    return clean("\n".join(parts))


def ruling_text(root, target, title):
    """Returns (full text with title and section labels, cleaned body length without them)."""
    parts, plain = [title], []
    for field, label in RULING_FIELDS[target]:
        value = clean(root.findtext(field) or "")
        if not value:
            continue
        plain.append(value)
        parts += ([f"【{label}】"] if label else []) + [value]
    return clean("\n".join(parts)), len("\n".join(plain))


def extract(target, body, item):
    """Return (title, text, None) or (None, None, drop_reason)."""
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return None, None, "parse_error"
    if len(root) == 0:  # e.g. <Law>일치하는 판례가 없습니다.</Law>: no JSON/XML body for this item
        return None, None, "no_body"
    spec = TARGETS[target]
    title = clean(item.get(spec["title"]) or "")
    if target == "law":
        title = clean(root.findtext("기본정보/법령명_한글") or title)
    elif target == "admrul":
        title = clean(root.findtext("행정규칙기본정보/행정규칙명") or title)
    elif target == "ordin":
        title = clean(root.findtext("자치법규기본정보/자치법규명") or title)
    else:
        title = clean(root.findtext(spec["title"]) or title)
    if target in STATUTES:
        text = statute_text(root, title)
        if len(text) < MIN_CHARS:
            return None, None, "short"
    else:
        text, body_chars = ruling_text(root, target, title)
        if body_chars < MIN_CHARS:
            return None, None, "stub_ruling"
    return title, text, None


class Client:
    """Shared by worker threads: one global request interval and counter."""
    def __init__(self, oc):
        self.oc = oc
        self.last = 0.0
        self.requests = 0
        self.robots = None
        self.lock = threading.Lock()

    def _wait(self):
        with self.lock:
            delay = self.last + INTERVAL - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self.last = time.monotonic()
            self.requests += 1

    def _open(self, url):
        self._wait()
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=90) as response:
            return response.read()

    def check_robots(self):
        if self.robots is None:
            parser = urllib.robotparser.RobotFileParser()
            parser.parse(self._open(HOST + "/robots.txt").decode("utf-8", "replace").splitlines())
            self.robots = parser
        for endpoint in ("lawSearch.do", "lawService.do"):
            if not self.robots.can_fetch(USER_AGENT, API + endpoint):
                raise FetchError(f"robots.txt disallows {API}{endpoint}")

    def get(self, endpoint, params):
        self.check_robots()
        query = urllib.parse.urlencode({"OC": self.oc, **params, "type": "XML"})
        url = f"{API}{endpoint}?{query}"
        error = None
        for attempt in range(len(RETRY_WAITS) + 1):
            if attempt:
                time.sleep(RETRY_WAITS[attempt - 1])
            try:
                body = self._open(url)
            except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as e:
                error = e
                continue
            if b"<Response>" in body[:300] and "사용자 정보 검증".encode() in body:
                raise AuthError(body.decode("utf-8", "replace").strip())
            try:
                ET.fromstring(body)
            except ET.ParseError as e:
                error = e
                continue
            return body
        raise FetchError(f"{endpoint} {params}: {error}")


def cached(path, fetch):
    """Read path, or fetch, store atomically and return. Only valid XML reaches the cache."""
    if path.exists():
        return path.read_bytes(), False
    body = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_bytes(body)
    os.replace(part, path)
    return body, True


def list_items(body, spec):
    root = ET.fromstring(body)
    total = int(root.findtext("totalCnt") or 0)
    items = [{c.tag: (c.text or "").strip() for c in e} for e in root if e.find(spec["id"]) is not None]
    return total, items


def list_page(client, raw, target, page, extra=None, display=DISPLAY):
    spec = TARGETS[target]
    params = {"target": target, **spec["params"], "display": str(display), "page": str(page), **(extra or {})}
    name = f"page-{page:05d}.xml" if display == DISPLAY else f"count-display{display}.xml"
    body, _ = cached(raw / target / "lists" / name, lambda: client.get("lawSearch.do", params))
    return list_items(body, spec)


def select(client, raw, target, limit, seed):
    """Seeded pilot sample spread over random list pages, or every item when limit is None."""
    total, _ = list_page(client, raw, target, 1, display=1)
    pages = max(1, math.ceil(total / DISPLAY))
    if limit is None:
        for page in range(1, pages + 1):
            yield from list_page(client, raw, target, page)[1]
        return
    rng = random.Random(f"{seed}:{target}")
    chosen = sorted(rng.sample(range(1, pages + 1), min(pages, max(1, math.ceil(limit / 10)))))
    per = math.ceil(limit / len(chosen))
    picks, spare = [], []
    for page in chosen:
        items = list_page(client, raw, target, page)[1]
        rng.shuffle(items)
        picks += items[:per]
        spare += items[per:]
    seen, out = set(), []
    for item in picks + spare:
        key = item[TARGETS[target]["id"]]
        if key not in seen and len(out) < limit:
            seen.add(key)
            out.append(item)
    yield from out


def source_url(item, spec):
    """List 상세링크 with the OC key removed; append &OC=<key> to open it."""
    link = item.get(spec["link"]) or ""
    parts = urllib.parse.urlsplit(link)
    query = [(k, v) for k, v in urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
             if k not in ("OC", "mobileYn")]
    return HOST + parts.path + "?" + urllib.parse.urlencode(query)


def item_path(raw, target, item_id):
    return raw / target / f"{re.sub(r'[^0-9A-Za-z_-]', '_', item_id)}.xml"


def word_shingles(text):
    words = re.findall(r"\w+", unicodedata.normalize("NFC", text).lower())
    return {" ".join(words[i:i + 5]) for i in range(max(1, len(words) - 4))}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def near_duplicate_estimate(docs):
    """Kept docs that near-duplicate an earlier one (MinHash candidates, exact Jaccard >= 0.8)."""
    from scripts.prepare_korean_continuation import NearDuplicates
    index, hits = NearDuplicates(), 0
    sample = docs[:NEAR_SAMPLE_CAP]
    for doc in sample:
        if index.match_and_add(doc["id"], doc["text"]):
            hits += 1
    return {"checked": len(sample), "near_duplicates": hits,
            "rate": round(hits / len(sample), 4) if sample else None}


def municipality_tokens(name):
    tokens = set()
    for word in (name or "").split():
        tokens.add(word)
        if len(word) > 2 and word[-1] in "시군구도":
            tokens.add(word[:-1])
        for suffix in ("특별자치시", "특별자치도", "광역시", "특별시"):
            if word.endswith(suffix) and len(word) > len(suffix):
                tokens.add(word[:-len(suffix)])
    return tokens


def template_title(title, municipality):
    words = [w for w in title.split() if w not in municipality_tokens(municipality)]
    return " ".join(words).strip()


def ordinance_report(client, raw, docs, issuers, probe, seed):
    """Cross-municipality near-duplicates: inside the pilot, and by same-title probe searches."""
    shingle_sets = {d["id"]: word_shingles(d["text"]) for d in docs}
    within = 0
    for d in docs:
        if any(o["id"] != d["id"] and issuers[o["id"]] != issuers[d["id"]]
               and jaccard(shingle_sets[d["id"]], shingle_sets[o["id"]]) >= 0.8 for o in docs):
            within += 1
    report = {"pilot_docs": len(docs), "pilot_near_duplicate_of_other_municipality": within,
              "pilot_rate": round(within / len(docs), 4) if docs else None,
              "note": "A random pilot rarely contains two municipalities' versions of one ordinance, so the "
                      "probe searches each sampled title (municipality name removed) and compares the "
                      "other municipalities' current versions."}
    rng = random.Random(f"{seed}:ordin-probe")
    order = sorted(docs, key=lambda d: d["id"])
    rng.shuffle(order)
    rows = []
    for d in order[:probe]:
        query = template_title(d["title"], issuers[d["id"]])
        row = {"id": d["id"], "municipality": issuers[d["id"]], "query": query, "other_municipality_hits": 0,
               "compared": 0, "max_jaccard": None}
        if len(query) >= 4:
            key = hashlib.sha256(query.encode()).hexdigest()[:16]
            params = {"target": "ordin", "nw": "1", "query": query, "display": str(DISPLAY), "page": "1"}
            try:
                body, _ = cached(raw / "ordin-probe" / f"search-{key}.xml",
                                 lambda: client.get("lawSearch.do", params))
            except FetchError:
                row["error"] = "fetch_error"
                rows.append(row)
                continue
            own = d["id"].rsplit(":", 1)[1]
            _, hits = list_items(body, TARGETS["ordin"])
            hits = [h for h in hits if h["자치법규ID"] != own and h.get("지자체기관명") != issuers[d["id"]]
                    and template_title(h.get("자치법규명", ""), h.get("지자체기관명", "")) == query]
            row["other_municipality_hits"] = len(hits)
            best = 0.0
            for h in rng.sample(hits, min(2, len(hits))):
                try:
                    other, _ = cached(item_path(raw / "ordin-probe", "", h["자치법규ID"]),
                                      lambda: client.get("lawService.do", {"target": "ordin", "ID": h["자치법규ID"]}))
                except FetchError:
                    continue
                _, text, reason = extract("ordin", other, h)
                if reason:
                    continue
                row["compared"] += 1
                best = max(best, jaccard(shingle_sets[d["id"]], word_shingles(text)))
            row["max_jaccard"] = round(best, 4) if row["compared"] else None
        rows.append(row)
    compared = [r for r in rows if r["compared"]]
    report["probe"] = {
        "probed": len(rows),
        "same_title_in_other_municipality": sum(1 for r in rows if r["other_municipality_hits"]),
        "compared": len(compared),
        "near_duplicate_of_other_municipality": sum(1 for r in compared if r["max_jaccard"] >= 0.8),
        "rate_among_probed": round(sum(1 for r in compared if r["max_jaccard"] >= 0.8) / len(rows), 4) if rows else None,
        "median_max_jaccard": round(statistics.median(r["max_jaccard"] for r in compared), 4) if compared else None,
        "rows": rows,
    }
    return report


def load_json(path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def provenance(out, oc, args, totals, counts, ledger):
    return {
        "source": "법제처 국가법령정보센터 Open API (DRF)",
        "endpoints": {"list": API + "lawSearch.do", "body": API + "lawService.do"},
        "response_type": "XML (the JSON responses drop line breaks inside articles and rulings)",
        "targets": {t: {"name": s["name"], "list_params": s["params"], "body_key": f"{s['body']}={s['id']}",
                        "group": f"{t}:{s['group']}", "total_listed": totals.get(t)} for t, s in TARGETS.items()},
        "oc": "public demo key 'test'" if oc == "test" else "own key (value not recorded)",
        "user_agent": USER_AGENT,
        "rate_limit": f"single connection, >= {INTERVAL} s between requests; retries after {list(RETRY_WAITS)} s",
        "robots_txt": f"{HOST}/robots.txt: 'User-agent:* Allow: /' (checked before the first request of each run)",
        "selection": (f"pilot: {args.limit} items per target, seeded ({args.seed}) random list pages "
                      f"(display={DISPLAY}), ~10 items per page" if args.limit else "full: every list page"),
        "license": LICENSE,
        "license_basis": {
            "저작권법 제7조": "다음 각 호의 어느 하나에 해당하는 것은 이 법에 의한 보호를 받지 못한다. "
                         "1. 헌법ㆍ법률ㆍ조약ㆍ명령ㆍ조례 및 규칙 2. 국가 또는 지방자치단체의 고시ㆍ공고ㆍ훈령 그 밖에 "
                         "이와 유사한 것 3. 법원의 판결ㆍ결정ㆍ명령 및 심판이나 행정심판절차 그 밖에 이와 유사한 절차에 "
                         "의한 의결ㆍ결정 등",
            "statute_url": "https://www.law.go.kr/lsLinkCommonInfo.do?lsJoLnkSeq=1029423769",
            "open_api_policy": "제공된 법령정보는 영리 목적의 이용을 포함하여 자유로운 활용이 보장됩니다",
            "open_api_policy_url": "https://open.law.go.kr/LSO/information/guide.do",
            "conditions": "출처를 법제처(국가법령정보센터)로 표시하고, 가공한 텍스트를 공식 원문인 것처럼 제시하지 않는다 "
                          "(저작권법 제137·138조 경고).",
        },
        "attribution": ATTRIBUTION,
        "changes": "XML text fields joined in document order; latest 부칙 only; 별표 text without 서식 forms; markup "
                   "(<br/> etc.), amendment notes (<개정 …>, [본조신설 …]), '삭제 <…>' stubs, contact lines, phone "
                   "numbers and e-mail addresses removed; whitespace normalized; NFC; docs < 200 chars dropped "
                   "(rulings: body without title/labels < 200 chars); exact dedup by text sha256. "
                   "Not an official text: verify against law.go.kr before citing.",
        "counts": counts,
        "network": ledger,
        "docs_sha256": file_sha256(out / "docs.jsonl"),
    }


ATTRIBUTION_TEXT = """출처: 법제처 국가법령정보센터 (https://www.law.go.kr), Open API (http://www.law.go.kr/DRF/).
Source: Korea Ministry of Government Legislation, National Law Information Center.

대상: 현행 법령, 현행 행정규칙, 현행 자치법규, 판례, 헌재결정례, 법령해석례, 행정심판재결례.
각 JSONL 행의 source는 해당 항목의 Open API 본문 주소입니다(OC 키를 붙여야 열립니다).

이용 근거: 저작권법 제7조(보호받지 못하는 저작물) 제1~3호. 헌법ㆍ법률ㆍ조약ㆍ명령ㆍ조례 및 규칙, 국가 또는
지방자치단체의 고시ㆍ공고ㆍ훈령, 법원의 판결ㆍ결정ㆍ명령 및 행정심판 등의 의결ㆍ결정은 저작권 보호를 받지 않습니다.
https://www.law.go.kr/lsLinkCommonInfo.do?lsJoLnkSeq=1029423769
법제처 Open API 이용안내: "제공된 법령정보는 영리 목적의 이용을 포함하여 자유로운 활용이 보장됩니다".
https://open.law.go.kr/LSO/information/guide.do
같은 안내는 법제처 외의 출처 표시나 원문 위조ㆍ변조가 저작권법 제137ㆍ138조로 처벌될 수 있다고 경고합니다.

변경: 마크업ㆍ개정 표시ㆍ'삭제' 조문ㆍ담당부서 연락처ㆍ전화번호ㆍ전자우편 제거, 최신 부칙만 유지, 서식 제외,
공백 정리, NFC 정규화, 200자 미만 문서 제거, 완전 중복 제거.
이 텍스트는 학습용 가공본이며 공식 법령 원문이 아닙니다. 인용할 때는 국가법령정보센터의 원문을 확인하세요.
"""


def run(args):
    out = Path(args.output)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    oc = args.oc or os.environ.get("TRID_LAW_OC") or "test"
    if oc == "test" and (args.limit is None or args.limit > 200):
        print("warning: OC=test is the public demo key; bulk collection needs your own key "
              "(open.law.go.kr, --oc or TRID_LAW_OC).", file=sys.stderr, flush=True)
    targets = args.targets.split(",") if args.targets else list(TARGETS)
    unknown = set(targets) - set(TARGETS)
    if unknown:
        raise SystemExit(f"unknown targets: {sorted(unknown)}")
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    client = Client(oc)
    ledger_path = raw / "ledger.json"
    ledger = load_json(ledger_path, {"requests": 0, "network_seconds": 0.0, "by_target": {}, "runs": []})
    run_start, totals = time.time(), {}
    # Docs stream to docs.jsonl.part; only a capped per-target sample (for estimates) stays in memory.
    samples, lengths, issuers, per = {}, {}, {}, {}
    seen_sha = set()
    part = out / "docs.jsonl.part"
    sink = part.open("w", encoding="utf-8")
    try:
        for target in targets:
            spec = TARGETS[target]
            stats = per[target] = {"selected": 0, "fetched_this_run": 0, "kept": 0, "dropped": Counter(),
                                   "characters": 0, "tokens": 0}
            t0, r0 = time.monotonic(), client.requests
            kept_docs = samples[target] = []
            lengths[target] = []
            def fetch_body(item):
                item_id = item[spec["id"]]
                return cached(item_path(raw, target, item_id),
                              lambda: client.get("lawService.do", {"target": target, spec["body"]: item_id}))
            for item, future in prefetch(select(client, raw, target, args.limit, args.seed), fetch_body, args.workers):
                stats["selected"] += 1
                item_id = item[spec["id"]]
                try:
                    body, fetched = future.result()
                except FetchError as e:
                    print(f"fetch_error {target} {item_id}: {e}", file=sys.stderr, flush=True)
                    stats["dropped"]["fetch_error"] += 1
                    continue
                stats["fetched_this_run"] += fetched
                title, text, reason = extract(target, body, item)
                if reason:
                    stats["dropped"][reason] += 1
                    continue
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if digest in seen_sha:
                    stats["dropped"]["exact_duplicate"] += 1
                    continue
                seen_sha.add(digest)
                doc = {"id": f"law-go-kr:{target}:{item_id}", "title": title, "source": source_url(item, spec),
                       "text": text, "sha256": digest, "license": LICENSE, "attribution": ATTRIBUTION,
                       "group": f"{target}:{item.get(spec['group']) or item_id}",
                       "date": norm_date(item.get(spec["date"])), "kind": target}
                if target == "ordin":
                    issuers[doc["id"]] = item.get("지자체기관명", "")
                stats["kept"] += 1
                stats["characters"] += len(text)
                stats["tokens"] += len(tokenizer.encode(text, add_special_tokens=False).ids)
                sink.write(json.dumps(doc, ensure_ascii=False) + "\n")
                lengths[target].append(len(text))
                if len(kept_docs) < NEAR_SAMPLE_CAP:
                    kept_docs.append(doc)
                if stats["selected"] % 25 == 0:
                    print(f"{target}: {stats['selected']} selected, {stats['kept']} kept, "
                          f"{client.requests} requests", flush=True)
            totals[target] = list_items((raw / target / "lists" / "count-display1.xml").read_bytes(), spec)[0]
            used = client.requests - r0
            if used:
                entry = ledger["by_target"].setdefault(target, {"requests": 0, "network_seconds": 0.0})
                entry["requests"] += used
                entry["network_seconds"] = round(entry["network_seconds"] + time.monotonic() - t0, 1)
            print(f"{target}: done, {stats['kept']}/{stats['selected']} kept", flush=True)
        ordin_docs = samples.get("ordin", [])[:ORDIN_WITHIN_CAP]
        r0 = client.requests
        ordinance = (ordinance_report(client, raw, ordin_docs, issuers, args.ordin_probe, args.seed)
                     if ordin_docs else None)
        probe_requests = client.requests - r0
    except AuthError as e:
        raise SystemExit(f"law.go.kr rejected OC: {e}")
    finally:
        sink.close()
        if client.requests:
            ledger["requests"] += client.requests
            ledger["network_seconds"] = round(ledger["network_seconds"] + time.time() - run_start, 1)
            ledger["runs"].append({"started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(run_start)),
                                   "requests": client.requests, "wall_seconds": round(time.time() - run_start, 1)})
            write_json(ledger_path, ledger)

    os.replace(part, out / "docs.jsonl")
    write_stats(out, args, per, samples, lengths, totals, ledger, ordinance, client.requests, probe_requests, run_start)
    counts = {t: {"selected": s["selected"], "kept": s["kept"], "dropped": dict(s["dropped"])} for t, s in per.items()}
    write_json(out / "provenance.json", provenance(out, oc, args, totals, counts, ledger))
    (out / "ATTRIBUTION.txt").write_text(ATTRIBUTION_TEXT, encoding="utf-8")
    print(json.dumps({"output": str(out), "docs": sum(s["kept"] for s in per.values()), "requests_this_run": client.requests},
                     ensure_ascii=False), flush=True)


def write_stats(out, args, per, samples, lengths, totals, ledger, ordinance, requests, probe_requests, run_start):
    targets, all_chars, all_tokens, all_selected, all_exact = {}, 0, 0, 0, 0
    extrapolation, est_seconds, est_tokens, est_tokens_dedup = {}, 0.0, 0, 0
    seconds_per_request = (ledger["network_seconds"] / ledger["requests"]) if ledger["requests"] else INTERVAL
    for target, s in per.items():
        exact = s["dropped"].get("exact_duplicate", 0)
        near = near_duplicate_estimate(samples.get(target, []))
        target_lengths = lengths.get(target, [])
        targets[target] = {
            "documents_fetched": s["selected"] - s["dropped"].get("fetch_error", 0),
            "fetched_this_run": s["fetched_this_run"], "kept": s["kept"], "dropped": dict(s["dropped"]),
            "characters": s["characters"], "tokens": s["tokens"],
            "chars_per_token": round(s["characters"] / s["tokens"], 4) if s["tokens"] else None,
            "median_chars": int(statistics.median(target_lengths)) if target_lengths else None,
            "mean_chars": int(statistics.mean(target_lengths)) if target_lengths else None,
            "exact_duplicate_rate": round(exact / (s["kept"] + exact), 4) if s["kept"] + exact else None,
            "near_duplicate": near,
        }
        all_chars += s["characters"]
        all_tokens += s["tokens"]
        all_selected += s["selected"]
        all_exact += exact
        total = totals.get(target, 0)
        requests_needed = total + math.ceil(total / DISPLAY) + 1
        by_target = ledger["by_target"].get(target)
        spr = (by_target["network_seconds"] / by_target["requests"]) if by_target and by_target["requests"] else seconds_per_request
        tokens_per_item = s["tokens"] / s["selected"] if s["selected"] else 0
        tokens = int(total * tokens_per_item)
        tokens_dedup = int(tokens * (1 - (near["rate"] or 0)))
        extrapolation[target] = {"total_listed": total, "requests": requests_needed,
                                 "seconds_per_request": round(spr, 3), "hours": round(requests_needed * spr / 3600, 1),
                                 "tokens": tokens, "tokens_after_pilot_near_dedup": tokens_dedup}
        est_seconds += requests_needed * spr
        est_tokens += tokens
        est_tokens_dedup += tokens_dedup
    all_kept = sum(s["kept"] for s in per.values())
    stats = {
        "tokenizer": str(TOKENIZER.relative_to(ROOT)),
        "mode": "pilot" if args.limit else "full",
        "limit_per_target": args.limit, "seed": args.seed,
        "documents_fetched": sum(t["documents_fetched"] for t in targets.values()),
        "kept": all_kept,
        "dropped": dict(sum((Counter(s["dropped"]) for s in per.values()), Counter())),
        "characters": all_chars, "tokens": all_tokens,
        "chars_per_token": round(all_chars / all_tokens, 4) if all_tokens else None,
        "exact_duplicate_rate": round(all_exact / (all_kept + all_exact), 4) if all_kept + all_exact else None,
        "near_duplicate": near_duplicate_estimate([d for docs in samples.values() for d in docs]),
        "near_duplicate_method": "word 5-shingles, 16-component MinHash (8 bands of 2) candidates, exact Jaccard >= 0.8 "
                                 "(scripts/prepare_korean_continuation.py NearDuplicates); approximate",
        "targets": targets,
        "ordinance_cross_municipality": ordinance,
        "run": {"wall_seconds": round(time.time() - run_start, 1), "requests": requests,
                "ordinance_probe_requests": probe_requests},
        "network_total": {"requests": ledger["requests"], "network_seconds": ledger["network_seconds"],
                          "seconds_per_request": round(seconds_per_request, 3)},
        "extrapolation_full_crawl": {
            "targets": extrapolation,
            "requests": sum(e["requests"] for e in extrapolation.values()),
            "days_at_measured_rate": round(est_seconds / 86400, 2),
            "tokens": est_tokens, "tokens_after_pilot_near_dedup": est_tokens_dedup,
            "method": "every list page + every body once at the measured seconds/request; tokens = listed items x "
                      "pilot tokens per selected item (drops included). Within-pilot near-dup rates understate "
                      "corpus-wide duplication (especially ordinances, see ordinance_cross_municipality).",
        },
    }
    write_json(out / "pilot-stats.json", stats)


def status(args):
    out = Path(args.output)
    raw = out / "raw"
    report = {"output": str(out), "targets": {}}
    for target in TARGETS:
        directory = raw / target
        report["targets"][target] = {
            "cached_items": len(list(directory.glob("*.xml"))) if directory.exists() else 0,
            "cached_list_pages": len(list((directory / "lists").glob("page-*.xml"))) if directory.exists() else 0,
        }
    docs = Counter()
    if (out / "docs.jsonl").exists():
        with (out / "docs.jsonl").open(encoding="utf-8") as f:
            for line in f:
                docs[json.loads(line)["kind"]] += 1
    for target, n in docs.items():
        report["targets"][target]["docs"] = n
    report["docs"] = sum(docs.values())
    report["network"] = {k: v for k, v in load_json(raw / "ledger.json", {}).items() if k != "runs"}
    stats = load_json(out / "pilot-stats.json", None)
    if stats:
        report["pilot"] = {k: stats[k] for k in ("mode", "kept", "tokens", "chars_per_token", "exact_duplicate_rate")}
        report["pilot"]["extrapolation"] = {k: v for k, v in stats["extrapolation_full_crawl"].items() if k != "targets"}
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--output", default="data/law-ko-source")
    p.add_argument("--limit", type=int, help="pilot: items per target from seeded random list pages")
    p.add_argument("--status", action="store_true", help="report cache and output state; no network")
    p.add_argument("--oc", help="Open API key (default: env TRID_LAW_OC, else the public demo key 'test')")
    p.add_argument("--targets", help=f"comma-separated subset of {','.join(TARGETS)}")
    p.add_argument("--seed", type=int, default=20261027)
    p.add_argument("--workers", type=int, default=8, help="concurrent body fetches (global rate limit still applies)")
    p.add_argument("--ordin-probe", type=int, default=20,
                   help="ordinances whose title is searched in other municipalities for near-duplicates")
    args = p.parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.status:
        status(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
