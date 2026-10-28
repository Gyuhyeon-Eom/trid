"""Collect 정책브리핑(korea.kr) press releases that carry the 공공누리 제1유형 notice.

Writes the shared pretraining-source layout used by collect_wiki_ko_full.py and collect_law_go_kr.py:
DIR/raw/ (list pages, view pages, attachments, per-item extraction cache), DIR/docs.jsonl,
DIR/provenance.json, DIR/ATTRIBUTION.txt, DIR/pilot-stats.json.

Enumeration is month by month from 2015-01 (pressReleaseList.do, 20 items per page). An item is kept
only if its view page says "이 자료는 텍스트에 한하여 공공누리 제1유형". The body comes from the first
attached document, preferring HWPX, then PDF (pdftotext), then HWP (rhwp export-doclang, MIT); the page body is used
only when no attachment yields usable text. Worker threads share one global request gap (0.1 s); every
response cached so a re-run skips finished items.
"""
import argparse, calendar, collections, datetime, hashlib, html, io, json, os, random, re, shutil
import http.client, subprocess, sys, tempfile, threading, time, unicodedata, urllib.error, urllib.request, urllib.robotparser, zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trid.corpus import sha256, write_json, prefetch

SITE = "https://www.korea.kr"
LIST_URL = SITE + "/briefing/pressReleaseList.do?startDate={start}&endDate={end}&pageIndex={page}"
VIEW_URL = SITE + "/briefing/pressReleaseView.do?newsId={news_id}"
FILE_URL = SITE + "/common/download.do?fileId={file_id}&tblKey=GMN"
USER_AGENT = ("trid-research/0.1 (Korean LM pretraining corpus collector for KOGL Type 1 press releases; "
              "one connection, max 10 requests/s)")
TOKENIZER = ROOT / "runs/trid-korean-medium-base-v1/tokenizer.json"
LICENSE = "KOGL-Type1"
KOGL_NOTICE = "이 자료는 텍스트에 한하여 공공누리 제1유형"
RETRY_WAITS = (5, 15, 45, 135, 300, 300, 300)  # ~18 min of outage before a request is given up
NETWORK = (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException)
NETWORK_ERROR = re.compile(r"URLError|TimeoutError|ConnectionError|IncompleteRead|RemoteDisconnected|HTTPError\((?:5\d\d|'5)")

def transient(e):
    """Network failure worth retrying later, as opposed to a permanent 4xx."""
    return isinstance(e, NETWORK) and not (isinstance(e, urllib.error.HTTPError) and e.code < 500)
CREDIT = "출처: 정책브리핑(www.korea.kr)"
FIRST_MONTH = (2015, 1)
PER_PAGE = 20
MIN_CHARS = 200
MAX_DOWNLOAD = 60 * 1024 * 1024
FORMATS = ("hwpx", "pdf", "hwp")  # preference order
TOOLS = {"pdf": "pdftotext", "hwp": "rhwp"}  # rhwp: MIT, https://github.com/edwardkim/rhwp (replaced AGPL hwp5txt)
HP = "{http://www.hancom.co.kr/hwpml/2011/paragraph}"


# ----------------------------------------------------------------------------- HTTP

class Fetcher:
    """HTTP shared by worker threads: one global minimum gap between request starts, robots.txt checks."""

    def __init__(self, log_path, interval=0.1):
        self.interval, self.last, self.log_path = interval, 0.0, Path(log_path)
        self.lock = threading.Lock()
        self.log = json.loads(self.log_path.read_text()) if self.log_path.exists() else \
            {"requests": 0, "bytes": 0, "seconds": 0.0, "errors": 0}
        self.run = {"requests": 0, "bytes": 0, "seconds": 0.0, "errors": 0}
        self.robots = None

    def allowed(self, url):
        if self.robots is None:
            self.robots = urllib.robotparser.RobotFileParser()
            self.robots.parse(self.get(SITE + "/robots.txt", check=False).decode("utf-8", "replace").splitlines())
        return self.robots.can_fetch(USER_AGENT, url)

    def get(self, url, check=True):
        if check and not self.allowed(url):
            raise PermissionError(f"robots.txt disallows {url}")
        for attempt in range(len(RETRY_WAITS) + 1):
            with self.lock:
                wait = self.last + self.interval - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                self.last = start = time.monotonic()
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=90) as response:
                    body = response.read(MAX_DOWNLOAD + 1)
                self._count(start, len(body))
                if len(body) > MAX_DOWNLOAD:
                    raise ValueError(f"response over {MAX_DOWNLOAD} bytes: {url}")
                return body
            except urllib.error.HTTPError as e:
                self._count(start, 0, error=True)
                if e.code < 500 or attempt == len(RETRY_WAITS):
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                self._count(start, 0, error=True)
                if attempt == len(RETRY_WAITS):
                    raise
            time.sleep(RETRY_WAITS[attempt])

    def _count(self, start, size, error=False):
        with self.lock:
            for d in (self.log, self.run):
                d["requests"] += 1; d["bytes"] += size; d["seconds"] += time.monotonic() - start
                d["errors"] += error

    def save(self):
        with self.lock:
            self.log["seconds"] = round(self.log["seconds"], 1)
            write_json(self.log_path, self.log)


def cached(fetcher, url, path, refresh=False):
    path = Path(path)
    if path.exists() and not refresh:
        return path.read_bytes()
    body = fetcher.get(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_bytes(body)
    os.replace(part, path)
    return body


# ----------------------------------------------------------------------------- HTML parsing

def strip_tags(fragment):
    fragment = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", fragment)
    fragment = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</li>|</tr>|</h\d>", "\n", fragment)
    return html.unescape(re.sub(r"<[^>]+>", " ", fragment))


def flat(text):
    return re.sub(r"\s+", " ", text).strip()


def parse_list(page):
    """(total count, [(newsId, date, ministry)]) from a pressReleaseList.do page."""
    total = re.search(r"총\s*<strong>([\d,]+)</strong>\s*건", page)
    items = []
    for m in re.finditer(r'href="/briefing/pressReleaseView\.do\?newsId=(\d+)&amp;pageIndex=.*?</a>', page, re.S):
        spans = re.findall(r"<span>([^<]*)</span>", m.group(0).split('class="source"', 1)[-1])
        date = next((s.strip() for s in spans if re.fullmatch(r"\s*\d{4}\.\d{2}\.\d{2}\s*", s)), None)
        ministry = next((flat(s) for s in spans if s.strip() and s.strip() != date), None)
        items.append((m.group(1), date and date.replace(".", "-"), ministry))
    return (int(total.group(1).replace(",", "")) if total else 0), items


def parse_view(page):
    """Metadata, licence flag, attachments and page body text from a pressReleaseView.do page."""
    title = re.search(r'id="titleText"[^>]*value="([^"]*)"', page) or re.search(r"<title>(.*?)</title>", page, re.S)
    title = flat(html.unescape(title.group(1))) if title else ""
    title = re.sub(r"\s*-\s*보도자료\s*\|.*$", "", title)
    date = re.search(r'<div class="info">\s*<span>(\d{4})\.(\d{2})\.(\d{2})</span>', page)
    attachments = []
    for m in re.finditer(r'<a href="/common/download\.do\?fileId=(\d+)&amp;tblKey=GMN">\s*<img[^>]*>([^<]*)</a>', page):
        name = flat(html.unescape(m.group(2)))
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        attachments.append({"file_id": m.group(1), "name": name, "ext": ext})
    body = re.search(r'<div class="view_cont">(.*?)<div class="article_footer">', page, re.S)
    body = body.group(1) if body else ""
    remark = re.search(r'<p class="remark">(.*?)</p>', body, re.S)
    ministry_line = flat(strip_tags(remark.group(1))).strip("“”\" ") if remark else None
    body = re.sub(r'(?is)<p class="remark">.*?</p>|<iframe\b.*?</iframe>', " ", body)
    text = strip_tags(body)
    page_text = flat(strip_tags(page))
    return {"title": title, "date": "-".join(date.groups()) if date else None,
            "kogl_type1": KOGL_NOTICE in page_text, "ministry_line": ministry_line,
            "attachments": attachments, "page_text": "\n".join(flat(x) for x in text.splitlines() if flat(x))}


def ministry_of(line):
    m = re.search(r"이 자료는\s*(.+?)의\s*보도자료를\s*전재", line or "")
    return m.group(1).strip() if m else None


# ----------------------------------------------------------------------------- attachment text

def _run_text(t):
    parts = [t.text or ""]
    for c in t:
        if c.tag in (HP + "tab", HP + "nbSpace", HP + "fwSpace", HP + "lineBreak"):
            parts.append(" ")
        parts.append(c.tail or "")
    return "".join(parts)


def _paragraphs(container):
    for p in container:
        if p.tag == HP + "p":
            yield from _paragraph(p)


def _paragraph(p):
    own, objects = [], []
    for run in p.findall(HP + "run"):
        for child in run:
            if child.tag == HP + "t":
                own.append(_run_text(child))
            else:
                objects.append(child)
    yield "".join(own)
    for obj in objects:
        yield from _objects(obj)


def _objects(e):
    if e.tag == HP + "tbl":
        for tr in e.findall(HP + "tr"):
            cells = [" ".join(flat(x) for x in _objects(tc) if flat(x)) for tc in tr.findall(HP + "tc")]
            cells = [c for c in cells if c]
            if len(cells) == 1:  # one-cell boxes often hold whole paragraphs; keep their lines
                yield from (x for tc in tr.findall(HP + "tc") for x in _objects(tc))
            elif cells:
                yield " ".join(cells)
    elif e.tag == HP + "subList":
        yield from _paragraphs(e)
    else:
        for c in e:
            yield from _objects(c)


def hwpx_text(data):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = sorted((n for n in z.namelist() if re.fullmatch(r"Contents/section\d+\.xml", n)),
                       key=lambda n: int(re.search(r"\d+", n.rsplit("/", 1)[1]).group()))
        lines = []
        for name in names:
            if z.getinfo(name).file_size > 200 * 1024 * 1024:
                raise ValueError("section too large")
            lines.extend(_paragraphs(ET.fromstring(z.read(name))))
    return "\n".join(lines)


def tool_text(kind, data, timeout=180):
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / f"in.{kind}"
        src.write_bytes(data)
        if kind == "pdf":
            out = subprocess.run(["pdftotext", "-enc", "UTF-8", str(src), "-"], capture_output=True,
                                 timeout=timeout, check=True).stdout
            return join_pdf_lines(out.decode("utf-8", "replace"))
        xml = Path(d) / "out.xml"  # --assets-dir keeps pictures out of the XML (else inlined as base64)
        subprocess.run(["rhwp", "export-doclang", str(src), "-o", str(xml), "--assets-dir", d],
                       capture_output=True, timeout=timeout, check=True)
        return doclang_text(xml.read_bytes())


DOCLANG_SKIP = {"page_header", "page_footer", "picture"}
DOCLANG_CELL = {"fcel", "lcel", "ucel", "ecel", "xcel", "ched"}  # cell starts (first/left-merged/up-merged/empty/...)


def doclang_text(xml):
    """rhwp DocLang v0.6 -> one line per paragraph; table rows joined like hwpx_text (<nl/> ends a row)."""
    lines = []

    def para(e):
        return re.sub(r"\s+", " ", "".join(e.itertext())).strip()

    def walk(e):
        for c in e:
            if c.tag == "text":
                lines.append(para(c))
            elif c.tag == "table":
                table(c)
            elif c.tag not in DOCLANG_SKIP:
                walk(c)

    def table(t):
        cells = []

        def row():
            texts = [" ".join(x for x in cell if x) for cell in cells if any(cell)]
            if len(texts) == 1:  # one-cell boxes often hold whole paragraphs; keep their lines
                lines.extend(x for cell in cells for x in cell if x)
            elif texts:
                lines.append(" ".join(texts))
            cells.clear()

        for c in t:
            if c.tag in DOCLANG_CELL:
                cells.append([])
            elif c.tag == "nl":
                row()
            elif c.tag == "text":
                if not cells:
                    cells.append([])
                cells[-1].append(para(c))
            elif c.tag == "table":
                table(c)
            elif c.tag not in DOCLANG_SKIP:
                walk(c)
        row()

    walk(ET.fromstring(xml))
    return "\n".join(lines)


BULLET = re.compile(r"^\s*(?:[□■○●◦◎◇◆▶▷►※*＊\-–·•ㅇ①-⑳<\[【(「]|\d{1,2}[.)]|[가-하][.)]|\(\d+\)|제\s*\d+\s*조)")
SENTENCE_END = re.compile(r"[.?!。:」』”\")]\s*$|다\s*$")


def join_pdf_lines(text):
    """pdftotext breaks every visual line; rejoin lines that continue a paragraph."""
    out = []
    for raw in text.replace("\f", "\n\f\n").splitlines():
        line = raw.strip()
        if (out and line and out[-1] and out[-1] != "\f" and not BULLET.match(line)
                and not SENTENCE_END.search(out[-1]) and not PAGE_MARK.match(line) and not PAGE_MARK.match(out[-1])):
            out[-1] += " " + line
        else:
            out.append(line)
    return "\n".join(out)


def available_tools():
    return {name: shutil.which(name) for name in TOOLS.values()}


# ----------------------------------------------------------------------------- cleaning

PHONE = re.compile(r"(?<![\d.-])(?:\(?0\d{1,2}\)?[\s.)-]?\d{3,4}[\s.-]\d{4}|1[5-9]\d{2}-\d{4})(?![\d-])")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PAGE_MARK = re.compile(r"^\s*(?:[-–—]\s*\d{1,3}\s*[-–—]|\d{1,3}\s*/\s*\d{1,3}|\f)\s*$")
HEADER_KEY = re.compile(r"^[\s<\[(【]*(?:보도\s*자료\s*)?(?:보도\s*시점|보도\s*일시|보도\s*시각|배포\s*일시|배포\s*시각|배\s*포|"
                        r"엠바고|담당\s*부서|담당\s*과|담\s*당\s*자|책\s*임\s*자|문\s*의|연\s*락\s*처|자료\s*문의)"
                        r"(?=[\s:：)\]>】]|$)")
HEADER_LINE = re.compile(r"^[\s<]*(?:보도\s*자료|보도\s*참고\s*자료|참고\s*자료|설명\s*자료|해명\s*자료|"
                         r"제\s*\d+\s*[-–]\s*\d+\s*호)[\s>]*$")
PICTURE = re.compile(r"원본 그림의 (?:이름|크기)|^\s*그림입니다\.?|^\s*사진 찍은 날짜|^\s*<(?:표|그림)>\s*$")
CONTACT_PROSE = re.compile(r"취재를 원하시면|연락\s*주시기\s*바랍니다")
DATE_ONLY = re.compile(r"^\d{4}\s*\.\s*\d{1,2}\s*\.\s*\d{1,2}\s*\.?\s*(?:\(\s*\w\s*\))?\s*(?:\d{1,2}\s*:\s*\d{2})?"
                       r"\s*(?:부터|까지|이후)?$")
PRIVATE_USE = re.compile("[\ue000-\uf8ff\U000f0000-\U000ffffd\U00100000-\U0010fffd]")


def clean_text(text):
    """NFC text with header/contact block, picture comments, page markers, phones and emails removed."""
    text = unicodedata.normalize("NFC", text).replace("\x00", "").replace("\r", "\n")
    lines = []
    for raw in text.split("\n"):
        line = re.sub(r"[ \t\u00a0\u3000\u200b]+", " ", PRIVATE_USE.sub(" ", raw)).strip()
        if not line or PAGE_MARK.match(line) or HEADER_LINE.match(line) or PICTURE.search(line):
            lines.append("")
            continue
        if HEADER_KEY.match(line) or CONTACT_PROSE.search(line) or DATE_ONLY.match(line) or \
                ((PHONE.search(line) or "☎" in line) and len(line) < 120):
            lines.append("")
            continue
        line = EMAIL.sub("", PHONE.sub("", line)).replace("☎", "")
        line = re.sub(r"\(\s*[:：,/]*\s*\)", "", line)
        line = re.sub(r" {2,}", " ", line).strip()
        lines.append(line)
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def keep(text):
    """(cleaned text, None) or (None, drop reason) for an extracted body."""
    text = clean_text(text)
    if len(text) < MIN_CHARS:
        return None, "too_short"
    return text, None


def attribution(ministry_line, ministry=None):
    line = ministry_line or (f"이 자료는 {ministry}의 보도자료를 전재하여 제공함을 알려드립니다." if ministry else None)
    return f"{line} {CREDIT}" if line else CREDIT


# ----------------------------------------------------------------------------- enumeration

def months(until):
    y, m = FIRST_MONTH
    while (y, m) <= until:
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def list_page(fetcher, raw, y, m, page, today):
    last = calendar.monthrange(y, m)[1]
    url = LIST_URL.format(start=f"{y:04d}-{m:02d}-01", end=f"{y:04d}-{m:02d}-{last:02d}", page=page)
    fresh = (today - datetime.date(y, m, last)).days < 14  # recent months still change
    body = cached(fetcher, url, raw / "list" / f"{y:04d}-{m:02d}" / f"p{page:05d}.html", refresh=fresh)
    return parse_list(body.decode("utf-8", "replace"))


def census(fetcher, raw, today):
    """Total releases per month, from page 1 of every month (cached; reused by the full crawl)."""
    return {f"{y:04d}-{m:02d}": list_page(fetcher, raw, y, m, 1, today)[0] for y, m in months((today.year, today.month))}


def pilot_selection(fetcher, raw, totals, limit, seed, today):
    """Seeded sample spread evenly over years: random months, random pages, random items."""
    path = raw / f"pilot-selection-{limit}-{seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    rng = random.Random(seed)
    years = sorted({k[:4] for k, v in totals.items() if v})
    quota = {y: limit // len(years) + (i < limit % len(years)) for i, y in enumerate(rng.sample(years, len(years)))}
    chosen = []
    for year in years:
        candidates = [k for k, v in totals.items() if k.startswith(year) and v]
        picks = rng.sample(candidates, min(2, len(candidates)))
        for i, month in enumerate(picks):
            want = quota[year] // len(picks) + (i < quota[year] % len(picks))
            y, m = map(int, month.split("-"))
            page = rng.randint(1, max(1, -(-totals[month] // PER_PAGE)))
            items = list_page(fetcher, raw, y, m, page, today)[1]
            for news_id, date, ministry in rng.sample(items, min(want, len(items))):
                chosen.append({"news_id": news_id, "list_date": date, "list_ministry": ministry, "month": month})
    write_json(path, chosen)
    return chosen


def full_enumeration(fetcher, raw, totals, today):
    for month, total in totals.items():
        y, m = map(int, month.split("-"))
        for page in range(1, -(-total // PER_PAGE) + 1):
            for news_id, date, ministry in list_page(fetcher, raw, y, m, page, today)[1]:
                yield {"news_id": news_id, "list_date": date, "list_ministry": ministry, "month": month}


# ----------------------------------------------------------------------------- items

def pick_attachments(attachments):
    """Attachments of the first document (same file stem), ordered by format preference."""
    docs = [a for a in attachments if a["ext"] in FORMATS]
    if not docs:
        return []
    stem = docs[0]["name"].rsplit(".", 1)[0]
    same = [a for a in docs if a["name"].rsplit(".", 1)[0] == stem]
    return sorted(same, key=lambda a: FORMATS.index(a["ext"]))


def process_item(fetcher, raw, item, tools, keep_files=True):
    """Fetch and extract one release; cached as raw/items/<newsId>.json."""
    news_id = item["news_id"]
    out = raw / "items" / f"{news_id}.json"
    if out.exists():
        record = json.loads(out.read_text())
        # A page-only fallback caused by a network failure on an attachment is retried, not trusted.
        if not (record.get("body_source") == "page" and any(NETWORK_ERROR.search(a.get("error") or "") for a in record.get("attempts", []))):
            return record
    record = {**item, "url": VIEW_URL.format(news_id=news_id), "error": None}
    try:
        page = cached(fetcher, record["url"], raw / "view" / f"{news_id}.html").decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001 - recorded and retried on the next run
        return {**record, "error": f"view: {e!r}"}
    view = parse_view(page)
    record.update({k: v for k, v in view.items() if k != "page_text"})
    record["formats"] = sorted({a["ext"] for a in view["attachments"] if a["ext"] in FORMATS})
    record["body_source"], record["body"], record["attempts"] = None, "", []
    if view["kogl_type1"]:
        for a in pick_attachments(view["attachments"]):
            if a["ext"] in TOOLS and not tools.get(TOOLS[a["ext"]]):
                record["attempts"].append({"file_id": a["file_id"], "ext": a["ext"], "error": "tool_missing"})
                continue
            path = raw / "files" / f"{news_id}-{a['file_id']}.{a['ext']}"
            try:
                data = cached(fetcher, FILE_URL.format(file_id=a["file_id"]), path)
                text = hwpx_text(data) if a["ext"] == "hwpx" else tool_text(a["ext"], data)
            except NETWORK as e:
                if transient(e):  # outage: do not persist a page-only fallback; the next run retries
                    return {**record, "error": f"file: {e!r}"}
                record["attempts"].append({"file_id": a["file_id"], "ext": a["ext"], "error": repr(e)[:300]})
                continue
            except Exception as e:  # noqa: BLE001 - unusable file; try the next format
                record["attempts"].append({"file_id": a["file_id"], "ext": a["ext"], "error": repr(e)[:300]})
                continue
            finally:
                if not keep_files:
                    path.unlink(missing_ok=True)
            usable = len(clean_text(text)) >= MIN_CHARS
            record["attempts"].append({"file_id": a["file_id"], "ext": a["ext"], "chars": len(text), "usable": usable})
            if usable:
                record["body_source"], record["body"], record["file"] = a["ext"], text, a
                break
        if record["body_source"] is None:
            record["body_source"], record["body"] = "page", view["page_text"]
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, record)
    return record


def build_doc(record):
    """(docs.jsonl row, None) or (None, drop reason)."""
    if record.get("error"):
        return None, "fetch_error"
    if not record["kogl_type1"]:
        return None, "no_kogl_type1"
    text, reason = keep(record["body"])
    if text is None:
        return None, reason
    return {"id": f"korea-kr-press-{record['news_id']}", "title": unicodedata.normalize("NFC", record["title"]),
            "source": record["url"], "text": text, "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "license": LICENSE,
            "attribution": unicodedata.normalize("NFC", attribution(record["ministry_line"], record.get("list_ministry"))),
            "group": f"korea-kr:{record['news_id']}", "date": record["date"] or record.get("list_date"),
            "kind": "press_release", "body_source": record["body_source"]}, None


# ----------------------------------------------------------------------------- stats and output

def near_duplicates(texts, threshold=0.8):
    """Exact all-pairs 5-word-shingle Jaccard. Pilot only (quadratic); the full corpus is near-deduplicated
    by scripts/build_pretraining_mix.py."""
    from scripts.prepare_korean_continuation import shingles
    sets = [shingles(t) for t in texts]
    flagged = set()
    for i in range(len(sets)):
        for j in range(i):
            if j in flagged:
                continue
            a, b = sets[i], sets[j]
            if min(len(a), len(b)) / max(len(a), len(b)) >= threshold and len(a & b) / len(a | b) >= threshold:
                flagged.add(i)
                break
    return len(flagged)


def light(record):
    """Fields the stats and extrapolation need; drops extracted text so a full crawl stays small in memory."""
    return {"error": record.get("error"), "kogl_type1": record.get("kogl_type1"), "formats": record.get("formats", []),
            "month": record.get("month"), "attempts": [{"error": a.get("error")} for a in record.get("attempts", [])]}


def collect(output, limit=None, seed=20261027, keep_files=True, workers=8):
    from tokenizers import Tokenizer
    root = Path(output)
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    start = time.time()
    tools = available_tools()
    for name, path in tools.items():
        if not path:
            print(f"warning: {name} not found; that attachment format is skipped", file=sys.stderr, flush=True)
    tok = Tokenizer.from_file(str(TOKENIZER))
    fetcher = Fetcher(raw / "fetch-log.json")
    today = datetime.date.today()
    # Stream: each item is fetched (or read from raw/items), turned into a row and written at once.
    records, seen, dropped = [], set(), collections.Counter()
    chars = tokens = kept = 0
    body_sources, kept_years, pilot_texts = collections.Counter(), collections.Counter(), []
    part = root / "docs.jsonl.part"
    try:
        totals = census(fetcher, raw, today)
        items = pilot_selection(fetcher, raw, totals, limit, seed, today) if limit else \
            full_enumeration(fetcher, raw, totals, today)
        work = lambda item: process_item(fetcher, raw, item, tools, keep_files)
        with part.open("w", encoding="utf-8") as f:
            for n, (item, future) in enumerate(prefetch(items, work, workers), 1):
                record = future.result()
                records.append(light(record))
                if record.get("kogl_type1") and not record.get("error"):
                    body_sources[f"all_kogl:{record.get('body_source')}"] += 1
                row, reason = build_doc(record)
                if row and row["sha256"] in seen:
                    row, reason = None, "exact_duplicate"
                if row is None:
                    dropped[reason] += 1
                else:
                    seen.add(row["sha256"])
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    kept += 1; chars += len(row["text"]); tokens += len(tok.encode(row["text"]).ids)
                    body_sources[f"kept:{row['body_source']}"] += 1
                    kept_years[(row["date"] or "????")[:4]] += 1
                    if limit:
                        pilot_texts.append(row["text"])
                if n % 25 == 0:
                    fetcher.save()
                    print(f"{n} items, {kept} kept, {fetcher.run['requests']} requests", flush=True)
    finally:
        fetcher.save()
    os.replace(part, root / "docs.jsonl")

    fetched = [r for r in records if not r.get("error")]
    kogl = [r for r in fetched if r["kogl_type1"]]
    near = near_duplicates(pilot_texts) if limit else None
    raw_bytes = sum(p.stat().st_size for p in raw.rglob("*") if p.is_file())
    stats = {
        "mode": f"pilot --limit {limit} --seed {seed}" if limit else "full",
        "documents_fetched": len(fetched), "fetch_errors": len(records) - len(fetched),
        "kogl_type1_pass": len(kogl), "kogl_type1_pass_rate": round(len(kogl) / max(1, len(fetched)), 4),
        "documents_kept": kept, "dropped": dict(dropped),
        "characters": chars, "tokens": tokens, "chars_per_token": round(chars / max(1, tokens), 3),
        "mean_tokens_per_kept_doc": round(tokens / max(1, kept), 1),
        "tokenizer": str(TOKENIZER.relative_to(ROOT)),
        "exact_duplicate_rate": round(dropped["exact_duplicate"] / max(1, kept + dropped["exact_duplicate"]), 4),
        "near_duplicate_docs": near, "near_duplicate_rate": None if near is None else round(near / max(1, kept), 4),
        "near_duplicate_method": "word 5-shingles (scripts/prepare_korean_continuation.shingles), exact all-pairs Jaccard >= 0.8"
                                 if limit else "not computed here; done at mixing time by scripts/build_pretraining_mix.py",
        "attachment_formats_available": dict(collections.Counter("+".join(r["formats"]) or "none" for r in kogl)),
        "body_source_kept": {k[5:]: v for k, v in body_sources.items() if k.startswith("kept:")},
        "body_source_all_kogl": {k[9:]: v for k, v in body_sources.items() if k.startswith("all_kogl:")},
        "kept_by_year": dict(sorted(kept_years.items())),
        "fetched_by_year": dict(sorted(collections.Counter(r["month"][:4] for r in fetched).items())),
        "tools": tools,
        "this_run": {"wall_seconds": round(time.time() - start, 1), **{k: round(v, 1) for k, v in fetcher.run.items()}},
        "cumulative_fetch": fetcher.log, "raw_bytes": raw_bytes,
    }
    stats["extrapolation"] = extrapolate(stats, totals, records, limit)
    write_json(root / "pilot-stats.json", stats)
    write_provenance(root, stats, totals, kept)
    print(json.dumps({k: stats[k] for k in ("documents_fetched", "kogl_type1_pass_rate", "documents_kept", "dropped",
                                             "tokens", "body_source_kept", "kept_by_year")}, ensure_ascii=False), flush=True)
    return stats


def extrapolate(stats, totals, records, limit):
    """Full-crawl size and duration from the census and the pilot's per-item costs."""
    releases = sum(totals.values())
    fetched = [r for r in records if not r.get("error")]
    per_item = (1 + sum(len([a for a in r.get("attempts", []) if a.get("error") != "tool_missing"]) for r in fetched)
                / max(1, len(fetched)))
    list_pages = sum(-(-v // PER_PAGE) for v in totals.values())
    requests = list_pages + releases * per_item
    log = stats["cumulative_fetch"]
    seconds_per_request = max(1.0, log["seconds"] / max(1, log["requests"]))
    keep_rate = stats["documents_kept"] / max(1, len(fetched))
    return {"releases_listed_since_2015": releases, "list_pages": list_pages,
            "requests_per_item": round(per_item, 2), "estimated_requests": round(requests),
            "seconds_per_request": round(seconds_per_request, 2),
            "estimated_days": round(requests * seconds_per_request / 86400, 1),
            "estimated_kept_documents": round(releases * keep_rate),
            "estimated_tokens": round(releases * keep_rate * stats["mean_tokens_per_kept_doc"]),
            "estimated_raw_gb_with_attachments": round(stats["raw_bytes"] / max(1, len(fetched)) * releases / 1e9, 1),
            "note": "pilot sample is stratified by year, not by month volume; treat as order-of-magnitude"
                    if limit else "measured"}


def write_provenance(root, stats, totals, kept):
    write_json(root / "provenance.json", {
        "source": "대한민국 정책브리핑 보도자료 (https://www.korea.kr/briefing/pressReleaseList.do)",
        "publisher": "문화체육관광부 국민소통실 정책브리핑; 원 작성 기관은 각 행의 attribution",
        "license": LICENSE,
        "license_basis": "Each kept view page states: '이 자료는 텍스트에 한하여 공공누리 제1유형(출처표시)의 조건에 따라 "
                         "이용할 수 있습니다.' Items without that notice are dropped.",
        "license_reference": "https://www.kogl.or.kr/info/licenseTypeEn.do",
        "copyright_policy": "https://www.korea.kr/guide/copyRight.do",
        "text_only": "Images, photos and other non-text material are not collected (KOGL notice covers text only).",
        "robots_txt": "https://www.korea.kr/robots.txt (User-Agent: * Allow: /), checked before every request",
        "user_agent": USER_AGENT, "rate_limit": "one connection, >= 0.1 s between requests",
        "body_preference": "first attached document: HWPX (Contents/section*.xml) > PDF (pdftotext) > "
                           "HWP (rhwp export-doclang, MIT; items cached before 2026-10-28 were extracted with "
                           "hwp5txt/pyhwp — tool output is plain text and carries no AGPL obligation) > page text",
        "changes": "NFC; removed 보도시점/배포/담당부서/담당자 header and contact lines, phone numbers, emails, "
                   "'그림입니다' picture comments, page markers; PDF line breaks rejoined; docs < 200 chars dropped",
        "deduplication": "exact text sha256 within this source; near-duplicate filtering happens at mixing time",
        "months": totals, "mode": stats["mode"], "documents": kept,
        "docs_sha256": sha256(root / "docs.jsonl"), "collected": datetime.date.today().isoformat(),
        "real_company_data": False})
    (root / "ATTRIBUTION.txt").write_text(
        "Source: 대한민국 정책브리핑 보도자료, https://www.korea.kr\n"
        "출처: 정책브리핑(www.korea.kr). Each JSONL row names the originating ministry "
        "('이 자료는 ○○의 보도자료를 전재하여 제공함을 알려드립니다.') and links the original page.\n"
        "License: 공공누리 제1유형(출처표시), text only. Only pages stating '이 자료는 텍스트에 한하여 "
        "공공누리 제1유형(출처표시)의 조건에 따라 이용할 수 있습니다' are included.\n"
        "https://www.kogl.or.kr/info/licenseTypeEn.do\nhttps://www.korea.kr/guide/copyRight.do\n"
        "Changes: text extracted from HWPX/PDF/HWP attachments, header/contact lines, phone numbers, emails, "
        "picture comments and page markers removed, NFC normalization, length filter, exact deduplication.\n",
        encoding="utf-8")


def status(root):
    raw = root / "raw"
    stats = root / "pilot-stats.json"
    info = {"items_cached": len(list((raw / "items").glob("*.json"))) if (raw / "items").exists() else 0,
            "list_pages_cached": len(list((raw / "list").rglob("*.html"))) if (raw / "list").exists() else 0,
            "docs_jsonl_rows": sum(1 for _ in (root / "docs.jsonl").open()) if (root / "docs.jsonl").exists() else 0,
            "fetch_log": json.loads((raw / "fetch-log.json").read_text()) if (raw / "fetch-log.json").exists() else None,
            "tools": available_tools()}
    if stats.exists():
        info["pilot_stats"] = json.loads(stats.read_text())
    print(json.dumps(info, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", default="data/korea-kr-briefing-source")
    p.add_argument("--limit", type=int, help="pilot: fetch N releases sampled across years (seeded)")
    p.add_argument("--seed", type=int, default=20261027)
    p.add_argument("--drop-attachments", action="store_true",
                   help="delete downloaded attachments after extraction (item JSON keeps the text)")
    p.add_argument("--status", action="store_true")
    p.add_argument("--workers", type=int, default=8, help="concurrent item fetches (global request gap still applies)")
    a = p.parse_args()
    if a.status:
        return status(Path(a.output))
    collect(a.output, a.limit, a.seed, keep_files=not a.drop_attachments, workers=a.workers)


if __name__ == "__main__":
    main()
