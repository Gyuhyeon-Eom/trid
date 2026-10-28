"""File → Document. txt/md read directly; .hwpx via its zip XML, .hwp via `rhwp export-doclang`, .pdf via
`pdftotext` (both local CLI tools, no network). The pure extraction helpers also serve
scripts/collect_korea_briefing.py, which imports them from here.
"""
import hashlib
import io
import re
import subprocess
import tempfile
import unicodedata
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

from . import acl as acl_module
from .structure import parse_document, is_regulation
from .types import Document

SUFFIXES = (".txt", ".md", ".hwpx", ".hwp", ".pdf")
HP = "{http://www.hancom.co.kr/hwpml/2011/paragraph}"
TOOLS = {"pdf": "pdftotext", "hwp": "rhwp"}  # rhwp: MIT, https://github.com/edwardkim/rhwp
PAGE_MARK = re.compile(r"^\s*(?:[-–—]\s*\d{1,3}\s*[-–—]|\d{1,3}\s*/\s*\d{1,3}|\f)\s*$")
BULLET = re.compile(r"^\s*(?:[□■○●◦◎◇◆▶▷►※*＊\-–·•ㅇ①-⑳<\[【(「]|\d{1,2}[.)]|[가-하][.)]|\(\d+\)|제\s*\d+\s*조)")
SENTENCE_END = re.compile(r"[.?!。:」』”\")]\s*$|다\s*$")
DOCLANG_SKIP = {"page_header", "page_footer", "picture"}
DOCLANG_CELL = {"fcel", "lcel", "ucel", "ecel", "xcel", "ched"}  # cell starts (first/left-merged/up-merged/empty/...)


def flat(text):
    return re.sub(r"\s+", " ", text).strip()


# ----------------------------------------------------------------------------- hwpx

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


# ----------------------------------------------------------------------------- hwp (rhwp) / pdf (pdftotext)

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


# ----------------------------------------------------------------------------- file → Document

def extract_text(path):
    path = Path(path)
    kind = path.suffix.lower().lstrip(".")
    data = path.read_bytes()
    if kind in ("txt", "md"):
        return data.decode("utf-8-sig", "replace")
    if kind == "hwpx":
        return hwpx_text(data)
    if kind in ("hwp", "pdf"):
        return tool_text(kind, data)
    raise ValueError(f"지원하지 않는 형식: {path.name} ({', '.join(SUFFIXES)})")


def normalize(text):
    """NFC, LF line ends, no trailing spaces, no form feeds, at most one blank line in a row."""
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip("\n")


def doc_id_for(path, root=None):
    rel = Path(path).relative_to(root) if root else Path(Path(path).name)
    stem = str(rel.with_suffix("")).replace("\\", "/")
    return re.sub(r"[^\w가-힣./-]+", "_", stem)


def title_of(text, fallback):
    for line in text.split("\n"):
        line = flat(line)
        if line:
            return line[:80]
    return fallback


def ingest_file(path, tags, root=None, redact=True):
    """Document with passages, refs and sha256 of the file bytes. `tags`: ACL tags; empty means nobody can see it."""
    path = Path(path)
    data = path.read_bytes()
    text = normalize(extract_text(path))
    if redact:
        from ..korean_quality import redact as redact_pii
        text, _ = redact_pii(text)
    doc_id = doc_id_for(path, root)
    title = title_of(text, path.stem)
    doc = Document(doc_id, title, hashlib.sha256(data).hexdigest(), set(tags), text)
    doc.passages, doc.refs = parse_document(text, doc_id, title)
    return doc


def ingest_dir(root, redact=True, log=None):
    """Every supported file under `root` (sorted, recursive) → [Document]. ACL tags come from
    `<file>.acl.json` or the directory's `acl.json` (trid.service.acl.document_tags)."""
    root = Path(root)
    docs, seen = [], set()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUFFIXES):
        tags = acl_module.document_tags(path, root)
        doc = ingest_file(path, tags, root, redact)
        if doc.doc_id in seen:
            doc.doc_id = f"{doc.doc_id}-{doc.sha256[:8]}"
            doc.passages, doc.refs = parse_document(doc.text, doc.doc_id, doc.title)
        seen.add(doc.doc_id)
        if log:
            log(f"{path.relative_to(root)}: {'규정' if is_regulation(doc.text) else '일반'} 문단 {len(doc.passages)} "
                f"참조 {len(doc.refs)} 권한 {sorted(doc.acl) or '없음(누구도 못 봄)'}")
        docs.append(doc)
    if not docs:
        raise ValueError(f"{root}에 문서가 없습니다 ({', '.join(SUFFIXES)}).")
    return docs
