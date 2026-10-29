"""Deterministic structure parser for Korean regulations (and a paragraph fallback for everything else).

A document is a regulation when at least two lines are 조 headings (`제12조(제목)`). Its text is cut into blocks
at 편/장/절 (path only, no passage), 조, 부칙, 별표/별지 서식 headings. A block longer than MAX_CHARS is split
at 항 (①②…), then 호 (1. 2.), 목 (가. 나.), lines, sentences, and finally whitespace; consecutive pieces are
packed greedily back up to MAX_CHARS. Every passage is a verbatim slice of the text (char_start/char_end), so
answers copied from a passage can be located in the document. References ("제12조에 따른", "제3조제2항",
"별표 2", "별지 제1호서식") are resolved to passage ids within the document; 「○○규정」 제N조 references are
resolved across documents by the index (external_references).
"""
import re

from .types import Passage

MAX_CHARS = 550
CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
MOK = "가나다라마바사아자차카타파하"

# Headings are recognised at line start only. `제N조` is a heading when a title in brackets, a 항 marker,
# '삭제' or the line end follows; `제12조에 따른 …` at line start is a reference, not a heading.
H_PYEON = re.compile(r"^제\s*(\d+)\s*편(?![가-힣])")
H_JANG = re.compile(r"^제\s*(\d+)\s*장(?![가-힣])")
H_JEOL = re.compile(r"^제\s*(\d+)\s*절(?![가-힣])")
H_JO = re.compile(r"^제\s*(\d+)\s*조(?:의\s*(\d+))?\s*(?=[(（【\[]|$|[①-⑳]|삭제)")
H_ANNEX = re.compile(r"^(?:■[^\n]*?\[\s*|[\[【(]?\s*)별표\s*(\d+(?:의\s*\d+)?)?\s*[\]】)]?(?=\s|$)")
H_FORM = re.compile(r"^(?:■[^\n]*?\[\s*|[\[【(]?\s*)(?:별지\s*)?(?:제\s*)?(\d+(?:의\s*\d+)?)\s*호\s*서식|^[\[【(]?\s*서식\s*(\d+)")
H_BUCHIK = re.compile(r"^부\s*칙(?![가-힣])")

# Split markers; line-start 호/목 may have gaps from repealed entries.
M_HANG = re.compile(r"(?:^|(?<=[\s)）\]】]))([①-⑳])")
M_HO = re.compile(r"(?:^|(?<=\s))(\d{1,3}(?:의\d+)?)\.\s(?!\d)")
M_MOK = re.compile(r"(?:^|(?<=\s))([가-하])\.\s")
M_PARA = re.compile(r"\n[ \t]*\n")
M_LINE = re.compile(r"\n")
M_SENT = re.compile(r"(?<=[.다])\s+")

_EXT_ARTICLE = r"제\s*\d+\s*조(?:의\s*\d+)?(?:\s*(?:ㆍ|및|,)?\s*제\s*\d+\s*[항호])*"
R_EXT = re.compile(r"「([^」]+)」\s*(?:\([^)\n]*\)\s*)?(" + _EXT_ARTICLE
                   + r"(?:(?:\s*(?:및|부터|까지|ㆍ|,)\s*)+" + _EXT_ARTICLE + r")*)")
R_JO = re.compile(r"제\s*(\d+)\s*조(?:의\s*(\d+))?(?:\s*제\s*(\d+)\s*항)?")
R_HANG = re.compile(r"(?<![조\d])제\s*(\d+)\s*항")
R_ANNEX = re.compile(r"별표\s*(\d+(?:의\s*\d+)?)")
R_FORM = re.compile(r"별지\s*제?\s*(\d+)\s*호\s*서식|서식\s*제?\s*(\d+)\s*호?")


def jo_label(m):
    return f"제{m.group(1)}조" + (f"의{m.group(2)}" if m.group(2) else "")


def hang_label(n):
    return CIRCLED[n - 1] if 1 <= n <= len(CIRCLED) else None


def is_regulation(text):
    return sum(1 for line in text.split("\n") if H_JO.match(line.strip())) >= 2


def external_references(text):
    """[(document title, 조 label)] for 「○○규정」 제N조 mentions."""
    return [(m.group(1).strip(), jo_label(article)) for m in R_EXT.finditer(text)
            for article in R_JO.finditer(m.group(2))]


# ----------------------------------------------------------------------------- blocks

def _blocks(text, title):
    """[(path, start, end, kind)] with kind in {'pre', 'jo', 'annex', 'buchik'}; 편/장/절 lines are dropped."""
    blocks, open_block = [], None
    pyeon = jang = jeol = None
    section = "본문"
    pos = 0

    def close(end):
        nonlocal open_block
        if open_block:
            path, start, kind = open_block
            if text[start:end].strip():
                blocks.append((path, start, end, kind))
            open_block = None

    def context():
        if section == "부칙":
            return (title, "부칙")
        return (title,) + tuple(x for x in (pyeon, jang, jeol) if x)

    open_block = ((title, "전문"), 0, "pre")
    for line in text.split("\n"):
        start, end = pos, pos + len(line)
        pos = end + 1
        s = line.strip()
        if not s:
            continue
        lead = start + (len(line) - len(line.lstrip()))
        in_annex = open_block and open_block[2] == "annex"
        if (m := H_ANNEX.match(s)):
            close(start)
            open_block = ((title, "별표" + (f" {m.group(1)}" if m.group(1) else "")), lead, "annex")
        elif (m := H_FORM.match(s)):
            close(start)
            open_block = ((title, f"서식 {m.group(1) or m.group(2)}"), lead, "annex")
        elif H_BUCHIK.match(s):
            close(start)
            section, pyeon, jang, jeol = "부칙", None, None, None
            open_block = ((title, "부칙"), lead, "buchik")
        elif in_annex:
            continue
        elif (m := H_JO.match(s)):
            close(start)
            open_block = (context() + (jo_label(m),), lead, "jo")
        elif (m := H_PYEON.match(s)):
            close(start)
            pyeon, jang, jeol = f"제{m.group(1)}편", None, None
        elif (m := H_JANG.match(s)):
            close(start)
            jang, jeol = f"제{m.group(1)}장", None
        elif (m := H_JEOL.match(s)):
            close(start)
            jeol = f"제{m.group(1)}절"
    close(len(text))
    return blocks


# ----------------------------------------------------------------------------- splitting

def _trim(text, start, end):
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _markers(text, start, end, level):
    """[(position, label)] of split markers of `level` inside [start, end)."""
    s = text[start:end]
    if level == "hang":
        return [(start + m.start(1), m.group(1)) for m in M_HANG.finditer(s)]
    if level in ("ho", "mok"):
        regex, out, previous = (M_HO, [], None) if level == "ho" else (M_MOK, [], None)
        for m in regex.finditer(s):
            value = tuple(map(int, m.group(1).split("의"))) if level == "ho" else (MOK.find(m.group(1)),)
            # Repealed entries leave gaps; inserted 호 use e.g. 2의2. Only line-start
            # entries may begin mid-list, avoiding ordinary inline decimal numbers.
            at_line_start = not s[s.rfind("\n", 0, m.start(1)) + 1:m.start(1)].strip()
            first = value == ((1,) if level == "ho" else (0,)) or at_line_start
            if value[0] >= 0 and ((previous is None and first) or (previous is not None and value > previous)):
                out.append((start + m.start(1), m.group(1) + "."))
                previous = value
        return out
    regex = {"para": M_PARA, "line": M_LINE, "sent": M_SENT}[level]
    return [(start + m.end(), None) for m in regex.finditer(s) if 0 < m.start()]


def _segments(text, start, end, level):
    marks = [(p, label) for p, label in _markers(text, start, end, level) if start <= p < end]
    if not marks:
        return None
    segs, prev, prev_label = [], start, None
    for p, label in marks:
        if p > prev and text[prev:p].strip():
            segs.append((prev, p, prev_label))
        prev, prev_label = p, label
    segs.append((prev, end, prev_label))
    return segs if len(segs) > 1 else None


def _hard(text, start, end, limit):
    """Pieces of at most `limit` chars cut at whitespace when possible."""
    segs = []
    while end - start > limit:
        cut = text.rfind(" ", start + limit // 2, start + limit)
        cut = cut if cut > start else start + limit
        segs.append((start, cut, None))
        start = cut
    segs.append((start, end, None))
    return segs


def _pack(segs, limit):
    groups, current = [], []
    for seg in segs:
        if current and seg[1] - current[0][0] > limit:
            groups.append(current)
            current = []
        current.append(seg)
    if current:
        groups.append(current)
    return groups


def _group_label(group):
    labels = [label for _, _, label in group if label]
    if not labels:  # only the head of its parent (e.g. the 조 heading line): keep the parent's path
        return None
    return labels[0] if len(labels) == 1 else f"{labels[0]}~{labels[-1]}"


HEAD_MERGE = 80  # a heading-only head ("제12조(국내출장여비)") joins an over-long first 항 instead of standing alone


def _split(text, start, end, path, levels, limit, out):
    start, end = _trim(text, start, end)
    if start >= end:
        return
    if end - start <= limit:
        out.append((path, start, end))
        return
    for i, level in enumerate(levels):
        segs = _segments(text, start, end, level)
        if segs:
            if segs[0][2] is None and segs[0][1] - segs[0][0] <= HEAD_MERGE and segs[1][1] - segs[0][0] > limit:
                segs = [(segs[0][0], segs[1][1], segs[1][2])] + segs[2:]
            for group in _pack(segs, limit):
                label = _group_label(group)
                _split(text, group[0][0], group[-1][1], path + ((label,) if label else ()), levels[i + 1:], limit, out)
            return
    for n, (a, b, _) in enumerate(_hard(text, start, end, limit), 1):
        out.append((path + (str(n),),) + _trim(text, a, b))


LEVELS = {"jo": ("hang", "ho", "mok", "line", "sent"), "buchik": ("hang", "ho", "mok", "line", "sent"),
          "annex": ("para", "line", "sent"), "pre": ("para", "line", "sent")}


# ----------------------------------------------------------------------------- references

def _hang_range(label):
    """'②' -> ['②']; '②~④' -> ['②', '③', '④']."""
    return list(CIRCLED[CIRCLED.index(label[0]):CIRCLED.index(label[-1]) + 1])


def _resolve_refs(text, passages, jo_of):
    """(from, to) pairs inside one document. jo_of: passage index -> its 조 label (None outside 조 blocks)."""
    by_jo, by_hang, by_name = {}, {}, {}
    for i, p in enumerate(passages):
        if (jo := jo_of[i]) and jo not in by_jo and p.path[1] != "부칙":
            by_jo[jo] = p.id
        for k, label in enumerate(p.path):
            if k and (label.startswith("별표") or label.startswith("서식")) and label not in by_name:
                by_name[label] = p.id
            if jo and k and label[0] in CIRCLED and p.path[1] != "부칙":
                for h in _hang_range(label):
                    by_hang.setdefault((jo, h), p.id)
    refs, seen = [], set()

    def add(src, dst):
        if dst and dst != src and (src, dst) not in seen:
            seen.add((src, dst))
            refs.append((src, dst))

    def blank(m):
        return " " * len(m.group())

    for i, p in enumerate(passages):
        body = R_EXT.sub(blank, p.text)  # external refs are not local ones
        for m in R_JO.finditer(body):
            jo = jo_label(m)
            hang = hang_label(int(m.group(3))) if m.group(3) else None
            if jo == jo_of[i] and not hang:
                continue
            # An unbound "법 제N조"/"같은 법 제N조" is not this document.
            prefix = body[max(0, m.start() - 16):m.start()]
            if re.search(r"(?:법|영|규칙|조례)\s*$", prefix) and not re.search(r"이\s+(?:법|영|규칙|조례)\s*$", prefix):
                continue
            add(p.id, (by_hang.get((jo, hang)) if hang else None) or by_jo.get(jo))
        if jo_of[i]:
            for m in R_HANG.finditer(R_JO.sub(blank, body)):
                add(p.id, by_hang.get((jo_of[i], hang_label(int(m.group(1))))))
        for m in R_ANNEX.finditer(body):
            add(p.id, by_name.get(f"별표 {m.group(1)}"))
        if "별표" in body and "별표" in by_name and not R_ANNEX.search(body):
            add(p.id, by_name["별표"])
        for m in R_FORM.finditer(body):
            add(p.id, by_name.get(f"서식 {m.group(1) or m.group(2)}"))
    return refs


# ----------------------------------------------------------------------------- entry

def parse_document(text, doc_id, title, limit=MAX_CHARS):
    """(passages, refs) for one document's normalised text."""
    pieces = []
    if is_regulation(text):
        for path, start, end, kind in _blocks(text, title):
            _split(text, start, end, path, LEVELS[kind], limit, pieces)
    else:
        _split(text, 0, len(text), (title,), ("para", "line", "sent"), limit, pieces)
        pieces = [((title, f"¶{n}"), a, b) for n, (_, a, b) in enumerate(pieces, 1)]
    passages = [Passage(f"{doc_id}:{n:04d}", doc_id, text[a:b], a, b, path) for n, (path, a, b) in enumerate(pieces, 1)]
    jo_of = [next((x for x in p.path[1:] if H_JO.match(x)), None) for p in passages]
    return passages, _resolve_refs(text, passages, jo_of)
