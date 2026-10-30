"""Six-type training candidates (condition, exception, table, combine, numeric change, explanation) at scale.

Reuses the C-002 item format and builder (scripts/build_task_pilot.py) and the teacher-QA llama.cpp helpers.

  select   candidate pools per type from allowed korea.kr briefings (KOGL Type 1). Excluded: frozen evaluation and
           public-document SFT documents, teacher QA selections, C-002 pilot documents, documents whose passages occur in
           development prompts, and quarantined documents. One document is used for at most one type. Tables are
           searched in the whole corpus: runs of lines with a label and >= 3 numeric cells under one header line.
  draft    a generator model writes one item per candidate in the C-002 format (or declines). Answer fields must be
           verbatim spans; numeric-change edits are applied by this script, not the model.
  check    every item goes through the C-002 builder rules one by one (verbatim spans, answer absent from the
           distractor, training fit and gold-length generation room in 1,024 tokens) and deterministic form rules
           (no meta text in the question, answer not inside the question, a type marker near the answer).
  verify   a different model sees only the built document context and the question and answers on its own; a second
           call judges the candidate against the documents (support, completeness, missing condition, target type,
           other valid answers, question clarity). Acceptance also needs content agreement with the independent
           answer (all numbers, otherwise >= 60% of words). String containment is recorded separately and never
           accepts an item. Any 'unsure' or missing agreement quarantines the item; any judge failure rejects it.
  export   accepted items -> C-002 spec (one item per document) for scripts/build_task_pilot.py build/check.

Every candidate ends in exactly one state in status.jsonl: declined, draft_error, check_reject, verify_error,
semantic_reject, quarantine or accepted. Questions and explanation framing are model-drafted; nothing here is
human-reviewed.
"""
import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from build_teacher_qa import EXCLUDE, SOURCE, append_jsonl, model_id, norm, passages, read_jsonl  # noqa: E402
import build_task_candidates_strong as S  # noqa: E402

TYPES = ('condition', 'exception', 'table', 'combine', 'numeric_change', 'explanation')
PATTERNS = {
    'condition': re.compile(r'(경우에는|경우에만|요건|자격|이상인|이하인|해당하는 경우|충족|대상은|신청할 수)'),
    'exception': re.compile(r'(제외|다만|예외|적용하지 않|해당하지 않|면제)'),
    'numeric_change': re.compile(r'(?<![\d.])\d[\d,.]*\s?(만\s?원|억\s?원|천\s?원|조\s?원|원|만\s?명|명|%|개소|개사|개|건|톤|배|ha|㎡|km)'),
    'explanation': re.compile(r'(위해|때문|목적|취지|함으로써)'),
}
NUMERIC_CELL = re.compile(r'^[△\-]?[\d,.]+%?$')
NOISE = re.compile(r'\(\s+|\s+\)|\s[,.]\s|\*\*|\s{2,}')
DEV_PROMPTS = (Path('/tmp/c004/rag310.json'), Path('/tmp/c004/pilot30.json'), Path('/tmp/c004/probe40.json'),
               ROOT / 'reports/coordination/C-003/prompts.json')

RULES = ('규칙: 질문은 기관·사업·제도·행사 이름을 넣어 문단 없이도 무엇을 묻는지 알 수 있어야 하고, 질문에 답을 넣지 마세요. '
         '답은 직접 쓰지 말고 아래 후보 구간 목록에서 번호로 고르세요(구간은 문단 원문 그대로입니다). '
         '알맞은 내용이나 후보가 없거나 확실하지 않으면 skip을 true로 하세요.')
DRAFT = {
    'condition': ('다음 문단에서 어떤 조건·자격·요건을 갖춰야 하는지(누가, 어떤 경우에) 묻는 질문 1개를 만들고, '
                  '그 조건을 담은 후보 구간 번호를 answer_id로 고르세요. ' + RULES, {'answer_id': 'clause'}),
    'exception': ('다음 문단에서 적용 제외·예외·단서(다만, 제외, 면제) 내용을 묻는 질문 1개를 만들고, '
                  '그 예외를 담은 후보 구간 번호를 answer_id로 고르세요. ' + RULES, {'answer_id': 'clause'}),
    'explanation': ('다음 문단에서 정책·조치의 목적이나 이유를 묻고 "근거와 함께 설명해 주세요."로 끝나는 질문 1개를 만드세요. '
                    'conclusion_id는 목적·이유를 담은 짧은 구간, evidence_id는 그 근거가 되는 문장, '
                    'condition_id는 적용 조건·대상·범위를 담은 짧은 구간이며 문단에 실제로 없으면 0입니다. 조건을 지어내지 마세요. ' + RULES,
                    {'conclusion_id': 'purpose', 'evidence_id': 'sentence', 'condition_id': 'clause0'}),
    'numeric_change': ('다음 문단의 수치 하나를 묻는 질문 1개를 만들고, 그 수치 구간 번호를 span_id로 고르세요. ' + RULES,
                       {'span_id': 'number'}),
    'combine': ('같은 문서의 두 문단 A, B가 있습니다. 하나의 대상(같은 사업·제도·행사)에 대해 A의 사실과 B의 사실이 모두 있어야 '
                '완전히 답할 수 있는 질문 1개를 만드세요(예: 같은 사업의 지원 대상과 지원 금액, 같은 제도의 시행일과 적용 범위). '
                '서로 무관한 두 사실을 묶지 마세요. value_a_id는 A 후보에서, value_b_id는 B 후보에서 고릅니다. ' + RULES,
                {'value_a_id': 'clause_a', 'value_b_id': 'clause_b'}),
    'table': ('다음 표에서 "{row}" 행의 "{col}" 값을 묻는 한국어 질문 1개를 쓰세요. 질문에는 표의 주제, 행 이름 "{row}", 열 이름 "{col}"이 모두 '
              '들어가야 하며 값은 쓰지 마세요. 예: "○○ 현황 표에서 서울의 2020년 값은 얼마인가요?"', ['question']),
}
ANSWER_PROMPT = ('다음 문서만 근거로 질문에 답하세요. 문서에 답이 없으면 "확인 불가"라고 쓰세요. 답에 조건이나 대상이 있으면 함께 쓰고, '
                 '여러 개면 모두 쓰세요. 세 문장 이내로 답하세요.\n\n{context}\n\n질문: {question}')
JUDGE_PROMPT = ('문서, 질문, 답안 A(검토 대상), 답안 B(다른 사람이 문서만 보고 쓴 답)가 있습니다. 문서를 기준으로 판정해 JSON으로만 출력하세요.\n'
                'supported: 문서가 답안 A를 뒷받침하는가(yes/no/unsure)\n'
                'complete: 답안 A가 질문이 묻는 범위를 빠짐없이 답하는가(yes/no/unsure)\n'
                'condition_missing: 질문에 답하려면 필요한 조건·대상·단위가 A에 빠졌는가(yes/no/unsure)\n'
                'target_ok: A가 질문이 묻는 종류(사람·기관·날짜·금액·조건 등)와 같은가(yes/no/unsure)\n'
                'other_answer: 문서에 A와 다른 정답이 따로 있는가(yes/no/unsure)\n'
                'question_clear: 질문만으로 무엇을 묻는지 분명한가(yes/no/unsure)\n'
                'agrees_with_b: A와 B가 같은 내용인가(yes/no/unsure)\n\n'
                '{context}\n\n질문: {question}\n답안 A: {candidate}\n답안 B: {independent}')
JUDGE_KEYS = ('supported', 'complete', 'condition_missing', 'target_ok', 'other_answer', 'question_clear', 'agrees_with_b')
GOOD = {'supported': 'yes', 'complete': 'yes', 'condition_missing': 'no', 'target_ok': 'yes', 'other_answer': 'no',
        'question_clear': 'yes', 'agrees_with_b': 'yes'}


def schema(fields, sizes=None):
    """Draft schema: skip, question and, per id field, an integer enum over the numbered candidate list."""
    props = {'skip': {'type': 'boolean'}, 'question': {'type': 'string'}}
    for field, kind in fields.items():
        n = sizes[kind.rstrip('0')]
        props[field] = {'type': 'integer', 'enum': list(range(0 if kind.endswith('0') else 1, n + 1))}
    return {'type': 'object', 'required': ['skip', 'question', *fields], 'properties': props}


def complete(url, prompt, max_tokens, json_schema=None):
    """One chat call; returns (content, meta) with the server's stop reason and token counts, so truncation is visible."""
    import urllib.request

    def post(path, payload):
        req = urllib.request.Request(url + path, json.dumps(payload).encode(), {'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r)
    rendered = post('/apply-template', {'messages': [{'role': 'user', 'content': prompt}],
                                        'chat_template_kwargs': {'enable_thinking': False}})['prompt']
    payload = dict(prompt=rendered, n_predict=max_tokens, temperature=0, seed=42, cache_prompt=False)
    if json_schema:
        payload['json_schema'] = json_schema
    r = post('/completion', payload)
    meta = dict(stop_type=r.get('stop_type'), truncated=r.get('truncated'), prompt_tokens=r.get('tokens_evaluated'),
                output_tokens=r.get('tokens_predicted'), max_tokens=max_tokens)
    return r['content'].strip(), meta


def sentence_list(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if 10 <= len(s.strip()) <= 220]


def clause_list(text):
    out = []
    for s in re.split(r'(?<=[.!?])\s+', text):
        for c in re.split(r'(?<=[,])\s+|\s+(?=다만)|\s+(?=[□○ㅇ▸※*-]\s)', s):
            c = re.sub(r'^[□○ㅇ▸※*\-◦•·\s]+', '', c).strip().rstrip('.,').strip()
            if 4 <= len(c) <= 90 and c not in out:
                out.append(c)
    return out[:40]


def number_list(text):
    """Quantities only (money, people, percent, counts, weight, area, length); dates and years are not edited."""
    out = []
    for m in re.finditer(r'(?<![\d.])\d[\d,.]*\s?(?:만\s?원|억\s?원|천\s?원|조\s?원|원|만\s?명|명|%|개소|개사|개|건|톤|배|ha|㎡|km)', text):
        span = m.group(0).strip()
        if span not in out:
            out.append(span)
    return out[:30]


META = re.compile(r'구간|번호|질문|묻는|만드세요|작성|다음과 같|편집된|수정된')
MARKERS = {
    'condition': re.compile(r'경우|이상|이하|미만|초과|대상|자격|요건|해당|충족|이면|라면|한하|기준|이내|부터|까지'),
    'exception': re.compile(r'제외|다만|예외|면제|않|아니|불가|단,'),
    'explanation': re.compile(r'위해|위하여|위한|때문|목적|취지|함으로써|도록'),
}


def form_check(item, text='', markers=True):
    """Deterministic question/answer form rules applied to every item (model-drafted or carried over). The type-marker
    regexes belong to the small-generator profile only; strong-profile items are judged by the verifier instead."""
    kind, q = item['type'], item['question']
    if META.search(q) or not 8 <= len(q) <= 200:
        raise ValueError('question form: meta text or length')
    answer = item.get('answer') or item.get('conclusion') or ', '.join(p[2] for p in item.get('parts', []))
    if norm(answer) and norm(answer) in norm(q):
        raise ValueError('question contains the answer')
    window = answer
    if kind in ('condition', 'exception') and answer in text:
        at = text.find(answer)
        window = text[max(0, at - 40):at + len(answer) + 40]   # the marker may sit next to the answer (e.g. '…은 면제')
    if markers and kind in MARKERS and not MARKERS[kind].search(window):
        raise ValueError(f'{kind} answer has no {kind} marker')
    if kind == 'numeric_change' and not re.search(r'\d', answer):
        raise ValueError('numeric answer has no number')
    if kind == 'combine' and norm(item['parts'][0][2]) == norm(item['parts'][1][2]):
        raise ValueError('combine values identical')


def agrees(candidate, independent):
    """Content agreement with the independent answer: every number of the candidate appears in it, otherwise >= 60% of
    the candidate's words do. Necessary for acceptance, never sufficient (the judge must also pass)."""
    flat = independent.replace(',', '')
    numbers = [n.replace(',', '').rstrip('.') for n in re.findall(r'\d[\d,.]*', candidate)]
    if numbers:
        return all(n in flat for n in numbers)
    words = re.findall(r'[가-힣A-Za-z0-9]{2,}', candidate)
    return bool(words) and (sum(w in independent for w in words) / len(words) >= .6 or norm(candidate) in norm(independent))


def purpose_list(text):
    """Purpose/reason spans that start at a clause start and end at the marker (e.g. '…을 강화하기 위해')."""
    out = []
    for clause in clause_list(text):
        m = re.match(r'(?:[가-힣A-Za-z0-9‘“\'"「｢(]).{4,88}?(?:위해서|위하여|위해|위한|때문에|목적으로|취지에서|취지로)', clause)
        if m and m.group(0) not in out:
            out.append(m.group(0))
    return out[:20]


def numbered(items):
    return '\n'.join(f'[{n}] {x}' for n, x in enumerate(items, 1))


JUDGE_SCHEMA = {'type': 'object', 'required': list(JUDGE_KEYS),
                'properties': {k: {'type': 'string', 'enum': ['yes', 'no', 'unsure']} for k in JUDGE_KEYS}}


def excluded_ids(extra):
    ids = {d['id'] for p in EXCLUDE for d in read_jsonl(p)}
    for path in extra:
        path = Path(path)
        if path.suffix == '.json':      # C-002-format spec
            spec = json.loads(path.read_text())
            ids |= {i['doc'] for i in spec['items'] if 'doc' in i} | {p[0] for i in spec['items'] for p in i.get('parts', [])}
            ids |= {t['doc'] for t in spec.get('tables', {}).values()}
            ids |= {d for s in spec.get('distractors', {}).values() for d, _ in s}
        elif path.suffix == '.jsonl':   # rows with id / doc_id
            ids |= {r.get('id') or r.get('doc_id') for r in read_jsonl(path)}
        else:                           # plain id list
            ids |= {x.strip() for x in path.read_text().split() if x.strip()}
    return ids


def find_table(text, aligned=True):
    """First run of >= 4 one-row-per-line numeric rows. Also returns the raw block from the header/caption lines above the
    rows down to the 8th row, verbatim (header hierarchy kept), with start/end lines that locate it uniquely.
    aligned=True (small-generator profile) additionally requires a single header line aligned with the row width."""
    lines = text.split('\n')
    run = []
    for k, line in enumerate(lines + ['']):
        toks = line.split()
        n = next((n for n in range(1, len(toks)) if all(NUMERIC_CELL.match(x) for x in toks[n:])), None)
        if n is not None and len(toks) - n >= 3 and len(line) < 160:
            run.append((k, [' '.join(toks[:n])] + toks[n:]))
            continue
        if len(run) >= 4:
            widths = Counter(len(r) for _, r in run)
            width, count = widths.most_common(1)[0]
            rows = [r for _, r in run if len(r) == width]
            first = run[0][0]
            header_line = lines[first - 1].split() if first > 0 else []
            caption = next((lines[j].strip() for j in range(first - 2, max(-1, first - 5), -1)
                            if j >= 0 and lines[j].strip()), '')
            top = first
            while top > 0 and first - top < 5 and lines[top - 1].strip() and len(lines[top - 1]) < 160 \
                    and not (top < first and lines[top].lstrip().startswith(('<', '[', '【'))):   # caption line reached
                top -= 1
            last = run[min(7, len(run) - 1)][0]
            start, end = lines[top].strip(), lines[last].strip()
            offset = sum(len(x) + 1 for x in lines[:top]) + lines[top].find(start)
            located = top < first and text.find(start) == offset and text.find(end) >= offset and \
                text.find(end) == sum(len(x) + 1 for x in lines[:last]) + lines[last].find(end)
            ok = count >= 4 and len(rows) == len(run) and len(header_line) in (width - 1, width) if aligned \
                else count >= 4 and located
            if ok:
                return dict(caption=caption, header_raw=lines[first - 1].strip(), rows=rows[:8], width=width,
                            raw_start=start, raw_end=end, raw_block='\n'.join(lines[top:last + 1]) if located else None)
        run = []
    return None


def select(work, per_type, seed, extra_exclude, profile='small', only=None):
    excluded = excluded_ids(extra_exclude)
    allowed = excluded_ids([only]) if only else None
    dev_text = re.sub(r'\s+', '', ''.join(r['prompt'] for p in DEV_PROMPTS if p.exists() for r in json.loads(p.read_text())['results']))
    pools, skipped = defaultdict(list), Counter()
    with SOURCE.open() as f:
        for line in f:
            doc = json.loads(line)
            if doc['id'] in excluded or (allowed is not None and doc['id'] not in allowed):
                skipped['excluded' if doc['id'] in excluded else 'not_in_only'] += 1
                continue
            cut = passages(doc['text'])
            if any(re.sub(r'\s+', '', p)[:80] in dev_text for p in cut):
                skipped['dev_prompt_overlap'] += 1
                continue
            h = int(hashlib.sha256((seed + doc['id']).encode()).hexdigest(), 16)
            base = dict(doc=doc['id'], title=doc['title'], source=doc['source'], key=h)
            table = find_table(doc['text'], aligned=profile == 'small')
            if table:
                pools['table'].append(dict(base, table=table))
            if not cut:
                continue
            clean = [i for i, p in enumerate(cut) if not NOISE.search(p)]
            for kind, pat in PATTERNS.items():
                hits = [i for i in clean if pat.search(cut[i])]
                if hits:
                    pools[kind].append(dict(base, passages={i: cut[i] for i in hits[:2]}))
            if len(clean) >= 2:
                pools['combine'].append(dict(base, passages={i: cut[i] for i in clean[:4]}))
    used, out = set(), []
    for kind in sorted(TYPES, key=lambda k: len(pools[k])):   # scarce types pick first
        picked = 0
        for cand in sorted(pools[kind], key=lambda c: c['key']):
            if cand['doc'] in used:
                continue
            used.add(cand['doc'])
            out.append(dict(cand, type=kind, cid=f"{kind}:{cand['doc']}"))
            picked += 1
            if picked == per_type:
                break
    if profile == 'strong':
        add_repeated(out)
    work.mkdir(parents=True, exist_ok=True)
    (work / 'candidates.jsonl').write_text(''.join(json.dumps({k: v for k, v in c.items() if k != 'key'}, ensure_ascii=False) + '\n' for c in out))
    meta = dict(seed=seed, per_type=per_type, profile=profile, pool_sizes={k: len(v) for k, v in pools.items()},
                selected=dict(Counter(c['type'] for c in out)), skipped=dict(skipped), excluded_ids=len(excluded),
                extra_exclude=[str(p) for p in extra_exclude], only=str(only) if only else None)
    (work / 'select.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps(meta, ensure_ascii=False))


def add_repeated(cands):
    """Strong profile, numeric_change: per passage, the values that also appear in the title or elsewhere in the document
    (second pass over the source; see build_task_candidates_strong.repeated_values)."""
    want = {c['doc']: c for c in cands if c['type'] == 'numeric_change'}
    with SOURCE.open() as f:
        for line in f:
            doc = json.loads(line)
            c = want.get(doc['id'])
            if c:
                c['repeated'] = {i: S.repeated_values(p, doc.get('title', ''), doc['text']) for i, p in c['passages'].items()}


def edit_number(span):
    """Change the first number in the span to a different plausible value with the same formatting."""
    m = re.search(r'\d[\d,]*(\.\d+)?', span)
    if not m:
        return None
    raw = m.group(0)
    digits = raw.replace(',', '')
    if '.' in digits:
        whole, frac = digits.split('.')
        value = f"{whole}.{(int(frac) + 3) % (10 ** len(frac)):0{len(frac)}d}"
        if value == digits:
            return None
    else:
        n = int(digits)
        value = str(n + max(1, round(n * .15)) if n >= 3 else n + 2)
        if ',' in raw:
            value = f'{int(value):,}'
    return span[:m.start()] + value + span[m.end():]


def draft_one(url, cand):
    """Returns (draft, meta). Answer fields are filled from the chosen candidate numbers, never from model text."""
    kind = cand['type']
    prompt, fields = DRAFT[kind]
    if kind == 'table':
        t = cand['table']
        r = random.Random(cand['cid']).randrange(len(t['rows']))
        c = random.Random(cand['cid'] + 'c').randrange(1, t['width'])
        header = t['header_raw'].split()
        col = header[c - (t['width'] - len(header))] if len(header) >= t['width'] - 1 else f'{c}번째 값'
        rendered = '\n'.join([t['caption'], t['header_raw']] + [' '.join(x) for x in t['rows']])
        out, meta = complete(url, prompt.format(row=t['rows'][r][0], col=col) + f'\n\n표:\n{rendered}', 160,
                             {'type': 'object', 'required': fields, 'properties': {f: {'type': 'string'} for f in fields}})
        return dict(json.loads(out), row=r, cell=c, answer=t['rows'][r][c], column=col), meta
    if kind == 'combine':
        (ia, pa), (ib, pb) = list(cand['passages'].items())[:2]
        lists = {'clause_a': clause_list(pa), 'clause_b': clause_list(pb)}
        body = f"문단 A:\n{pa}\n\nA 후보 구간:\n{numbered(lists['clause_a'])}\n\n문단 B:\n{pb}\n\nB 후보 구간:\n{numbered(lists['clause_b'])}"
        extra = dict(passage_a=int(ia), passage_b=int(ib))
    else:
        i, p = next(iter(cand['passages'].items()))
        lists = {'clause': clause_list(p), 'sentence': sentence_list(p), 'number': number_list(p), 'purpose': purpose_list(p)}
        shown = {k: lists[k.rstrip('0')] for k in set(fields.values())}
        label = {'sentence': '문장', 'number': '수치', 'purpose': '목적·이유'}
        body = f'문단:\n{p}\n\n' + '\n\n'.join(f"{label.get(k.rstrip('0'), '구간')} 후보:\n{numbered(v)}"
                                               for k, v in sorted(shown.items()))
        extra = dict(passage=int(i))
    sizes = {k: len(v) for k, v in lists.items()}
    if any(sizes[k.rstrip('0')] == 0 for k in fields.values()):
        return dict(skip=True, question='', reason='no candidate spans', **extra), None
    out, meta = complete(url, f'{prompt}\n\n{body}', 200, schema(fields, sizes))
    reply = json.loads(out)
    for field, k in fields.items():
        n = reply.get(field, 0)
        reply[field.replace('_id', '')] = lists[k.rstrip('0')][n - 1] if n else ''
    return dict(reply, **extra), meta


TRANSIENT = (OSError, TimeoutError)


def latest(path):
    """Last record per candidate. Earlier records (e.g. transient failures before a retry) stay in the file."""
    return {r['cid']: r for r in read_jsonl(path)} if Path(path).exists() else {}


def settled(path):
    """Candidates whose last record is final; a transient failure (timeout, connection) is retried on resume."""
    return {c for c, r in latest(path).items() if not r.get('transient')}


def draft(work, url, workers, profile='small'):
    out = work / 'drafts.jsonl'
    done = settled(out)
    gen = model_id(url)
    jobs = [c for c in read_jsonl(work / 'candidates.jsonl') if c['cid'] not in done]
    one = S.draft_one if profile == 'strong' else draft_one

    def run(cand):
        try:
            result, meta = one(url, cand)
            return dict(cid=cand['cid'], type=cand['type'], generator=gen, error=None, draft=result, call=meta)
        except (ValueError, OSError, KeyError, IndexError) as exc:
            return dict(cid=cand['cid'], type=cand['type'], generator=gen, error=f'{type(exc).__name__}: {exc}', draft=None,
                        transient=isinstance(exc, TRANSIENT))

    with ThreadPoolExecutor(workers) as pool:
        for n, row in enumerate(pool.map(run, jobs), 1):
            append_jsonl(out, [row])
            if n % 50 == 0:
                print(f'drafted {n}/{len(jobs)}', flush=True)


def snap(span, text, repaired):
    """The verbatim passage span for a model-copied span: exact, whitespace-insensitive, or the longest common block
    when it covers >= 85% of the model span. Anything else is rejected; repaired spans are recorded."""
    span = span.strip().rstrip('.')
    if not span:
        raise ValueError('empty span')
    if span in text:
        return span
    pattern = r'\s*'.join(re.escape(ch) for ch in re.sub(r'\s+', '', span))
    m = re.search(pattern, text)
    if m:
        repaired.append('whitespace')
        return m.group(0)
    from difflib import SequenceMatcher
    block = SequenceMatcher(None, text, span, autojunk=False).find_longest_match(0, len(text), 0, len(span))
    if block.size >= .85 * len(span) and block.size >= 4:
        repaired.append('partial')
        return text[block.a:block.a + block.size].strip()
    raise ValueError(f'not a verbatim span: {span[:40]!r}')


def unique_span(span, text):
    """Extend a non-unique span with right context until it occurs once in the passage."""
    start = text.find(span)
    end = start + len(span)
    while text.count(text[start:end]) > 1 and end < len(text) and end - start < 40:
        end += 1
    if text.count(text[start:end]) != 1:
        raise ValueError('span not unique in passage')
    return text[start:end].rstrip()


def to_item(cand, d):
    """C-002 spec item (and table spec) from a draft; raises ValueError for unusable drafts."""
    kind, doc = cand['type'], cand['doc']
    iid = f"{kind}-{doc}"
    q = d['question'].strip()
    if not q:
        raise ValueError('empty question')
    repaired = []
    if kind in ('condition', 'exception'):
        text = cand['passages'][str(d['passage'])]
        item = dict(id=iid, type=kind, doc=doc, passage=d['passage'], question=q, answer=snap(d['answer'], text, repaired))
        return dict(item, span_repair=repaired), None
    if kind == 'explanation':
        text = cand['passages'][str(d['passage'])]
        cond = d['condition'].strip()
        item = dict(id=iid, type=kind, doc=doc, passage=d['passage'], question=q,
                    conclusion=snap(d['conclusion'], text, repaired), evidence=snap(d['evidence'], text, repaired),
                    condition=snap(cond, text, repaired) if cond else None)
        return dict(item, span_repair=repaired), None
    if kind == 'numeric_change':
        text = cand['passages'][str(d['passage'])]
        span = unique_span(snap(d['span'], text, repaired), text)
        new = edit_number(span)
        if not new or new == span:
            raise ValueError('no editable number')
        item = dict(id=iid, type=kind, doc=doc, passage=d['passage'], edits=[[span, new]], question=q, answer=new)
        return dict(item, span_repair=repaired), None
    if kind == 'combine':
        a = snap(d['value_a'], cand['passages'][str(d['passage_a'])], repaired)
        b = snap(d['value_b'], cand['passages'][str(d['passage_b'])], repaired)
        item = dict(id=iid, type=kind, question=q, parts=[[doc, d['passage_a'], a], [doc, d['passage_b'], b]])
        return dict(item, span_repair=repaired), None
    t = cand['table']
    key = f'tab-{doc}'
    header = t['header_raw'].split()
    if len(header) == t['width'] - 1:
        header = ['구분'] + header
    elif len(header) != t['width']:
        raise ValueError('header does not align with rows')
    rendered = '\n'.join([t['caption'], ' | '.join(header)] + [' | '.join(r) for r in t['rows']])
    if rendered.count(d['answer']) != 1:
        raise ValueError('table answer not unique')
    if t['rows'][d['row']][0] not in q or d['column'] not in q:
        raise ValueError('table question does not name the row and the column')
    table = dict(doc=doc, caption=t['caption'], header=header, rows=t['rows'],
                 note='Rows copied verbatim from one-row-per-line raw text; header line copied (row-label column named 구분 when absent).')
    return dict(id=iid, type='table', table=key, question=q, answer=d['answer']), (key, table)


def check(work, distractor_ids, dev_mod, include_spec, profile='small'):
    """Per-item C-002 builder rules; writes built.jsonl (one built row per passing item) and check.jsonl."""
    from tokenizers import Tokenizer
    import build_task_pilot as B
    cands = {c['cid']: c for c in read_jsonl(work / 'candidates.jsonl')}
    rows = []
    for d in latest(work / 'drafts.jsonl').values():
        if d['error']:
            rows.append(dict(cid=d['cid'], state='draft_error', reason=d['error']))
        elif d['draft'].get('skip'):
            rows.append(dict(cid=d['cid'], state='declined', reason='generator declined'))
        else:
            rows.append(dict(cid=d['cid'], state='pending', draft=d['draft'], generator=d['generator']))
    extra = json.loads(Path(include_spec).read_text()) if include_spec else None
    tok = Tokenizer.from_file(str(B.TOKENIZER))
    dis_ids = [x.strip() for x in Path(distractor_ids).read_text().split() if x.strip()]
    wanted = set(dis_ids) | {cands[r['cid']]['doc'] for r in rows if r['state'] == 'pending'}
    if extra:
        wanted |= {i['doc'] for i in extra['items'] if 'doc' in i} | {p[0] for i in extra['items'] for p in i.get('parts', [])}
        wanted |= {t['doc'] for t in extra['tables'].values()}
    docs = B.load_docs(SOURCE, wanted)
    cut = {k: passages(v['text']) for k, v in docs.items()}

    def split_of(doc):
        return 'valid' if int(hashlib.sha256(('c007-dev' + doc).encode()).hexdigest(), 16) % 10 < dev_mod else 'train'
    distractors = {'train': [], 'valid': []}
    for d in dis_ids:
        if cut.get(d):
            distractors[split_of(d)].append([d, 0])
    spec = dict(domains={}, tables={}, distractors=distractors)
    items = []
    for r in rows:
        if r['state'] != 'pending':
            continue
        try:
            item, table = (S.to_item if profile == 'strong' else to_item)(cands[r['cid']], r['draft'])
            if table:
                spec['tables'][table[0]] = table[1]
            items.append((r, item))
        except (ValueError, KeyError) as exc:
            r.update(state='check_reject', reason=f'draft: {exc}')
    if extra:
        spec['tables'].update(extra['tables'])
        for item in extra['items']:
            if item['id'] in {'comb-c7-003', 'comb-c7-009'}:
                continue   # unrelated facts combined (C-007 review); excluded, not re-verified
            r = dict(cid=f"pilot60:{item['id']}", state='pending', generator='assistant (C-007 pilot)')
            rows.append(r)
            items.append((r, item))
    built = []
    for r, item in items:
        kind = item['type']
        try:
            targets = []
            if kind == 'table':
                t = spec['tables'][item['table']]
                targets.append((t['doc'], B.table_text(t, docs[t['doc']]['text']), None))
            elif kind == 'combine':
                if item['parts'][0][1] == item['parts'][1][1]:
                    raise ValueError('combine parts come from one passage')
                for doc, index, _ in item['parts']:
                    targets.append((doc, cut[doc][index], None))
            else:
                original = cut[item['doc']][item['passage']]
                text = original
                for old, new in item.get('edits', []):
                    B.require(old, original, 'Edited original')
                    text = text.replace(old, new)
                targets.append((item['doc'], text, original if item.get('edits') else None))
                if profile == 'strong' and item.get('edits'):
                    # The counterfactual must not contradict the rest of the document (title, summary lines, other passages,
                    # tables): an original value found outside the edited passage makes the edit unsafe, so the item is skipped.
                    outside = S.outside_text(original, docs[item['doc']].get('title', ''), docs[item['doc']]['text'])
                    found = [k for k in S.changed_values(item['edits']) if S.value_in(k, outside)]
                    if found:
                        raise ValueError(f'unsafe counterfactual: original value {found} also appears elsewhere in the document')
            form_check(item, targets[0][1], markers=profile == 'small')
            split = split_of(targets[0][0])
            pool = [(d, cut[d][i]) for d, i in distractors[split] if d not in {t[0] for t in targets}]
            random.Random(item['id']).shuffle(pool)
            row = None
            for distractor in [(d, t, None) for d, t in pool[:6]] + ([None] if kind == 'combine' else []):
                cand_row = B.assemble(item, kind, targets, distractor, split, docs, spec, random.Random(item['id']))
                m = B.measure(tok, cand_row)
                if m['fits_training'] and m['gold_fits_generation']:
                    row = dict(cand_row, measure=m)
                    break
            if row is None:
                raise ValueError('does not fit 1024 tokens with any distractor')
            r.update(state='checked', item=item)
            built.append(dict(cid=r['cid'], type=kind, item=item, row=row,
                              doc_title=docs[targets[0][0]].get('title', '')))
        except (ValueError, KeyError, IndexError) as exc:
            r.update(state='check_reject', reason=str(exc)[:200])
    (work / 'spec-tables.json').write_text(json.dumps(spec, ensure_ascii=False, indent=1))
    (work / 'built.jsonl').write_text(''.join(json.dumps(b, ensure_ascii=False) + '\n' for b in built))
    (work / 'check.jsonl').write_text(''.join(json.dumps({k: v for k, v in r.items() if k != 'draft'}, ensure_ascii=False) + '\n' for r in rows))
    print(json.dumps(dict(Counter(r['state'] for r in rows)), ensure_ascii=False))


def candidate_text(b):
    item, kind = b['item'], b['type']
    if kind == 'explanation':
        return f"{item['conclusion']} (적용 조건: {item['condition'] or '없음'})"
    if kind == 'combine':
        return ', '.join(p[2] for p in item['parts'])
    return item['answer']


def verify(work, url, workers, profile='small'):
    out = work / 'verified.jsonl'
    done = settled(out)
    checker = model_id(url)
    jobs = [b for b in read_jsonl(work / 'built.jsonl') if b['cid'] not in done]

    def run_strong(b):
        try:
            return dict(cid=b['cid'], type=b['type'], verifier=checker, **S.verify_one(url, b))
        except (ValueError, OSError, KeyError) as exc:
            return dict(cid=b['cid'], type=b['type'], verifier=checker, state='verify_error',
                        error=f'{type(exc).__name__}: {exc}', transient=isinstance(exc, TRANSIENT))

    def run(b):
        body, question = b['row']['prompt'].rsplit('\n질문: ', 1)
        context = body[body.find('[D1]'):]
        cand = candidate_text(b)
        calls = {}
        try:
            independent, calls['answer'] = complete(url, ANSWER_PROMPT.format(context=context, question=question), 200)
            raw, calls['judge'] = complete(url, JUDGE_PROMPT.format(context=context, question=question, candidate=cand,
                                                                   independent=independent), 160, JUDGE_SCHEMA)
            judge = json.loads(raw)
        except (ValueError, OSError) as exc:
            return dict(cid=b['cid'], type=b['type'], verifier=checker, state='verify_error', calls=calls,
                        error=f'{type(exc).__name__}: {exc}', transient=isinstance(exc, TRANSIENT))
        # The judge's own 'agrees_with_b' proved unreliable in the pilot (e.g. '36.5%' vs '36.5% 감소' -> no); it is
        # recorded, and the deterministic content agreement below is the agreement gate instead.
        gates = [k for k in JUDGE_KEYS if k != 'agrees_with_b']
        failed = [k for k in gates if judge.get(k) != GOOD[k] and judge.get(k) != 'unsure']
        unsure = [k for k in gates if judge.get(k) == 'unsure']
        if calls['answer']['stop_type'] == 'limit' or calls['answer']['truncated']:
            unsure.append('independent_answer_truncated')
        content = b['item']['conclusion'] if b['type'] == 'explanation' else cand
        agreement = agrees(content, independent)
        if not failed and not agreement:
            unsure.append('no_content_agreement')
        state = 'semantic_reject' if failed else 'quarantine' if unsure else 'accepted'
        a, i = norm(cand), norm(independent)
        return dict(cid=b['cid'], type=b['type'], verifier=checker, candidate=cand, independent=independent, judge=judge,
                    string_agree=bool(a) and (a in i or i in a), content_agree=agreement, failed=failed, unsure=unsure,
                    state=state, calls=calls)

    with ThreadPoolExecutor(workers) as pool:
        for n, row in enumerate(pool.map(run_strong if profile == 'strong' else run, jobs), 1):
            append_jsonl(out, [row])
            if n % 50 == 0:
                print(f'verified {n}/{len(jobs)}', flush=True)


def status(work):
    """One final state per candidate (and per carried-over pilot item)."""
    final = {r['cid']: r for r in read_jsonl(work / 'check.jsonl')}
    for v in read_jsonl(work / 'verified.jsonl') if (work / 'verified.jsonl').exists() else []:
        final[v['cid']] = dict(final[v['cid']], state=v['state'], failed=v.get('failed'), unsure=v.get('unsure'),
                               string_agree=v.get('string_agree'))
    for r in final.values():
        if r['state'] == 'checked':
            r['state'] = 'unprocessed'
    (work / 'status.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in final.values()))
    table = defaultdict(Counter)
    for r in final.values():
        table[r['cid'].split(':')[0]][r['state']] += 1
    print(json.dumps({k: dict(v) for k, v in sorted(table.items())}, ensure_ascii=False))
    return final


def export(work, out_spec, purpose):
    final = status(work)
    built = {b['cid']: b for b in read_jsonl(work / 'built.jsonl')}
    spec = json.loads((work / 'spec-tables.json').read_text())
    items = [built[c]['item'] for c, r in final.items() if r['state'] == 'accepted']
    tables = {i['table']: spec['tables'][i['table']] for i in items if i['type'] == 'table'}
    docs = {i['doc'] for i in items if 'doc' in i} | {p[0] for i in items for p in i.get('parts', [])} | {t['doc'] for t in tables.values()}
    dev = sorted(d for d in docs if int(hashlib.sha256(('c007-dev' + d).encode()).hexdigest(), 16) % 10 == 0)
    out = dict(purpose=purpose, source='data/korea-kr-briefing-source/docs.jsonl', dev_docs=dev, domains={},
               dev_rule="Document-level split fixed before generation: sha256('c007-dev'+doc_id) mod 10 == 0 -> valid.",
               distractors=spec['distractors'], tables=tables, items=items)
    Path(out_spec).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(dict(items=len(items), types=dict(Counter(i['type'] for i in items)), documents=len(docs)), ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=['select', 'draft', 'check', 'verify', 'status', 'export'])
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--url', default='http://127.0.0.1:18473')
    p.add_argument('--profile', choices=['small', 'strong'], default='small',
                   help='small: C-008 Kanana/Qwen-1.7B numbered-span path; strong: C-009 >= 32B generator and verifier')
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--per-type', type=int, default=300)
    p.add_argument('--seed', default='c008')
    p.add_argument('--exclude', type=Path, action='append', default=[])
    p.add_argument('--only', type=Path, help='restrict selection to these documents (id list / jsonl / spec)')
    p.add_argument('--distractors', type=Path)
    p.add_argument('--dev-mod', type=int, default=1)
    p.add_argument('--include-spec', type=Path)
    p.add_argument('--out-spec', type=Path)
    p.add_argument('--purpose', default='C-008 six-type training candidates; model-drafted, independently model-verified, not human-reviewed.')
    a = p.parse_args()
    if a.stage == 'select':
        select(a.work, a.per_type, a.seed, a.exclude, a.profile, a.only)
    elif a.stage == 'draft':
        draft(a.work, a.url.rstrip('/'), a.workers, a.profile)
    elif a.stage == 'check':
        check(a.work, a.distractors, a.dev_mod, a.include_spec, a.profile)
    elif a.stage == 'verify':
        verify(a.work, a.url.rstrip('/'), a.workers, a.profile)
    elif a.stage == 'status':
        status(a.work)
    else:
        export(a.work, a.out_spec, a.purpose)


if __name__ == '__main__':
    main()
