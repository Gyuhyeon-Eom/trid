"""C-010: structured evidence records for the six task types, taken from structure the source text states itself.

Records are written before any question is generated; the generator later only phrases a question for the record.
Every span is verbatim. Each record carries the passages (verbatim excerpts of one document: title line, article
heading, paragraph) that a built item will show, and the spans' role in them:

  condition   statute/administrative-rule paragraph "…경우에는 …" (no proviso, no 다음 각 호 list):
              condition span (up to 경우) -> effect span
  exception   paragraph with a rule followed by a proviso "다만, …": rule span, exception span (proviso without 다만,)
  explanation statutory interpretation (법령해석례): 【회답】 conclusion and the 【이유】 paragraph that immediately
              precedes the "따라서 …" paragraph (the stated ground for the conclusion)
  combine     paragraph B that cites another article of the same document ("제N조(제M항)에 따른/따라"):
              cited paragraph A + citing paragraph B (an explicit cross-reference relation)
  table       korea.kr raw table block with its header lines (from C-009 main candidates)
  numeric_change korea.kr passage with a quantity that appears only in that passage (C-009 main candidates)

Excluded documents: base EXCLUDE, earlier C-00x documents (--exclude), the multi-hop evaluation documents,
the A-003 reserve partial-review law rows, quarantine list.

usage: build_task_sources.py --out DIR [--per-type N] [--exclude PATH ...] [--korea-candidates PATH]
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from build_task_candidates import excluded_ids  # noqa: E402

LAW = ROOT / 'data/law-ko-source/docs.jsonl'
EVAL_DOCS = ROOT / 'data/korean-rag-eval-multihop-v1-source/documents.jsonl'
RESERVE_PARTIAL = ROOT / 'reports/coordination/A-003/reserve-148-vs-mix-v3.json'
LAW_LICENSE = 'KR-Copyright-Act-Art7 (law.go.kr); 출처: 법제처 국가법령정보센터'
ARTICLE = re.compile(r'^제(\d+)조(?:의(\d+))?\s*\(([^)]*)\)\s*(.*)$')
CIRCLED = '①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳'
CITE = re.compile(r'제(\d+)조(?:의(\d+))?(?:제(\d+)항)?에\s?따(?:른|라)')
SENT_END = re.compile(r'다\.(?=\s|$)')
LIST_REF = re.compile(r'다음\s*각\s*호|다음\s*사항|다음과\s*같|제\d+호')
OTHER_LAW = ('법', '령', '칙', '」', '규정', '동법', '같은 법', '영', '고시', '훈령', '예규')


def bigram_overlap(a, b):
    grams = lambda s: {s[i:i + 2] for i in range(len(s) - 1)}
    ga, gb = grams(re.sub(r'\s+', '', a)), grams(re.sub(r'\s+', '', b))
    return len(ga & gb) / max(1, min(len(ga), len(gb)))

def articles(text):
    """[(label, heading_line, [paragraph lines])]; paragraph = one ①.. line (or the heading body), 호 lines excluded."""
    out, cur = [], None
    for line in text.split('\n'):
        s = line.strip()
        if s.startswith('부칙'):
            break
        m = ARTICLE.match(s)
        if m:
            cur = (f"제{m.group(1)}조" + (f"의{m.group(2)}" if m.group(2) else ''), s, [])
            out.append(cur)
            if m.group(4):
                cur[2].append(m.group(4))
        elif cur and s and s[0] in CIRCLED:
            cur[2].append(s)
    return out


def para_no(p):
    return CIRCLED.index(p[0]) + 1 if p and p[0] in CIRCLED else 1


def excerpt(title, heading, para):
    """Passage shown to the model: verbatim title line, article heading, paragraph (the heading may already hold it)."""
    return title + '\n' + (heading if heading.endswith(para) else heading + '\n' + para)


def law_records(doc):
    title = doc['text'].split('\n', 1)[0].strip()
    arts = articles(doc['text'])
    index = {a[0]: a for a in arts}
    recs = defaultdict(list)
    for label, heading, paras in arts:
        for para in paras:
            body = para[1:].strip() if para[0] in CIRCLED else para
            if LIST_REF.search(body) or len(body) > 420:
                continue   # the rule or condition would depend on a list that the passage does not show
            if '다만, ' in body:
                i = body.index('다만, ')
                rule, rest = body[:i].strip(), body[i + 4:]
                m = SENT_END.search(rest)
                exc = rest[:m.end()] if m else ''
                if len(rule) >= 30 and len(exc) >= 15 and rule.endswith('다.') and body.count('다만') == 1:
                    recs['exception'].append(dict(passages=[excerpt(title, heading, para)], record=dict(
                        article=label, paragraph=para_no(para), rule=rule, exception=exc)))
            elif '다만' not in body and body.count('경우에는') == 1 and body.endswith('다.'):
                i = body.index('경우에는')
                cond, effect = body[:i + 2].strip(), body[i + 4:].strip()
                if '하되' in cond:   # "A는 …하되, B인 경우에는 C": the condition is the part after 하되
                    cond = re.split(r'하되,?\s*', cond)[-1].strip()
                if len(cond) >= 15 and len(effect) >= 10:
                    recs['condition'].append(dict(passages=[excerpt(title, heading, para)], record=dict(
                        article=label, paragraph=para_no(para), condition=cond, effect=effect)))
            for m in CITE.finditer(body):
                # "영 제16조 및 제17조에 따라": strip the whole citation chain before looking for the act it belongs to
                before = re.sub(r'(제\d+조(의\d+)?(제\d+항)?\s*(및|또는|,|ㆍ|·|과|와)\s*)+$', '', body[:m.start()]).rstrip()
                if before.endswith(OTHER_LAW):
                    continue   # 법 제N조 / 「…」 제N조 / 동법 제N조: an article of another act, not of this document
                target = f"제{m.group(1)}조" + (f"의{m.group(2)}" if m.group(2) else '')
                if target == label or target not in index:
                    continue
                t_label, t_heading, t_paras = index[target]
                n = int(m.group(3)) if m.group(3) else 1
                cited = next((p for p in t_paras if para_no(p) == n), None)
                if not cited or LIST_REF.search(cited) or len(cited) > 420:
                    continue
                recs['combine'].append(dict(passages=[excerpt(title, t_heading, cited), excerpt(title, heading, para)], record=dict(
                    cited_article=f'{target}' + (f'제{n}항' if m.group(3) else ''), cited=cited[1:].strip() if cited[0] in CIRCLED else cited,
                    citing_article=label, citing=body, reference=m.group(0))))
                break
    return recs


def expc_record(doc):
    """법령해석례: conclusion = 【회답】, ground = the 【이유】 paragraph right before the '따라서' paragraph."""
    text = doc['text']
    if not all(k in text for k in ('【질의요지】', '【회답】', '【이유】')):
        return None
    answer = text.split('【회답】', 1)[1].split('【이유】', 1)[0].strip()
    reason = text.split('【이유】', 1)[1]
    paras = [p.strip() for p in re.split(r'\n(?=○)', reason) if p.strip()]
    k = next((i for i, p in enumerate(paras) if p.lstrip('○ ').startswith('따라서')), None)
    if not answer or k is None or k == 0 or len(answer) > 450 or len(paras[k - 1]) > 600 or re.search(r'질의\s*[가나다1-9]', answer):
        return None   # no stated ground, too long, or a multi-question answer (가. … 나. …)
    ground = paras[k - 1]
    if bigram_overlap(answer, ground) > .6:
        return None   # the 'ground' only restates the conclusion
    title = text.split('\n', 1)[0].strip()
    passage = f"{title}\n【회답】\n{answer}\n【이유】\n{ground}"
    return dict(passages=[passage], record=dict(conclusion=answer, ground=ground.lstrip('○ ').strip()))


def verify_record(doc_text, rec):
    """Every passage line is verbatim in the document; every span is verbatim in a passage."""
    for p in rec['passages']:
        for line in p.split('\n'):
            if line not in ('【회답】', '【이유】') and line.strip() and line.strip() not in doc_text:
                raise ValueError(f'excerpt line not in document: {line[:40]!r}')
    joined = '\n'.join(rec['passages'])
    for k, v in rec['record'].items():
        if isinstance(v, str) and k not in ('article', 'cited_article', 'citing_article') and v not in joined:
            raise ValueError(f'span {k} not in passages: {v[:40]!r}')


def write_candidates(records_path, out, orders, review=None):
    """Records -> candidates.jsonl for build_task_candidates.py --profile strong. Records judged not viable in the
    source review are excluded before any generation."""
    lo, hi = (int(x) for x in orders.split('-'))
    dropped = {r['rid'] for r in map(json.loads, Path(review).open()) if not r['viable']} if review else set()
    rows, skipped = [], Counter()
    for r in map(json.loads, Path(records_path).open()):
        if not lo <= r['order'] <= hi:
            continue
        if r['rid'] in dropped:
            skipped[r['type']] += 1
            continue
        if 'korea_candidate' in r:
            rows.append(dict(r['korea_candidate'], corpus=r['corpus'], license=r['license'], order=r['order']))
        else:
            rows.append(dict(cid=r['rid'], type=r['type'], doc=r['doc'], title=r['title'], source_url=r['source_url'],
                             corpus=r['corpus'], kind=r['kind'], license=r['license'], order=r['order'],
                             passages={str(i): p for i, p in enumerate(r['passages'])}, record=r['record']))
    out.mkdir(parents=True, exist_ok=True)
    (out / 'candidates.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in rows))
    print(json.dumps(dict(candidates=dict(Counter(c['type'] for c in rows)), excluded_by_review=dict(skipped)), ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--per-type', type=int, default=400)
    p.add_argument('--seed', default='c010')
    p.add_argument('--exclude', type=Path, action='append', default=[])
    p.add_argument('--korea-candidates', type=Path, help='C-009 main candidates (strong profile) for table / numeric_change')
    p.add_argument('--korea-skip', type=int, default=10, help='first N per type already consumed in C-009 development')
    p.add_argument('--candidates', type=Path, help='instead of extracting: write candidates from --out/records.jsonl here')
    p.add_argument('--orders', default='0-9', help='record order range per type for --candidates, e.g. 0-9 (dev) or 10-49')
    p.add_argument('--review', type=Path, help='source review (viable false -> excluded) for --candidates')
    a = p.parse_args()
    if a.candidates:
        return write_candidates(a.out / 'records.jsonl', a.candidates, a.orders, a.review)
    excluded = excluded_ids(a.exclude)
    excluded |= {json.loads(l)['id'] for l in EVAL_DOCS.open()}
    reserve_rows = {r['corpus_id'] for r in json.loads(RESERVE_PARTIAL.read_text())['partial_review']}
    excluded |= reserve_rows
    pools, skipped, kinds = defaultdict(list), Counter(), Counter()
    with LAW.open() as f:
        for line in f:
            kind = line[line.index('"law-go-kr:') + 11:].split(':', 1)[0] if '"law-go-kr:' in line[:40] else ''
            if kind not in ('law', 'admrul', 'expc'):
                continue
            doc = json.loads(line)
            if doc['id'] in excluded:
                skipped['excluded'] += 1
                continue
            kinds[kind] += 1
            key = int(hashlib.sha256((a.seed + doc['id']).encode()).hexdigest(), 16)
            found = {'explanation': [expc_record(doc)]} if kind == 'expc' else law_records(doc)
            for t, recs in found.items():
                for r in recs:
                    if not r:
                        continue
                    try:
                        verify_record(doc['text'], r)
                    except ValueError:
                        skipped[f'{t}_not_verbatim'] += 1
                        continue
                    pools[t].append(dict(r, doc=doc['id'], title=doc['title'], source_url=doc['source'], corpus='law-ko-source',
                                         kind=kind, license=LAW_LICENSE, key=key))
    out, used = [], set()
    for t in ('explanation', 'combine', 'exception', 'condition'):
        by_doc = {}
        for r in sorted(pools[t], key=lambda r: (r['key'], r['record'].get('article', ''))):
            if r['doc'] not in used and r['doc'] not in by_doc:
                by_doc[r['doc']] = r
        picked = list(by_doc.values())[:a.per_type]
        used |= {r['doc'] for r in picked}
        for n, r in enumerate(picked):
            out.append(dict({k: v for k, v in r.items() if k != 'key'}, type=t, rid=f'{t}:{r["doc"]}', order=n))
    if a.korea_candidates:
        seen = Counter()
        for line in a.korea_candidates.open():
            c = json.loads(line)
            if c['type'] not in ('table', 'numeric_change'):
                continue
            seen[c['type']] += 1
            if seen[c['type']] <= a.korea_skip or seen[c['type']] > a.korea_skip + a.per_type:
                continue
            out.append(dict(type=c['type'], rid=c['cid'], doc=c['doc'], title=c['title'], corpus='korea-kr-briefing-source',
                            license='KOGL-Type1', order=seen[c['type']] - a.korea_skip - 1, korea_candidate=c))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'records.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in out))
    meta = dict(seed=a.seed, per_type=a.per_type, pool_sizes={t: len(v) for t, v in pools.items()},
                distinct_docs={t: len({r['doc'] for r in v}) for t, v in pools.items()}, selected=dict(Counter(r['type'] for r in out)),
                law_docs_scanned=dict(kinds), skipped=dict(skipped), excluded_ids=len(excluded), reserve_partial_rows=len(reserve_rows),
                exclude=[str(x) for x in a.exclude], korea_candidates=str(a.korea_candidates) if a.korea_candidates else None)
    (a.out / 'sources.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == '__main__':
    main()
