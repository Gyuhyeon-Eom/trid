"""Pilot hand-off set for A's specialisation experiment (C-017; inputs and freeze rule added in C-018), CPU only.

Selects, from the given batches:
  - official interpretations with a content-matched AI review rated high/medium work relevance, no review defect and no
    'abolition or rename suspected' flag. Relevance-unrated and low-relevance cases are left out;
  - statute_application, table_lookup and code_unit_lookup cases with a content-matched AI review;
  - plain explanations whose source case is selected; they share the source case's split group.
Cases whose document (full id), 안건번호 or question prefix appears in evaluation-side data are dropped.
Every selected case must carry a review record whose content_sha256 equals the recomputed hash of question+target, must
not be held, and must not come from a reserve-overlap or B 'valid'/'test' document; otherwise the build stops.
Splits are per document group (sha256 of the document id). With --frozen, every case of the frozen version must reappear
unchanged and every frozen group keeps its split (a frozen pilot_dev group never moves to train).

No case is labelled current law: official interpretations are 'as of the interpretation date, current law unverified'.
AI review only, one reviewer, no independent or human verification: not production approval.

usage: build_pilot_manifest.py --out DIR [--main F --excluded F --plain F --table F --public F --statute F]
                               [--reviews F ...] [--frozen PILOT_JSONL] [--official-ids JSON_LIST]

--official-ids (added in C-019) limits official interpretations to the listed ids (cases of the frozen version are always
kept), so long verbatim official answers do not grow just because more cases were reviewed.
"""
import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
C016 = ROOT / 'reports/coordination/C-016'
DEFAULTS = dict(main=C016 / 'batch-v4/candidates.jsonl', excluded=C016 / 'batch-v4/excluded.jsonl',
                plain=C016 / 'batch-v4/plain-candidates.jsonl', table=C016 / 'table-batch/table-candidates.jsonl',
                public=C016 / 'public-data-batch/public-data-candidates.jsonl', statute=None)
DEFAULT_REVIEWS = [ROOT / 'reports/coordination/C-017/review-c017-ai.jsonl']
EVAL_SIDE = ['data/benchmark-reserve-001', 'data/dialogue-eval-dev.jsonl', 'data/dialogue-eval-final.jsonl',
             'data/institution-qa-frozen-01', 'data/korean-eval-v1', 'data/korean-rag-eval-v1',
             'data/korean-rag-eval-multihop-v1-source', 'data/korean-rag-eval-v1-source',
             'reports/coordination/C-010/records', 'reports/coordination/C-010/dev', 'reports/coordination/C-010/dev2']
DEV_SHARE = 0.15
KEEP = ('reviewer_id', 'reviewer_type', 'verdict', 'note', 'scope', 'work_relevance', 'outdated_flag')


def load(path):
    return [json.loads(l) for l in Path(path).open()] if path and Path(path).exists() else []


def eval_side_text():
    chunks = []
    for rel in EVAL_SIDE:
        p = ROOT / rel
        files = [p] if p.is_file() else sorted(x for x in p.rglob('*') if x.is_file() and x.suffix in ('.jsonl', '.json', '.md', '.txt'))
        chunks += [f.read_text(errors='ignore') for f in files]
    return '\n'.join(chunks)


def relevance(case):
    return next((r.get('work_relevance') for r in reversed(case['review']['records']) if r.get('work_relevance')), None)


def reviewed_ok(case):
    return case['review']['status'] == 'ai_reviewed' and all(r['verdict'] == 'correct' for r in case['review']['records'])


def exposed(case, eval_text):
    doc = case['doc']
    if doc.startswith('data-go-kr:'):          # a bare dataset number also occurs inside hashes: need the title too
        return (re.search(rf"(?<!\d){re.escape(doc.split(':')[1])}(?!\d)", eval_text) is not None
                and case['title'] in eval_text)
    return (doc in eval_text or bool(case.get('agenda') and case['agenda'] in eval_text)
            or case['question'][:40] in eval_text)


def time_label(case):
    if case['kind'] in ('official_interpretation', 'plain_explanation'):
        return f"as_of_interpretation_date={case['date']}; current_law_unverified"
    if case['kind'] == 'pack_behavior':
        return f"statute text in force from {case.get('effective')}; later amendments unverified"
    return f"snapshot {case.get('date') or case.get('currency')}; current value unverified"


def apply_reviews(cases, records):
    """Attach review records whose content_sha256 matches the case as built; an unreviewed case becomes ai_reviewed."""
    by_id = defaultdict(list)
    for r in records:
        by_id[r['id']].append(r)
    for c in cases:
        matching = [r for r in by_id.get(c['id'], []) if r['content_sha256'] == c['content_sha256']]
        if not matching:
            continue
        if c['review']['status'] != 'ai_reviewed':
            c['review'] = dict(status='ai_reviewed', records=[])
        c['review']['records'] += [{k: r[k] for k in KEEP if k in r} for r in matching]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    for k, v in DEFAULTS.items():
        ap.add_argument(f'--{k}', type=Path, action='append' if k == 'table' else None,
                        default=None if k == 'table' else v)
    ap.add_argument('--reviews', type=Path, action='append', default=[])
    ap.add_argument('--frozen', type=Path, help='pilot.jsonl of a frozen earlier version')
    ap.add_argument('--official-ids', type=Path, help='JSON list of official interpretation ids allowed in this version')
    a = ap.parse_args()
    tables = a.table or [DEFAULTS['table']]
    data = dict(main=load(a.main), excluded=load(a.excluded), plain=load(a.plain),
                table=[c for t in tables for c in load(t)], public=load(a.public), statute=load(a.statute))
    review_files = DEFAULT_REVIEWS + [p.resolve() for p in a.reviews]
    apply_reviews([c for k in ('main', 'plain', 'table', 'public', 'statute') for c in data[k]],
                  [r for p in review_files for r in load(p)])
    held_docs = {c['doc']: c['training_hold'] for c in data['excluded']}
    eval_text = eval_side_text()
    frozen = load(a.frozen)
    allowed = (set(json.loads(a.official_ids.read_text())) | {r['id'] for r in frozen}) if a.official_ids else None
    selected, dropped = [], Counter()

    for c in data['main']:
        if c['kind'] != 'official_interpretation':
            (selected.append(c) if reviewed_ok(c) else dropped.update([f"{c['kind']}: not AI-reviewed"]))
        elif not reviewed_ok(c):
            dropped['official: not AI-reviewed'] += 1
        elif any(r.get('outdated_flag') for r in c['review']['records']):
            dropped['official: abolition or rename suspected'] += 1
        elif relevance(c) is None:
            dropped['official: work relevance not rated'] += 1
        elif relevance(c) == 'low':
            dropped['official: low work relevance'] += 1
        elif allowed is not None and c['id'] not in allowed:
            dropped['official: not in --official-ids selection'] += 1
        else:
            selected.append(c)
    for key in ('table', 'public', 'statute'):
        for c in data[key]:
            (selected.append(c) if reviewed_ok(c) and not c.get('training_hold') else dropped.update([f'{key}: not AI-reviewed or held']))
    chosen = {c['id'] for c in selected}
    for c in data['plain']:
        if c['verbatim_case'] in chosen and reviewed_ok(c):
            selected.append(c)
        else:
            dropped['plain: source case not selected or plain not reviewed'] += 1

    kept, exposed_ids = [], []
    for c in selected:
        if exposed(c, eval_text):
            dropped[f"{c['kind']}: document or question appears in evaluation-side data"] += 1
            exposed_ids.append(c['id'])
        else:
            kept.append(c)
    ids = {c['id'] for c in kept}       # a plain explanation goes only with its source case
    kept = [c for c in kept if c['kind'] != 'plain_explanation' or c['verbatim_case'] in ids]

    problems, seen = [], {}
    for c in kept:
        sha = hashlib.sha256((c['question'] + '\n' + c['target']).encode()).hexdigest()
        if sha != c['content_sha256'] or not c['review']['records']:
            problems.append(f"{c['id']}: content hash or review record mismatch")
        if c.get('training_hold'):
            problems.append(f"{c['id']}: held ({c['training_hold']})")
        if str(held_docs.get(c['doc'], '')).startswith(('reserve-overlap', 'B v3 split')) or c.get('split_hint') in ('valid', 'test'):
            problems.append(f"{c['id']}: reserve-overlap or eval-side split document")
        if sha in seen:
            problems.append(f"{c['id']}: same question and target as {seen[sha]}")
        seen[sha] = c['id']

    frozen_split = {r['split_group']: r['split'] for r in frozen}
    by_id = {c['id']: c for c in kept}
    for r in frozen:                     # a frozen version is carried over unchanged
        c = by_id.get(r['id'])
        if c is None or c['content_sha256'] != r['content_sha256']:
            problems.append(f"{r['id']}: frozen case missing or changed")
    if problems:
        raise SystemExit('pilot checks failed:\n' + '\n'.join(problems))

    groups = defaultdict(list)
    for c in kept:
        groups[c['doc']].append(c)
    rows = []
    frozen_ids = {r['id'] for r in frozen}
    for g, cases in sorted(groups.items()):
        split = frozen_split.get(g) or ('pilot_dev' if int(hashlib.sha256(g.encode()).hexdigest(), 16) % 1000 < DEV_SHARE * 1000 else 'train')
        for c in cases:
            row = dict(id=c['id'], split=split, split_group=g, task_type=c['task_type'], behavior=c['behavior'],
                       source_answer_present=c['source_answer_present'], question=c['question'], context=c['context'],
                       target=c['target'], doc=c['doc'], institution=c.get('institution'), time_label=time_label(c),
                       synthetic_parts=c.get('synthetic_parts'), work_relevance=relevance(c) or 'n/a (behaviour case)',
                       review=dict(status=c['review']['status'], reviewer_type='ai', independent=False,
                                   records=len(c['review']['records'])),
                       content_sha256=c['content_sha256'], verbatim_case=c.get('verbatim_case'))
            if a.frozen:
                row['in_frozen_version'] = c['id'] in frozen_ids
            rows.append(row)
    a.out.mkdir(parents=True, exist_ok=True)
    body = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
    (a.out / 'pilot.jsonl').write_text(body)
    per_group = Counter(r['split_group'] for r in rows)
    inputs = [a.main, a.excluded, a.plain, *tables, a.public, a.statute, *review_files, a.frozen, a.official_ids]
    meta = dict(
        purpose='pilot hand-off for A specialisation experiment; AI review only (one reviewer; statute/table/public-data/'
                'plain cases are C self-review); independent verification 0; not production approval',
        sha256=hashlib.sha256(body.encode()).hexdigest(), cases=len(rows),
        by_split=dict(Counter(r['split'] for r in rows)),
        by_task_type={s: dict(Counter(r['task_type'] for r in rows if r['split'] == s)) for s in ('train', 'pilot_dev')},
        by_behavior=dict(Counter(r['behavior'] for r in rows)),
        characters={t: sum(len(r['question']) + sum(map(len, r['context'])) + len(r['target']) for r in rows if r['task_type'] == t)
                    for t in sorted({r['task_type'] for r in rows})},
        tokens='not counted: to be agreed after A fixes the tokenizer/template',
        document_groups=len(per_group),
        repeated_documents={g: n for g, n in per_group.most_common() if n > 1},
        frozen_version=dict(file=str(a.frozen.resolve().relative_to(ROOT)) if a.frozen else None, cases=len(frozen),
                            carried_over=sum(r.get('in_frozen_version', False) for r in rows)),
        dropped=dict(dropped), dropped_for_evaluation_exposure=exposed_ids,
        checks=['content_sha256 recomputed and matched; each case has a content-matched AI review record',
                'no held case; no reserve-overlap or B valid/test document',
                'evaluation-side data searched for full document id, 안건번호 and question prefix: ' + ', '.join(EVAL_SIDE),
                'no two cases with the same question and target; split assigned per document group',
                'frozen version (if given): every case carried over unchanged and every frozen group keeps its split'],
        time_labels='official interpretations: as of the interpretation date, current law unverified (no case is labelled current)',
        inputs={str(Path(p).resolve().relative_to(ROOT)): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in inputs if p and Path(p).exists()})
    (a.out / 'manifest.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps({k: meta[k] for k in ('cases', 'by_split', 'by_task_type', 'document_groups', 'frozen_version', 'dropped',
                                           'dropped_for_evaluation_exposure')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
