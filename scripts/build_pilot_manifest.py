"""Pilot hand-off set for A's specialisation experiment (C-017), CPU only.

Reads the C-016 batches (official interpretations + statute cases, plain explanations, tables, public data) and selects:
  - official interpretations with a content-matched AI review rated high/medium work relevance, no review defect and no
    'abolition or rename suspected' flag. Relevance-unrated and low-relevance cases are left out;
  - statute_application, table_lookup and code_unit_lookup cases with a content-matched AI review;
  - plain explanations whose source case is selected; they share the source case's split group.
Cases whose document (full id), 안건번호 or question prefix appears in evaluation-side data are dropped.
Every selected case must carry a review record whose content_sha256 equals the recomputed hash of question+target, must
not be held, and must not come from a reserve-overlap or B 'valid'/'test' document; otherwise the build stops. Splits are
assigned per document group (sha256 of the document id), so all cases on one document land in the same split.

No case is labelled current law: official interpretations are 'as of the interpretation date, current law unverified'.
AI review only, one reviewer, no independent or human verification: not production approval.

usage: build_pilot_manifest.py --out DIR
"""
import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
C016 = ROOT / 'reports/coordination/C-016'
SOURCES = dict(main=C016 / 'batch-v4/candidates.jsonl', excluded=C016 / 'batch-v4/excluded.jsonl',
               plain=C016 / 'batch-v4/plain-candidates.jsonl', table=C016 / 'table-batch/table-candidates.jsonl',
               public=C016 / 'public-data-batch/public-data-candidates.jsonl')
EXTRA_REVIEWS = ROOT / 'reports/coordination/C-017/review-c017-ai.jsonl'
EVAL_SIDE = ['data/benchmark-reserve-001', 'data/dialogue-eval-dev.jsonl', 'data/dialogue-eval-final.jsonl',
             'data/institution-qa-frozen-01', 'data/korean-eval-v1', 'data/korean-rag-eval-v1',
             'data/korean-rag-eval-multihop-v1-source', 'data/korean-rag-eval-v1-source',
             'reports/coordination/C-010/records', 'reports/coordination/C-010/dev', 'reports/coordination/C-010/dev2']
DEV_SHARE = 0.15


def load(path):
    return [json.loads(l) for l in path.open()] if path.exists() else []


def eval_side_text():
    chunks = []
    for rel in EVAL_SIDE:
        p = ROOT / rel
        files = [p] if p.is_file() else sorted(x for x in p.rglob('*') if x.is_file() and x.suffix in ('.jsonl', '.json', '.md', '.txt'))
        chunks += [f.read_text(errors='ignore') for f in files]
    return '\n'.join(chunks)


def relevance(case):
    return next((r.get('work_relevance') for r in case['review']['records'] if r.get('work_relevance')), None)


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    data = {k: load(v) for k, v in SOURCES.items()}
    extra = {}
    for r in load(EXTRA_REVIEWS):
        extra.setdefault(r['id'], []).append(r)
    for case in data['main'] + data['plain']:        # C-017 relevance judgements on already reviewed cases
        for r in extra.get(case['id'], []):
            if r['content_sha256'] == case['content_sha256'] and case['review']['status'] == 'ai_reviewed':
                case['review']['records'].append({k: r[k] for k in ('reviewer_id', 'reviewer_type', 'verdict', 'note', 'scope', 'work_relevance') if k in r})
    held_docs = {c['doc']: c['training_hold'] for c in data['excluded']}
    eval_text = eval_side_text()
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
        else:
            selected.append(c)
    for key in ('table', 'public'):
        for c in data[key]:
            (selected.append(c) if reviewed_ok(c) and not c.get('training_hold') else dropped.update([f'{key}: not usable']))
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
    if problems:
        raise SystemExit('pilot checks failed:\n' + '\n'.join(problems))

    groups = defaultdict(list)
    for c in kept:
        groups[c['doc']].append(c)
    rows = []
    for g, cases in sorted(groups.items()):
        split = 'pilot_dev' if int(hashlib.sha256(g.encode()).hexdigest(), 16) % 1000 < DEV_SHARE * 1000 else 'train'
        for c in cases:
            rows.append(dict(id=c['id'], split=split, split_group=g, task_type=c['task_type'], behavior=c['behavior'],
                             source_answer_present=c['source_answer_present'], question=c['question'], context=c['context'],
                             target=c['target'], doc=c['doc'], institution=c.get('institution'), time_label=time_label(c),
                             synthetic_parts=c.get('synthetic_parts'), work_relevance=relevance(c) or 'n/a (behaviour case)',
                             review=dict(status=c['review']['status'], reviewer_type='ai', independent=False,
                                         records=len(c['review']['records'])),
                             content_sha256=c['content_sha256'], verbatim_case=c.get('verbatim_case')))
    a.out.mkdir(parents=True, exist_ok=True)
    body = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
    (a.out / 'pilot.jsonl').write_text(body)
    per_group = Counter(r['split_group'] for r in rows)
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
        dropped=dict(dropped), dropped_for_evaluation_exposure=exposed_ids,
        checks=['content_sha256 recomputed and matched; each case has a content-matched AI review record',
                'no held case; no reserve-overlap or B valid/test document',
                'evaluation-side data searched for full document id, 안건번호 and question prefix: ' + ', '.join(EVAL_SIDE),
                'no two cases with the same question and target; split assigned per document group'],
        time_labels='official interpretations: as of the interpretation date, current law unverified (no case is labelled current)',
        inputs={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [*SOURCES.values(), EXTRA_REVIEWS] if p.exists()})
    (a.out / 'manifest.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps({k: meta[k] for k in ('cases', 'by_split', 'by_task_type', 'document_groups', 'dropped',
                                           'dropped_for_evaluation_exposure')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
