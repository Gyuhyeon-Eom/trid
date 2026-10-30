"""Internal-work training-case candidates for the specialised small model (TRID-BASE-CUTOVER-001), CPU only.

Two sources, both from material B verified (B-012/B-013); no teacher-model generation, no institution data:

  official   법령해석례 rows of data/internal-rules/v3/docs.jsonl. Question = the whole 【질의요지】 (whitespace
             normalised; '< 질의 배경 >' dropped), so its premises, parties, dates and definitions stay with it.
             Context and target carry the whole 【회답】 and the whole 【이유】 (up to the 법령정비 권고 / 관계 법령
             appendix), so the conclusion is justified by the context itself. Rows whose question or reasons are too
             long to include whole are not excerpted; they are left out. The only added text is a fixed line with the
             interpretation date (meta.date_type 해석일자, checked against the source XML by B-013) and the source.
  pack       behaviour cases written by C (AI, not human-reviewed) on the B-013 evidence-pack articles, in
             reports/coordination/C-011/pack-cases.json: answer with condition / exception / effective
             date / source, ask back when facts are missing, say a live value must be looked up, hold when the
             basis is not secured. Every quoted span is checked to be verbatim in the pack article.

Exclusions: rows whose document is in a B-013 reserve-overlap list (full match or partial review) are written
with training_hold; C-010 development documents are excluded so earlier development measurements stay separate.
Nothing here is an evaluation item.

usage: build_internal_cases.py --out DIR
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / 'data/internal-rules/v3/docs.jsonl'
PACKS = ROOT / 'reports/coordination/B-013/evidence-packs.json'
PACK_CASES = ROOT / 'reports/coordination/C-011/pack-cases.json'
OVERLAPS = [ROOT / 'reports/coordination/B-013/overlap-law-admrul-expc-all-148.json',
            ROOT / 'reports/coordination/B-013/overlap-pack-documents-148.json']
C010_DEV = ROOT / 'reports/coordination/C-010/dev60-source-review.jsonl'
REVIEW_HOLD = ROOT / 'reports/coordination/C-011/review-hold.json'   # id -> reason, from AI review; held until a person checks
PLAIN_CASES = ROOT / 'reports/coordination/C-012/plain-cases.json'
HOLD_REASON = 'reserve-overlap (B-013, unresolved): training input on hold; never an evaluation item'
# What a case trains (C-013). Official cases carry the official 회답 and 이유 in the context, so the target is found and
# restated from a supplied answer: retrieval-grounded extraction/explanation, not evidence of independent legal reasoning.
TASK_OFFICIAL = 'official_answer_extraction'      # context holds the official answer; target restates it with its reasons
TASK_PLAIN = 'official_answer_plain_explanation'   # same context; target is a plain-language restatement (synthetic)
TASK_PACK = 'statute_application'                 # context is the article text only; C wrote the conclusion from it


def overlap_ids():
    ids = set()
    for p in OVERLAPS:
        r = json.loads(p.read_text())
        ids |= {m['corpus_id'] for m in r.get('matches', [])} | {m['corpus_id'] for m in r.get('partial_review', [])}
    return ids


MAX_QUESTION, MAX_ANSWER, MAX_REASONS = 800, 600, 2500


def sections(text):
    question = text.split('【질의요지】', 1)[1].split('【회답】', 1)[0].split('< 질의 배경 >', 1)[0]
    answer = text.split('【회답】', 1)[1].split('【이유】', 1)[0]
    reasons = re.split(r'\n※\s*법령정비|\n<\s*관계\s*법령\s*>|\n【관계법령】', text.split('【이유】', 1)[1])[0]
    return re.sub(r'\s+', ' ', question).strip(), answer.strip(), reasons.strip()


def official_case(row):
    text = row['text']
    if not all(k in text for k in ('【질의요지】', '【회답】', '【이유】')):
        return None, 'no sections'
    question, answer, reasons = sections(text)
    if re.search(r'질의\s*[가나다1-9]', answer) or re.search(r'(^|\s)가\.\s.*\s나\.\s', question):
        return None, 'multi-part question or answer'
    if not question or not answer or not reasons:
        return None, 'empty section'
    if len(question) > MAX_QUESTION or len(answer) > MAX_ANSWER or len(reasons) > MAX_REASONS:
        return None, 'question or reasons too long to include whole (not excerpted)'
    if re.search(r'\d\)\d\)', question + answer + reasons):
        return None, 'footnote text leaked into a section (extraction artifact)'
    meta = row.get('meta') or {}
    raw = meta.get('raw_fields') or {}
    date, kind = row.get('date') or '미상', meta.get('date_type') or 'unknown'
    agenda = raw.get('안건번호')
    context = f"{row['title']}\n【회답】\n{answer}\n【이유】\n{reasons}"
    target = (f"결론: {answer}\n근거: {reasons}\n"
              f"시점: 이 해석은 {date}({kind}) 당시 시행되던 법령을 기준으로 한 것이므로, 지금 적용하려면 현행 조문과 이후 개정 여부를 확인해야 합니다.\n"
              f"출처: 법제처 법령해석례{' ' + agenda if agenda else ''} ({row['id']})")
    return dict(kind='official_interpretation', behavior='answer_with_basis', question=question, context=[context],
                task_type=TASK_OFFICIAL, source_answer_present=True,
                target=target, evidence=dict(conclusion=answer, reasons=reasons), date=date, date_field=kind,
                agenda=agenda, title=row['title'],
                authored='question (whole 질의요지), conclusion and reasons verbatim from the official interpretation; '
                         'date/source lines are a fixed template'), None


def pack_cases():
    packs = {(a['id'], a['article']): a for arts in json.loads(PACKS.read_text()).values() for a in arts}
    rows = []
    for c in json.loads(PACK_CASES.read_text())['cases']:
        a = packs[(c['doc'], c['article'])]
        for span in c['evidence']:
            if span not in a['text']:
                raise ValueError(f"{c['id']}: evidence not verbatim in {c['article']}: {span[:40]!r}")
        rows.append(dict(kind='pack_behavior', behavior=c['behavior'], question=c['question'], context=[a['text']],
                         task_type=TASK_PACK, source_answer_present=False,
                         target=c['target'], evidence=dict(spans=c['evidence']), doc=c['doc'], article=c['article'],
                         effective=a['raw_dates'].get('시행일자'), date_fields=a['raw_dates'], source=a['source'],
                         text_sha256=a['text_sha256'], authored='question and explanation written by C (AI assistant); not human-reviewed',
                         id=c['id'], missing=c.get('missing', [])))
    return rows

def plain_cases(batch):
    """Plain-language versions (AI-written, synthetic) of official cases that are in the batch and whose AI review
    matched their current content and found them correct. Every must_keep item (article, number, condition) has to
    appear both in the plain text and in the official question/answer/reasons. Kept in a separate file."""
    by_id = {c['id']: c for c in batch}
    out, rejected = [], {}
    for p in json.loads(PLAIN_CASES.read_text())['cases'] if PLAIN_CASES.exists() else []:
        src = by_id.get(p['source_case'])
        if not src or src['review']['status'] != 'ai_reviewed' or any(r['verdict'] != 'correct' for r in src['review']['records']):
            rejected[p['source_case']] = 'source case not in the batch or not AI-reviewed correct in its current content'
            continue
        official = src['question'] + src['evidence']['conclusion'] + src['evidence']['reasons']
        missing = [k for k in p['must_keep'] if k not in p['target_plain'] or k not in official]
        if missing:
            rejected[p['source_case']] = f'must_keep missing in plain text or official text: {missing}'
            continue
        target = p['target_plain']
        out.append(dict(src, id=f"plain:{src['doc']}", kind='plain_explanation', target=target,
                        task_type=TASK_PLAIN, source_answer_present=True,
                        verbatim_case=src['id'], verbatim_content_sha256=src['content_sha256'], must_keep=p['must_keep'],
                        authored='question and context verbatim (same as verbatim_case); target written by C (AI) in plain language',
                        synthetic_parts='target (AI-written plain-language explanation)',
                        content_sha256=hashlib.sha256((src['question'] + '\n' + target).encode()).hexdigest(),
                        review=dict(status='unreviewed', records=[])))
    return out, rejected



def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--reviews', type=Path, action='append', default=[],
                   help='review records (jsonl: id, content_sha256, reviewer_id, reviewer_type, verdict, note); a record '
                        'applies only when its content_sha256 matches the case as built now')
    a = p.parse_args()
    a.reviews = [r.resolve() for r in a.reviews]
    reviews = {}
    for path in a.reviews:
        for r in map(json.loads, path.open()):
            reviews.setdefault(r['id'], []).append(dict(r, file=str(path.relative_to(ROOT))))
    review_hold = json.loads(REVIEW_HOLD.read_text()) if REVIEW_HOLD.exists() else {}
    held = overlap_ids()
    dev = {json.loads(l)['doc'] for l in C010_DEV.open()}
    out, skipped = [], Counter()
    for line in V3.open():
        row = json.loads(line)
        if row['kind'] != 'legal_interpretation':
            continue
        if row['id'] in dev:
            skipped['C-010 development document'] += 1
            continue
        case, why = official_case(row)
        if not case:
            skipped[why] += 1
            continue
        case.update(id=f"official:{row['id']}", doc=row['id'], source=row['source'], license=row['license'],
                    text_sha256=row['sha256'], split_hint=row.get('split'))
        out.append(case)
    out += pack_cases()
    def apply_review(c):
        """Attach review records whose content_sha256 matches the case as built now; a non-correct verdict holds it."""
        matching = [r for r in reviews.get(c['id'], []) if r['content_sha256'] == c['content_sha256']]
        keep = ('reviewer_id', 'reviewer_type', 'verdict', 'note', 'file', 'scope', 'work_relevance', 'dimensions')
        c['review'] = (dict(status='ai_reviewed', records=[{k: r[k] for k in keep if k in r} for r in matching])
                       if matching else dict(status='unreviewed', records=[]))
        if matching and any(r['verdict'] != 'correct' for r in matching) and not c.get('training_hold'):
            c['training_hold'] = 'review: AI review found a defect (see review.records)'

    for c in out:
        c['content_sha256'] = hashlib.sha256((c['question'] + '\n' + c['target']).encode()).hexdigest()
        c['training_hold'] = HOLD_REASON if c['doc'] in held else (f"review hold: {review_hold[c['id']]}" if c['id'] in review_hold else None)
        if not c['training_hold'] and c.get('split_hint') in ('valid', 'test'):
            c['training_hold'] = f"B v3 split '{c['split_hint']}': kept out of training to keep evaluation-side rows separate"
        apply_review(c)
        c['synthetic_parts'] = 'template lines' if c['kind'] == 'official_interpretation' else 'question and explanation (AI-written)'
    batch = [c for c in out if not c['training_hold']]
    excluded = [c for c in out if c['training_hold']]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'candidates.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in batch))
    (a.out / 'excluded.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in excluded))
    plain, plain_rejected = plain_cases(batch)
    for c in plain:
        apply_review(c)
        if c.get('training_hold'):
            plain_rejected[c['verbatim_case']] = c['training_hold']
    plain = [c for c in plain if not c.get('training_hold')]
    (a.out / 'plain-candidates.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in plain))
    # Every document behind a built case (kept or held) is training-side: it must not appear in a final evaluation set.
    (a.out / 'final-eval-exclusion.json').write_text(json.dumps(dict(
        rule='training-side documents: do not use in any final evaluation set (C-013)',
        docs=sorted({c['doc'] for c in out})), ensure_ascii=False, indent=2))
    meta = dict(purpose='internal-work SFT candidates (question / context / target / evidence); not evaluation items; '
                        'AI review only where review.status says so, never human-verified here; tokenizer/template left to A',
                review_status=dict(Counter(c['review']['status'] for c in batch)),
                task_types=dict(Counter(f"{c['task_type']} (source_answer_present={c['source_answer_present']})" for c in batch + plain)),
                candidates=dict(Counter(c['behavior'] for c in batch)), candidates_total=len(batch),
                excluded=dict(Counter(c['training_hold'][:40] for c in excluded)), skipped_before_case=dict(skipped),
                evidence_docs=sorted({c['doc'] for c in batch}),
                plain_explanations=dict(written=len(plain), rejected=plain_rejected,
                                        note='separate file plain-candidates.jsonl; AI-written, synthetic; review status per case; '
                                             'the verbatim case stays in candidates.jsonl'),
                inputs={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (V3, PACKS, PACK_CASES, *OVERLAPS, C010_DEV, REVIEW_HOLD, PLAIN_CASES, *a.reviews)},
                reserve_note='148 reserve families checked by B (v3 quarantine) and B-013 full scan; the other 852 families are not checked')
    (a.out / 'manifest.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != 'evidence_docs'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
