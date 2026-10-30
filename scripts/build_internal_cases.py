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


def section_spans(text):
    """Character spans in the v3 text: 질의요지 (without '< 질의 배경 >'), 회답, 이유 (without the appendix)."""
    q0 = text.index('【질의요지】') + len('【질의요지】')
    a_head = text.index('【회답】')
    bg = text.find('< 질의 배경 >', q0, a_head)
    y_head = text.index('【이유】')
    y0 = y_head + len('【이유】')
    tail = re.search(r'\n※\s*법령정비|\n<\s*관계\s*법령\s*>|\n【관계법령】', text[y0:])
    return dict(question=(q0, bg if bg != -1 else a_head), answer=(a_head + len('【회답】'), y_head),
                reasons=(y0, y0 + tail.start() if tail else len(text)))


FOOTNOTE_BODY = re.compile(r'^\s*(\d+)\)\1\)\s*')


def attach_footnotes(text, span, edits):
    """Section text with each footnote (B v3-norm2 'footnote' edit, offsets into the v3 text) replaced by its
    [각주N] marker, plus the bodies of exactly the footnotes cited in this section, in citation order.
    Raises ValueError when a footnote crosses the section boundary, a marker has no body or a body no marker,
    or a number repeats in the section."""
    start, end = span
    out, notes, pos = [], [], start
    for e in sorted((e for e in edits if e['src_end'] > start and e['src_start'] < end), key=lambda e: e['src_start']):
        if e['src_start'] < start or e['src_end'] > end:
            raise ValueError('footnote crosses a section boundary')
        raw = text[e['src_start']:e['src_end']]
        m = FOOTNOTE_BODY.match(raw)
        n = re.fullmatch(r'\[각주(\d+)\]', e['replacement'])
        if not m or not n or m.group(1) != n.group(1):
            raise ValueError(f'footnote edit does not match its source span: {raw[:20]!r}')
        out += [text[pos:e['src_start']], e['replacement']]
        notes.append((n.group(1), raw[m.end():].strip(), e['src_start'], e['src_end']))
        pos = e['src_end']
    out.append(text[pos:end])
    body = ''.join(out)
    cited = re.findall(r'\[각주(\d+)\]', body)
    numbers = [n for n, *_ in notes]
    if sorted(cited) != sorted(numbers) or len(set(numbers)) != len(numbers):
        raise ValueError('footnote markers and bodies do not match one to one in the section')
    return body, notes


def with_notes(body, notes):
    return body + ''.join(f'\n[각주{n}] {b}' for n, b, *_ in notes)


def official_case(row, footnote_edits=None):
    """footnote_edits: B v3-norm2 footnote edits for this row (offsets into the v3 text). Without them, rows whose
    sections contain footnote text are skipped; with them, each section carries its own cited footnotes."""
    text = row['text']
    if not all(k in text for k in ('【질의요지】', '【회답】', '【이유】')):
        return None, 'no sections'
    spans = section_spans(text)
    notes = {}
    try:
        parts = {}
        for name, span in spans.items():
            body, notes[name] = attach_footnotes(text, span, footnote_edits or [])
            parts[name] = body
    except ValueError as err:
        return None, f'footnote: {err}'
    owners = {}
    for name in notes:
        for _, b, *_ in notes[name]:
            owners.setdefault(b, set()).add(name)
    if any(len(s) > 1 for s in owners.values()):
        return None, 'footnote: the same footnote body is attached in two sections'
    question = with_notes(re.sub(r'\s+', ' ', parts['question']).strip(), notes['question'])
    answer = with_notes(parts['answer'].strip(), notes['answer'])
    reasons = with_notes(parts['reasons'].strip(), notes['reasons'])
    if re.search(r'질의\s*[가나다1-9]', answer) or re.search(r'(^|\s)가\.\s.*\s나\.\s', question):
        return None, 'multi-part question or answer'
    if not parts['question'].strip() or not parts['answer'].strip() or not parts['reasons'].strip():
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
    case = dict(kind='official_interpretation', behavior='answer_with_basis', question=question, context=[context],
                task_type=TASK_OFFICIAL, source_answer_present=True,
                target=target, evidence=dict(conclusion=answer, reasons=reasons), date=date, date_field=kind,
                agenda=agenda, title=row['title'],
                authored='question (whole 질의요지), conclusion and reasons verbatim from the official interpretation; '
                         'date/source lines are a fixed template')
    if any(notes.values()):
        case['footnotes'] = {name: [dict(n=n, body=b, src_start=s, src_end=e) for n, b, s, e in ns]
                             for name, ns in notes.items() if ns}
        case['authored'] += '; footnotes cited in a section are moved, verbatim, to the end of that section ([각주N])'
    return case, None


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
    p.add_argument('--footnotes', type=Path,
                   help='B v3-norm2 docs.jsonl: only its footnote edits (offsets into v3) are used, per section')
    a = p.parse_args()
    a.reviews = [r.resolve() for r in a.reviews]
    footnotes = {}
    if a.footnotes:
        for line in a.footnotes.resolve().open():
            r = json.loads(line)
            norm = r['meta']['normalization']
            footnotes[r['id']] = dict(source_sha256=norm['source_sha256'],
                                      edits=[e for e in norm['edits'] if e['rule'] == 'footnote'])
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
        fn = footnotes.get(row['id'])
        if fn and fn['source_sha256'] != row['sha256']:
            raise ValueError(f"{row['id']}: footnote offsets were made for a different v3 text")
        case, why = official_case(row, fn['edits'] if fn else None)
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
        keep = ('reviewer_id', 'reviewer_type', 'verdict', 'note', 'file', 'scope', 'work_relevance', 'dimensions', 'outdated_flag')
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
    # Per-document provenance (C-014). Says what C did with each document; it is not a removal list: existing
    # evaluation sets are never pruned from it, and final evaluation design is decided by the supervisor and A.
    def doc_status(c):
        hold = c.get('training_hold') or ''
        if hold.startswith('reserve-overlap'):
            return 'held_reserve_overlap'          # kept out of training; not training-side
        if hold.startswith('B v3 split'):
            return 'held_eval_side_split'
        if hold:
            return 'held_review_defect'
        return 'training_candidate_ai_reviewed' if c['review']['status'] == 'ai_reviewed' else 'training_candidate_unreviewed'
    provenance = {}
    for c in out:
        provenance.setdefault(c['doc'], set()).add(doc_status(c))
    (a.out / 'doc-provenance.json').write_text(json.dumps(dict(
        statuses={'training_candidate_ai_reviewed': 'C built a training candidate and an AI review matched its content',
                  'training_candidate_unreviewed': 'C built a training candidate; content not reviewed',
                  'held_reserve_overlap': 'B-013 reserve overlap: C keeps it out of training; it is not training-side',
                  'held_eval_side_split': "B v3 'valid'/'test' split row: kept out of training",
                  'held_review_defect': 'AI review found a defect: kept out of training'},
        training_used='unknown to C: whether a candidate was actually trained on is recorded by A',
        rule='provenance only; not a list for removing documents from existing evaluation sets; final evaluation '
             'design is decided by the supervisor and A',
        docs={d: sorted(s) for d, s in sorted(provenance.items())}), ensure_ascii=False, indent=2))
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
                        for p in (V3, PACKS, PACK_CASES, *OVERLAPS, C010_DEV, REVIEW_HOLD, PLAIN_CASES, *a.reviews,
                                  *([a.footnotes.resolve()] if a.footnotes else []))},
                footnotes=('B v3-norm2 footnote edits (offsets into v3) applied per section; other v3-norm2 rules not used'
                           if a.footnotes else 'not applied: rows with footnote text in a section are skipped'),
                reserve_note='148 reserve families checked by B (v3 quarantine) and B-013 full scan; the other 852 families are not checked')
    (a.out / 'manifest.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != 'evidence_docs'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
