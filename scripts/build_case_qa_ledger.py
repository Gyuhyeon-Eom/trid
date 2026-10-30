"""QA ledger of a fixed case-training view (B-031): one row per train row, the reviewer's decision with a source span for
every cited sentence, plus deterministic heuristics kept apart from the review.

Inputs (never modified):
  --train       the fixed view, e.g. data/case-training-view-v2/train.jsonl
  --prior       an earlier per-row review (MODEL-REPAIR-002/train-review-v2.jsonl); rows it decided pass/fix are
                carried over as `reused` (their decision and checks are copied, not re-judged); rows marked `author`
                (targets written by the reviewer's own builder) do not count as reviewed
  --judgements  manual judgements {rows: {id: {decision, note, issues, line_issues?, corrected_target?}}}

Per sentence (a target line split after its [Dn] markers) the ledger stores, for each cited passage, the char span with
the highest character-3-gram overlap and its text: the span the reviewer read. The overlap score is a locator, not a
verdict. review_kind is `full_manual` (judged here), `reused` (prior pass/fix) or `heuristic_only` (neither).

Heuristics (diagnostics only; a flag is not a defect and a clean row is not a pass):
  bad_cites          [Dn] numbers with no passage
  uncited_lines      target lines without any [Dn]
  missing_numbers    numbers of a cited sentence found neither in its cited passages nor in the question
  question_overlap   share of the target's 3-grams that also occur in the question (copying)
  instruction_copy   8-character runs shared with the training instruction (CASE_SYSTEM + rag_prompt frame)
  target_chars       length; long_target when above --long-chars
  time               date of time_label vs dates on the passages' '자료 정보' lines and in the target
  mixed_time         passages of one row carrying different '기준' dates
  min_overlap        lowest locator score of the row (paraphrase indicator)
  implicit_subject   the first line refers to '이 사안' instead of naming who/what (copied from the 회답)

  .venv/bin/python scripts/build_case_qa_ledger.py --train data/case-training-view-v2/train.jsonl \
      --prior reports/coordination/MODEL-REPAIR-002/train-review-v2.jsonl \
      --judgements reports/coordination/B-031/manual-judgements.json --out reports/coordination/B-031
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trid.case_prompts import CASE_SYSTEM  # noqa: E402
from trid.prompts import rag_prompt  # noqa: E402

CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()  # as loaded
CITE = re.compile(r'\[D(\d+)\]')
SENTENCE = re.compile(r'(.*?(?:\s*\[D\d+\])+\.?)\s*')
NUM = re.compile(r'\d[\d,.]*\d|\d')
DATE = re.compile(r'(\d{4})[-.]\s?(\d{1,2})[-.]\s?(\d{1,2})')
INFO = re.compile(r'자료 정보:[^\n]*')
QUOTE_MAX = 300
INSTRUCTION = CASE_SYSTEM + '\n' + rag_prompt([('D1', '')], '')  # the training instruction around every target


def grams(s, n=3):
    s = re.sub(r'\s', '', s)
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def best_span(sentence, passage, step=10):
    """(start, end, score): passage window of ~1.3x the sentence length with the highest 3-gram overlap."""
    g = grams(sentence)
    if not g:
        return 0, 0, 0.0
    width = max(40, int(len(sentence) * 1.3))
    best = (0, min(len(passage), width), 0.0)
    for a in range(0, max(1, len(passage) - 20), step):
        score = len(g & grams(passage[a:a + width])) / len(g)
        if score > best[2]:
            best = (a, min(len(passage), a + width), score)
    return best


def sentences(line):
    """Split a target line after each run of [Dn] markers; a trailing part without markers is its own sentence."""
    out, pos = [], 0
    for m in SENTENCE.finditer(line):
        if m.end() == m.start():
            break
        if not CITE.search(m.group(1)):
            break
        out.append(m.group(1).strip())
        pos = m.end()
    rest = line[pos:].strip()
    if rest:
        out.append(rest)
    return out


def dates(text):
    return {f'{y}-{int(m):02d}-{int(d):02d}' for y, m, d in DATE.findall(text)}


def numbers(text):
    return [n.replace(',', '') for n in NUM.findall(text)]


def heuristics(row, sents, long_chars):
    ctx, q = row['context'], row['question']
    flat = [re.sub(r'[,\s]', '', p) for p in ctx]
    qflat = re.sub(r'[,\s]', '', q)
    bad = sorted({int(n) for n in CITE.findall(row['target']) if not 1 <= int(n) <= len(ctx)})
    uncited = [line for line in row['target'].split('\n') if line.strip() and not CITE.search(line)]
    missing = []
    for s in sents:
        cited = [int(n) for n in CITE.findall(s['text']) if 1 <= int(n) <= len(ctx)]
        body = CITE.sub('', s['text'])
        for n in numbers(body):
            if not any(n in flat[d - 1] for d in cited) and n not in qflat:
                missing.append({'sentence': s['index'], 'number': n})
    tg = grams(row['target'])
    overlap = round(len(tg & grams(q)) / len(tg), 3) if tg else 0.0
    instruction = sorted(grams(row['target'], 8) & grams(INSTRUCTION, 8))
    label = dates(row.get('time_label', ''))
    info = [dates(m.group(0)) for p in ctx for m in [INFO.search(p)] if m]
    info_dates = set().union(*info) if info else set()
    target_dates = dates(row['target'])
    time = {'label_dates': sorted(label), 'passage_info_dates': sorted(info_dates),
            'target_dates': sorted(target_dates),
            'label_not_on_passage': sorted(label - info_dates) if info_dates else [],
            'target_date_not_on_passage': sorted(target_dates - info_dates - dates(' '.join(ctx)))}
    scores = [e['score'] for s in sents for e in s['evidence'] if 'score' in e]
    flags = []
    if bad:
        flags.append('bad_cites')
    if missing:
        flags.append('missing_numbers')
    if overlap >= 0.6:
        flags.append('question_copy')
    if len(row['target']) > long_chars:
        flags.append('long_target')
    if time['label_not_on_passage'] or time['target_date_not_on_passage']:
        flags.append('time_mismatch')
    if len([d for d in info if d]) > 1 and len({tuple(sorted(d)) for d in info if d}) > 1:
        flags.append('mixed_time')
    if '이 사안' in row['target'].split('\n')[0]:
        flags.append('implicit_subject')
    if instruction:
        flags.append('instruction_copy')
    return {'bad_cites': bad, 'uncited_lines': uncited, 'missing_numbers': missing, 'question_overlap': overlap,
            'instruction_8grams': instruction, 'target_chars': len(row['target']), 'time': time,
            'min_overlap': min(scores) if scores else None, 'flags': flags}


def ledger_row(row, prior, judged, long_chars):
    ctx, sents = row['context'], []
    for li, line in enumerate(row['target'].split('\n')):
        for s in sentences(line):
            ev = []
            for d in sorted({int(n) for n in CITE.findall(s)}):
                if not 1 <= d <= len(ctx):
                    ev.append({'cite': f'D{d}', 'error': 'no such passage'})
                    continue
                a, b, score = best_span(CITE.sub('', s), ctx[d - 1])
                ev.append({'cite': f'D{d}', 'char_span': [a, b], 'score': round(score, 3),
                           'quote': ctx[d - 1][a:b][:QUOTE_MAX]})
            sents.append({'index': len(sents), 'line': li, 'text': s, 'evidence': ev})
    if judged:
        kind, decision = 'full_manual', judged['decision']
        line_issues = judged.get('line_issues', {})
        for s in sents:
            hit = line_issues.get(str(s['line']))
            s['verdict'] = hit['issue'] if hit else ('supported' if s['evidence'] else 'no_citation_needed')
        review = {k: judged[k] for k in ('note', 'issues', 'line_issues') if judged.get(k)}
    elif prior and prior['decision'] in ('pass', 'fix'):
        kind, decision = 'reused', {'pass': 'pass', 'fix': 'fix_applied'}[prior['decision']]
        for s in sents:
            s['verdict'] = 'reused_prior_review'
        review = {'prior_checks': prior.get('checks'), 'prior_reason': prior.get('reason'),
                  'prior_note': prior.get('note')}
        if prior['decision'] == 'fix':
            review['fix_applied_in_view'] = row['target'] == prior.get('corrected_target')
    else:
        kind, decision, review = 'heuristic_only', 'unreviewed', {}
        for s in sents:
            s['verdict'] = 'unreviewed'
    return {'id': row['id'], 'task_type': row.get('task_type'), 'time_label': row.get('time_label'),
            'prior_review': prior['decision'] if prior else None, 'review_kind': kind, 'decision': decision,
            'review': review, 'sentences': sents, 'heuristics': heuristics(row, sents, long_chars)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--train', required=True)
    ap.add_argument('--prior', required=True)
    ap.add_argument('--judgements', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--long-chars', type=int, default=600)
    args = ap.parse_args()
    train, out = ROOT / args.train, ROOT / args.out
    ledger_path, fixes_path, summary_path = out / 'v2-qa-ledger.jsonl', out / 'v2-fix-proposals.jsonl', out / 'summary.json'
    for p in (ledger_path, fixes_path, summary_path):
        if p.exists():
            raise SystemExit(f'{p} exists; versions are never overwritten')
    rows = [json.loads(x) for x in train.read_text().splitlines() if x.strip()]
    prior = {d['id']: d for d in map(json.loads, (ROOT / args.prior).read_text().splitlines()) if d}
    judgements = json.loads((ROOT / args.judgements).read_text())['rows']
    unknown = set(judgements) - {r['id'] for r in rows}
    if unknown:
        raise SystemExit(f'judgements for ids not in the view: {sorted(unknown)}')
    led = [ledger_row(r, prior.get(r['id']), judgements.get(r['id']), args.long_chars) for r in rows]
    fixes = []
    for r, e in zip(rows, led):
        j = judgements.get(r['id'])
        if j and j['decision'] == 'fix_proposed':
            evidence = []
            for li, issue in sorted(j.get('line_issues', {}).items()):
                for ev in issue.get('evidence', []):
                    passage = r['context'][int(ev['cite'][1:]) - 1]
                    a = passage.find(ev['quote'])
                    if a < 0:
                        raise SystemExit(f"{r['id']}: evidence quote not in {ev['cite']}: {ev['quote'][:60]}")
                    evidence.append({'line': int(li), 'cite': ev['cite'], 'char_span': [a, a + len(ev['quote'])],
                                     'quote': ev['quote']})
            fixes.append({'id': r['id'], 'issues': j['issues'], 'line_issues': j.get('line_issues'), 'note': j['note'],
                          'evidence': evidence, 'current_target': r['target'], 'proposed_target': j['corrected_target'],
                          'status': 'proposal only; the fixed view is unchanged'})
    out.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in led))
    fixes_path.write_text(''.join(json.dumps(f, ensure_ascii=False) + '\n' for f in fixes))
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    summary = {
        'code_sha256': CODE_SHA256, 'train': {'path': args.train, 'rows': len(rows), 'sha256': sha(train)},
        'prior': {'path': args.prior, 'sha256': sha(ROOT / args.prior)},
        'judgements': {'path': args.judgements, 'sha256': sha(ROOT / args.judgements)},
        'review_kind': dict(Counter(e['review_kind'] for e in led)),
        'decision_by_kind': {k: dict(Counter(e['decision'] for e in led if e['review_kind'] == k))
                             for k in sorted({e['review_kind'] for e in led})},
        'issues': dict(Counter(i for j in judgements.values() for i in j['issues'])),
        'heuristic_flags': dict(Counter(f for e in led for f in e['heuristics']['flags'])),
        'heuristic_rows_flagged': sum(bool(e['heuristics']['flags']) for e in led),
        'fix_proposals': len(fixes),
        'outputs': {'ledger': sha(ledger_path), 'fix_proposals': sha(fixes_path)},
        'reviewer': 'B (AI, one reviewer; not human or independent review). Heuristics are diagnostics, not review.'}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
