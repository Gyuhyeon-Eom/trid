"""Over-long table cases (held 'over_context_budget') shortened by whole structural units, as a separate candidate.

Cutting a table down to the row that answers the question would make the case easy and leak the answer's location.
This step removes whole units instead and keeps what a reader needs to tell rows apart:
  kept always   the provenance line, the title/heading lines before the table, the column header row, every line after
                the table (notes, units, footnotes)
  units         a unit is one block of the table's first column: a row that starts while no first-column rowspan is
                active, plus the rows its rowspan covers (e.g. '7. 예산집행 및 사업 발주' with its 가./나. rows, or
                '8 | 인사업무' with its sub-rows). Units are never split
  removal       starting from the unit farthest (by position) from the unit the reviewed target cites, whole units are
                removed one at a time until the training sequence fits --max-total-tokens; neighbours of the cited unit,
                which hold the most similar rows, go last
  marking       the provenance line says how many units were left out before and after the kept block
Every candidate must still meet the case contract (numbers of cited sentences in the cited passage) and the budgets,
and the cited unit must be kept. The fixed diagnostic set (v1r1) and v2 are not changed: the output is a separate
candidate for review.

  .venv/bin/python scripts/build_structural_context_candidates.py --view data/case-training-view-v2 \
      --tokenizer runs/pilot-sft-001-a-qwen3-1p7b/train/merged --out data/case-structural-context-candidates-v1
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from apply_case_review import contract_problems  # noqa: E402
from build_case_training_view import squash  # noqa: E402

CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()  # as loaded
ROW = re.compile(r'<tr>.*?</tr>', re.S)


def units(table_body):
    """[(start, end)] character spans of first-column units in the rows of one <table>."""
    rows = [(m.start(), m.end(), m.group(0)) for m in ROW.finditer(table_body)]
    spans, remaining = [], 0
    for start, end, row in rows:
        if '<th' in row:
            continue
        if remaining <= 0:
            first = re.search(r'<td(?:\s+rowspan="(\d+)")?', row)
            remaining = int(first.group(1)) if first and first.group(1) else 1
            spans.append([start, end])
        else:
            spans[-1][1] = end
        remaining -= 1
    return [tuple(s) for s in spans]


def overlap(a, b, k=6):
    a, b = squash(a), squash(b)
    grams = {a[i:i + k] for i in range(len(a) - k + 1)}
    return sum(g in b for g in grams) / max(1, len(grams))


def unit_scores(target, texts, k=4):
    """How strongly the target's wording points at each unit: 4-character pieces of the target (no spaces/punctuation)
    found in a unit, weighted by how few units contain them, so table-wide words (과장, 전결) count for little."""
    import math
    t = squash(target)
    grams = {t[i:i + k] for i in range(len(t) - k + 1)}
    units_sq = [squash(x) for x in texts]
    df = {g: sum(g in u for u in units_sq) for g in grams}
    n = len(texts)
    return [sum(math.log(n / df[g]) for g in grams if df[g] and g in u) for u in units_sq]


def shorten(row, count_tokens, budget):
    passage = row['context'][0]
    t0, t1 = passage.find('<table>'), passage.rfind('</table>')
    if t0 < 0 or t1 < 0:
        return None, 'no_table'
    body = passage[t0:t1]
    spans = units(body)
    if len(spans) < 3:
        return None, 'too_few_units'
    header_end = spans[0][0]
    target = re.sub(r' (?:\[D\d\])+', '', row['target'])
    score = unit_scores(target, [re.sub(r'<[^>]+>', ' ', body[s:e]) for s, e in spans])
    cited = max(range(len(spans)), key=score.__getitem__)
    protected = {i for i, s in enumerate(score) if s >= 0.5 * score[cited]}
    keep = list(range(len(spans)))

    def build(kept):
        lo, hi = kept[0], kept[-1]
        note = f' · 표 일부 생략(앞 단위업무 {lo}개, 뒤 {len(spans) - 1 - hi}개)' if (lo or hi < len(spans) - 1) else ''
        head = passage[:t0]
        first_line, rest = head.split('\n', 1) if '\n' in head else (head, '')
        rows = ''.join(body[spans[i][0]:spans[i][1]] + '\n' for i in kept)
        return first_line + note + '\n' + rest + body[:header_end] + rows + passage[t1:]

    while True:
        new = build(keep)
        if count_tokens([new] + row['context'][1:]) <= budget:
            break
        if len(keep) <= 3:
            return None, 'still_over_budget'
        far = max((i for i in (keep[0], keep[-1]) if i not in protected), key=lambda i: abs(i - cited), default=None)
        if far is None:
            return None, 'still_over_budget'
        keep.remove(far)
    plain = lambda s: re.sub(r'<[^>]+>', ' ', s)  # noqa: E731
    if overlap(target, plain(new)) < overlap(target, plain(passage)):
        return None, 'target_wording_lost'
    return {'context': [new] + row['context'][1:], 'units_total': len(spans), 'units_kept': len(keep),
            'cited_unit': cited, 'protected_units': sorted(protected), 'kept_range': [keep[0], keep[-1]]}, None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--view', required=True)
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--max-total-tokens', type=int, default=4096)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    from transformers import AutoTokenizer
    from trid.case_prompts import CASE_SYSTEM
    from trid.prompts import rag_prompt
    out = ROOT / args.out
    if out.exists():
        raise SystemExit(f'{out} exists; versions are never overwritten')
    tok = AutoTokenizer.from_pretrained(str(ROOT / args.tokenizer), local_files_only=True)
    rows = [json.loads(x) for x in (ROOT / args.view / 'held.jsonl').read_text().splitlines() if x]
    rows = [r for r in rows if r['split'] == 'train' and r['held'] == ['over_context_budget']]
    kept, failed = [], []
    for row in rows:
        answer = len(tok(row['target'] + tok.eos_token, add_special_tokens=False)['input_ids'])

        def count(context):
            user = rag_prompt([(f'D{i}', t) for i, t in enumerate(context, 1)], row['question'])
            prompt = tok.apply_chat_template([{'role': 'system', 'content': CASE_SYSTEM}, {'role': 'user', 'content': user}],
                                             tokenize=False, add_generation_prompt=True, enable_thinking=False)
            return len(tok(prompt, add_special_tokens=False)['input_ids']) + answer
        result, why = shorten(row, count, args.max_total_tokens)
        if result is None:
            failed.append({'id': row['id'], 'reason': why})
            continue
        problems = contract_problems(row['target'], result['context'], row['question'])
        if problems:
            failed.append({'id': row['id'], 'reason': f'contract:{problems}'})
            continue
        kept.append({**{k: v for k, v in row.items() if k != 'held'}, 'context': result['context'],
                     'answer_tokens': answer, 'total_tokens': count(result['context']),
                     'structural_context': {k: result[k] for k in ('units_total', 'units_kept', 'cited_unit', 'protected_units', 'kept_range')},
                     'transforms': row['transforms'] + ['structural_context_units'], 'status': 'candidate_needs_review'})
    out.mkdir(parents=True)
    body = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in kept).encode()
    (out / 'train-candidates.jsonl').write_bytes(body)
    manifest = {'version': out.name, 'code_sha256': CODE_SHA256, 'source_view': args.view, 'max_total_tokens': args.max_total_tokens,
                'candidates': len(kept), 'sha256': hashlib.sha256(body).hexdigest(), 'failed': failed,
                'per_row': {r['id']: r['structural_context'] | {'total_tokens': r['total_tokens']} for r in kept},
                'note': 'not part of any training manifest; needs review before use'}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(json.dumps({k: manifest[k] for k in ('candidates', 'failed', 'per_row')}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
