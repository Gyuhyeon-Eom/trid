"""Apply a per-row review of a case training view's train split and write a new, fixed version (MODEL-REPAIR-002).

The citations of build_case_training_view.py are a heuristic (numbers present, or 15% shingle overlap); they are not a
verification. This step takes a review record for every train row (decision pass / fix / hold, with what was checked:
negation, condition, exception, subject, time) and writes a new version:
  pass  the row is kept as built
  fix   the target is replaced by the reviewed corrected_target, which must still meet the contract: every [Dn] names an
        existing passage, every number of a cited sentence is in its cited passages, the question or the provenance
        line, answer <= --max-answer-tokens and the whole sequence <= --max-total-tokens (counted as A's trainer does)
  hold  the row moves to held.jsonl with the review reason
  author  (rows the source view held) the row comes back with the authored 'target', checked like a fix
A train row without a review record stops the build. The dev split is carried unchanged (it is not reviewed here); so
are the source view's held rows that no 'author' record brings back, with their original reasons.
In every row the provenance line names a statute's title '법령 …' instead of '기관 …' (the pilot stores the statute
title in its institution field for 별표/서식 rows).

  .venv/bin/python scripts/apply_case_review.py --view data/case-training-view-v1 \
      --review reports/coordination/MODEL-REPAIR-002/train-review-v1.jsonl \
      --tokenizer runs/pilot-sft-001-a-qwen3-1p7b/train/merged --out data/case-training-view-v1r1
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
sys.path.insert(0, str(ROOT / 'scripts'))
from build_case_training_view import nums, sentences  # noqa: E402

CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()  # as loaded
CITE = re.compile(r' ((?:\[D\d\])+)')


def relabel(row):
    """'기관 <statute title>' -> '법령 <statute title>' in the provenance line of law.go.kr statute rows."""
    if not (row.get('doc') or '').startswith(('law-go-kr:law-byl:', 'law-go-kr:law:')) or not row.get('institution'):
        return row['context'], False
    first = row['context'][0].replace(f"· 기관 {row['institution']}", f"· 법령 {row['institution']}", 1)
    return [first] + row['context'][1:], first != row['context'][0]


def contract_problems(target, context, question):
    problems = []
    allowed = nums(question)
    for a, b in sentences(target):
        sent = target[a:b]
        refs = {int(k) for m in CITE.finditer(sent) for k in re.findall(r'D(\d)', m.group(1))}
        if any(k < 1 or k > len(context) for k in refs):
            problems.append(f'citation_out_of_range:{sorted(refs)}')
            continue
        body = CITE.sub('', sent)
        missing = nums(body) - allowed - set().union(set(), *(nums(context[k - 1]) for k in refs))
        if refs and missing:
            problems.append(f'number_not_in_cited_passage:{sorted(missing)[0]}')
    if '[D' not in target:
        problems.append('no_citation')
    return problems


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--view', required=True)
    p.add_argument('--review', required=True)
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--max-answer-tokens', type=int, default=256)
    p.add_argument('--max-total-tokens', type=int, default=4096)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    from transformers import AutoTokenizer
    from trid.case_prompts import CASE_SYSTEM
    from trid.prompts import rag_prompt
    view, out = ROOT / args.view, ROOT / args.out
    if out.exists():
        raise SystemExit(f'{out} exists; versions are never overwritten')
    src = json.loads((view / 'manifest.json').read_text())
    for name in ('train', 'dev'):
        if hashlib.sha256((view / f'{name}.jsonl').read_bytes()).hexdigest() != src['files'][name]['sha256']:
            raise SystemExit(f'{view}/{name}.jsonl differs from its manifest')
    review_bytes = (ROOT / args.review).read_bytes()
    review = {r['id']: r for r in map(json.loads, review_bytes.decode().splitlines())}
    tok = AutoTokenizer.from_pretrained(str(ROOT / args.tokenizer), local_files_only=True)

    def tokens(context, question, target):
        user = rag_prompt([(f'D{i}', t) for i, t in enumerate(context, 1)], question)
        prompt = tok.apply_chat_template([{'role': 'system', 'content': CASE_SYSTEM}, {'role': 'user', 'content': user}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False)
        answer = len(tok(target + tok.eos_token, add_special_tokens=False)['input_ids'])
        return answer, len(tok(prompt, add_special_tokens=False)['input_ids']) + answer

    kept, held, decisions, relabelled = {'train': [], 'dev': []}, [], Counter(), 0
    for name in ('train', 'dev'):
        for row in map(json.loads, (view / f'{name}.jsonl').read_text().splitlines()):
            context, changed = relabel(row)
            relabelled += changed
            row = {**row, 'context': context}
            if name == 'train':
                rec = review.get(row['id'])
                if rec is None:
                    raise SystemExit(f"{row['id']}: no review record")
                decisions[rec['decision']] += 1
                if rec['decision'] == 'hold':
                    held.append({**row, 'held': [rec['reason']], 'review': rec})
                    continue
                if rec['decision'] == 'fix':
                    problems = contract_problems(rec['corrected_target'], context, row['question'])
                    if problems:
                        raise SystemExit(f"{row['id']}: corrected target breaks the contract: {problems}")
                    row = {**row, 'target': rec['corrected_target'], 'transforms': row['transforms'] + ['reviewed_fix']}
                row = {**row, 'review': {k: rec[k] for k in ('decision', 'checks') if k in rec}}
            a, t = tokens(row['context'], row['question'], row['target'])
            if a > args.max_answer_tokens or t > args.max_total_tokens:
                raise SystemExit(f"{row['id']}: over budget after review ({a} answer, {t} total tokens)")
            kept[name].append({**row, 'answer_tokens': a, 'total_tokens': t})
    for row in map(json.loads, (view / 'held.jsonl').read_text().splitlines()):
        context, changed = relabel(row)
        relabelled += changed
        row = {**row, 'context': context}
        rec = review.get(row['id'])
        if row['split'] != 'train' or rec is None or rec['decision'] != 'author':
            held.append(row)
            continue
        decisions['author'] += 1
        problems = contract_problems(rec['target'], context, row['question'])
        if problems:
            raise SystemExit(f"{row['id']}: authored target breaks the contract: {problems}")
        row = {k: v for k, v in row.items() if k != 'held'}
        row = {**row, 'target': rec['target'], 'transforms': row['transforms'] + ['authored_short_target'],
               'review': {k: rec[k] for k in ('decision', 'checks') if k in rec}}
        a, t = tokens(row['context'], row['question'], row['target'])
        if a > args.max_answer_tokens or t > args.max_total_tokens:
            raise SystemExit(f"{row['id']}: authored target over budget ({a} answer, {t} total tokens)")
        kept['train'].append({**row, 'answer_tokens': a, 'total_tokens': t})
    unused = review.keys() - {r['id'] for r in kept['train']} - {r['id'] for r in held}
    if unused:
        raise SystemExit(f'review records for rows not in the train split: {sorted(unused)}')
    out.mkdir(parents=True)
    files = {}
    for name, rows in (('train', kept['train']), ('dev', kept['dev']), ('held', held)):
        body = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode()
        (out / f'{name}.jsonl').write_bytes(body)
        files[name] = {'rows': len(rows), 'sha256': hashlib.sha256(body).hexdigest()}
    manifest = {
        'version': out.name, 'code_sha256': CODE_SHA256, 'source_view': {'path': args.view, 'files': src['files']},
        'review': {'path': args.review, 'sha256': hashlib.sha256(review_bytes).hexdigest(), 'decisions': dict(decisions),
                   'reviewer': 'B (AI, one reviewer; not human or independent review)', 'scope': 'train split only'},
        'provenance_relabelled_rows': relabelled, 'files': files,
        'tokens': {name: {'answer_max': max((r['answer_tokens'] for r in rows), default=0),
                          'total_max': max((r['total_tokens'] for r in rows), default=0),
                          'answer_total': sum(r['answer_tokens'] for r in rows)} for name, rows in kept.items()},
        'max_answer_tokens': args.max_answer_tokens, 'max_total_tokens': args.max_total_tokens,
        'contract': src.get('contract')}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    a_manifest = {'id': f'{out.name}-a', 'source': f'{args.out} (B, MODEL-REPAIR-002 reviewed train split)',
                  'train': [{'path': 'train.jsonl', 'sha256': files['train']['sha256'], 'adapter': 'case', 'split': 'train', 'epochs': 2}],
                  'valid': [{'path': 'dev.jsonl', 'sha256': files['dev']['sha256'], 'adapter': 'case', 'split': 'pilot_dev'}]}
    (out / 'a-train-manifest.json').write_text(json.dumps(a_manifest, ensure_ascii=False, indent=1))
    print(json.dumps({k: manifest[k] for k in ('files', 'review', 'provenance_relabelled_rows', 'tokens')}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
