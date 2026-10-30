"""New candidate version of a fixed case training view with a few reviewed target changes (B-032: v2 -> v2r1).

Only the listed train rows change, and only their `target` (plus answer/total token counts, transforms and review).
Every other train row and the dev and held files are written byte for byte as in the parent; the build checks this
after writing. A change is accepted only if:
  - the parent files match the parent manifest, and the row's current target matches `before_target_sha256`
  - the new target splits into exactly the listed sentences, and every sentence that cites [Dn] carries at least one
    evidence quote found verbatim in that passage (its char span is recorded in the manifest)
  - the case contract of apply_case_review.py holds (existing [Dn], numbers in the cited passage or the question)
  - answer <= --max-answer-tokens and the whole training sequence <= --max-total-tokens, counted as A's trainer does;
    the tokenizer must reproduce the parent's stored counts for every unchanged train row
The output is a candidate, not a training approval; no training is run and no A training manifest is written.

  .venv/bin/python scripts/apply_case_target_changes.py --view data/case-training-view-v2 \
      --changes reports/coordination/B-032/v2r1-changes.json \
      --tokenizer runs/pilot-sft-001-a-qwen3-1p7b/train/merged --out data/case-training-view-v2r1
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from apply_case_review import contract_problems  # noqa: E402
from build_case_qa_ledger import CITE, sentences  # noqa: E402

CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()  # as loaded
sha = lambda b: hashlib.sha256(b).hexdigest()


def anchor(row_id, spec, target, context):
    """Sentence records with char spans; stops the build when the spec does not match the target or the passage."""
    split = [s for line in target.split('\n') for s in sentences(line)]
    if split != [s['text'] for s in spec]:
        raise SystemExit(f'{row_id}: listed sentences differ from the split of the new target')
    out = []
    for s in spec:
        cited = {int(n) for n in CITE.findall(s['text'])}
        if cited and not s['evidence']:
            raise SystemExit(f"{row_id}: cited sentence without evidence: {s['text'][:60]}")
        ev = []
        for e in s['evidence']:
            d = int(e['cite'][1:])
            if d not in cited:
                raise SystemExit(f"{row_id}: evidence from {e['cite']} for a sentence citing {sorted(cited)}")
            a = context[d - 1].find(e['quote'])
            if a < 0:
                raise SystemExit(f"{row_id}: quote not verbatim in {e['cite']}: {e['quote'][:60]}")
            ev.append({'cite': e['cite'], 'char_span': [a, a + len(e['quote'])], 'quote': e['quote']})
        out.append({'text': s['text'], 'verdict': s['verdict'], 'evidence': ev})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--view', required=True)
    p.add_argument('--changes', required=True)
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
    manifest_bytes = (view / 'manifest.json').read_bytes()
    parent = json.loads(manifest_bytes)
    raw = {name: (view / f'{name}.jsonl').read_bytes() for name in ('train', 'dev', 'held')}
    for name, body in raw.items():
        if sha(body) != parent['files'][name]['sha256']:
            raise SystemExit(f'{view}/{name}.jsonl differs from its manifest')
    changes_bytes = (ROOT / args.changes).read_bytes()
    changes = json.loads(changes_bytes)
    tok = AutoTokenizer.from_pretrained(str(ROOT / args.tokenizer), local_files_only=True)

    def tokens(row, target):
        user = rag_prompt([(f'D{i}', t) for i, t in enumerate(row['context'], 1)], row['question'])
        prompt = tok.apply_chat_template([{'role': 'system', 'content': CASE_SYSTEM}, {'role': 'user', 'content': user}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False)
        answer = len(tok(target + tok.eos_token, add_special_tokens=False)['input_ids'])
        return answer, len(tok(prompt, add_special_tokens=False)['input_ids']) + answer

    lines = raw['train'].decode().splitlines(keepends=True)
    todo, records, new_lines, train_rows = dict(changes['rows']), [], [], []
    for line in lines:
        row = json.loads(line)
        spec = todo.pop(row['id'], None)
        if spec is None:
            if tokens(row, row['target']) != (row['answer_tokens'], row['total_tokens']):
                raise SystemExit(f"{row['id']}: tokenizer does not reproduce the parent's token counts")
            new_lines.append(line)
            train_rows.append(row)
            continue
        if sha(row['target'].encode()) != spec['before_target_sha256']:
            raise SystemExit(f"{row['id']}: current target differs from before_target_sha256")
        target = spec['target']
        problems = contract_problems(target, row['context'], row['question'])
        if problems:
            raise SystemExit(f"{row['id']}: new target breaks the contract: {problems}")
        sents = anchor(row['id'], spec['sentences'], target, row['context'])
        a, t = tokens(row, target)
        if a > args.max_answer_tokens or t > args.max_total_tokens:
            raise SystemExit(f"{row['id']}: over budget ({a} answer, {t} total tokens)")
        new = {**row, 'target': target, 'answer_tokens': a, 'total_tokens': t,
               'transforms': row['transforms'] + ['b032_reviewed_fix'],
               'review': {'decision': 'fix', 'issue': spec['issue'], 'source': args.changes,
                          'reviewer': changes['reviewer']}}
        new_lines.append(json.dumps(new, ensure_ascii=False) + '\n')
        train_rows.append(new)
        records.append({'id': row['id'], 'issue': spec['issue'], 'before_target_sha256': spec['before_target_sha256'],
                        'after_target_sha256': sha(target.encode()), 'answer_tokens': [row['answer_tokens'], a],
                        'total_tokens': [row['total_tokens'], t],
                        'deviation_from_b031_proposal': spec.get('deviation_from_b031_proposal'), 'sentences': sents})
    if todo:
        raise SystemExit(f'changes for rows not in the train split: {sorted(todo)}')
    out.mkdir(parents=True)
    bodies = {'train': ''.join(new_lines).encode(), 'dev': raw['dev'], 'held': raw['held']}
    for name, body in bodies.items():
        (out / f'{name}.jsonl').write_bytes(body)
    # after writing: unchanged rows byte-identical, dev/held identical, only the listed rows differ
    written = (out / 'train.jsonl').read_bytes().decode().splitlines(keepends=True)
    changed_ids = {r['id'] for r in records}
    differ = [json.loads(b)['id'] for a, b in zip(lines, written) if a != b]
    if len(written) != len(lines) or set(differ) != changed_ids:
        raise SystemExit(f'unexpected train differences: {differ}')
    for name in ('dev', 'held'):
        if sha((out / f'{name}.jsonl').read_bytes()) != parent['files'][name]['sha256']:
            raise SystemExit(f'{name} changed')
    files = {name: {'rows': len(body.decode().splitlines()), 'sha256': sha(body)} for name, body in bodies.items()}
    manifest = {
        'version': out.name, 'status': 'candidate; not approved for training; no training run',
        'code_sha256': CODE_SHA256,
        'parent': {'path': args.view, 'manifest_sha256': sha(manifest_bytes), 'files': parent['files']},
        'changes': {'path': args.changes, 'sha256': sha(changes_bytes), 'rows': records},
        'unchanged_train_rows': {'count': len(lines) - len(records), 'check': 'byte-identical JSON lines to parent'},
        'reviewer': changes['reviewer'], 'time_basis': changes['time_basis'], 'kept_memos': changes['kept_memos'],
        'files': files,
        'tokens': {'train': {'answer_max': max(r['answer_tokens'] for r in train_rows),
                             'total_max': max(r['total_tokens'] for r in train_rows),
                             'answer_total': sum(r['answer_tokens'] for r in train_rows)},
                   'dev': parent['tokens']['dev']},
        'max_answer_tokens': args.max_answer_tokens, 'max_total_tokens': args.max_total_tokens,
        'contract': parent.get('contract')}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(json.dumps({k: manifest[k] for k in ('parent', 'files', 'unchanged_train_rows', 'tokens')}, ensure_ascii=False,
                     indent=1))
    for r in records:
        print(r['id'], r['answer_tokens'], r['total_tokens'], r['after_target_sha256'][:12])


if __name__ == '__main__':
    main()
