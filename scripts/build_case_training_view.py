"""Case training view: a frozen C pilot made to satisfy the prompt it is trained under (MODEL-REPAIR-001).

A trains `case` rows as CASE_SYSTEM + rag_prompt(context as D1..Dn, question) (scripts/train_open_weight_lora.py). That
user turn says '답에 출처 번호를 붙이세요' and '근거가 없으면 확인 불가'. The frozen pilots break that contract (pilot v3,
train split, 134 rows): 0 targets carry a [Dn] number; 72 rows are official answers copied whole (median 1,205 answer
tokens, >= 80% of the target copied from the passage) and hold ~89% of all answer tokens, so the run mostly teaches
copying long text; 80 targets exceed the 256-token serving answer budget; table targets cite a snapshot date that is
not in any passage. This builder writes a new version and never edits the pilot:

  provenance line   D1 starts with '자료 정보: 기관 … · 기준 …' built from the row's institution and time label, so a
                    date or institution the target names is in the evidence
  citations         every target sentence is matched to the passages that support it: passages that together hold
                    every number of the sentence (numbers also in the question need no source), or whose 6-character
                    shingles (text without spaces or punctuation) cover >= 15% of the sentence; ' [Dn]' is put before
                    the sentence's final period. A sentence with a number that neither a passage nor the question holds
                    holds the row (ungrounded_number); a row with no cited sentence is held (no_supported_sentence)
  budgets           rows whose target (after the above) exceeds --max-answer-tokens (serving budget) are held
                    (over_answer_budget; the copied official answers all fall here), and rows whose whole training
                    sequence (chat template of CASE_SYSTEM + rag_prompt, target, EOS: the count A's trainer drops at)
                    exceeds --max-total-tokens are held (over_context_budget) instead of being dropped silently
The original target is recoverable: removing the inserted ' [Dn]' markers and the '확인 불가: ' prefix gives it back
(checked per row; original_target_sha256 kept). Evaluation items are not read.

  .venv/bin/python scripts/build_case_training_view.py --pilot reports/coordination/C-019/pilot-v3-candidate/pilot.jsonl \
      --pilot-sha256 9d4d7b9f… --tokenizer runs/pilot-sft-001-a-qwen3-1p7b/train/merged \
      --out data/case-training-view-v1
"""
import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()  # as loaded
NO_EVIDENCE = {'hold_missing_basis', 'hold_time', 'lookup_needed'}
PREFIX = '확인 불가: '
SHINGLE = 6
MIN_SUPPORT = 0.15
SENTENCE_END = re.compile(r'(?<=[다요까음함됨임])\.(?=\s|$)')
# a comma is a thousands separator only before exactly three digits, so CSV rows ('100,D02,13010602') stay apart
NUMBER = re.compile(r'\d{1,3}(?:,\d{3})+(?![\d,])(?:\.\d+)?|\d+(?:\.\d+)?')


def squash(text):
    return re.sub(r'[\s\W_]+', '', text)


def nums(text):
    """Numbers without commas; circled paragraph numerals (①, ②, …) count as the digits they stand for."""
    text = re.sub('[\u2460-\u2473]', lambda m: f' {ord(m.group(0)) - 0x245F} ', text)
    return {n.replace(',', '') for n in NUMBER.findall(text)}


def provenance(row):
    label = row.get('time_label') or ''
    m = re.search(r'snapshot (\d{4}-\d{2}-\d{2})', label)
    s = re.search(r'in force from (\d{4})(\d{2})(\d{2})', label)
    o = re.search(r'as_of_interpretation_date=(\d{4}-\d{2}-\d{2})', label)
    when = (f'{m.group(1)} 자 수집본(이후 개정 여부 미확인)' if m else
            f'{s.group(1)}-{s.group(2)}-{s.group(3)} 시행 조문(이후 개정 여부 미확인)' if s else
            f'{o.group(1)} 법제처 해석 시점(현행 여부 미확인)' if o else None)
    parts = [f"문서 {row['doc']}" if row.get('doc') else None,
             f"기관 {row['institution']}" if row.get('institution') else None, f'기준 {when}' if when else None]
    parts = [p for p in parts if p]
    return f"자료 정보: {' · '.join(parts)}\n" if parts else ''


def sentences(target):
    """(start, end) spans of sentences; a sentence ends at '다.' etc. or at a line end."""
    out = []
    for line in re.finditer(r'[^\n]+', target):
        start = line.start()
        for m in SENTENCE_END.finditer(line.group(0)):
            out.append((start, line.start() + m.start()))  # end excludes the period
            start = line.start() + m.end()
        rest = target[start:line.end()]
        if rest.strip():
            out.append((start + len(rest) - len(rest.lstrip()), line.end()))
    return out


def support(sentence, passage):
    s = squash(sentence)
    if len(s) < SHINGLE:
        return 0.0
    p = squash(passage)
    grams = {s[i:i + SHINGLE] for i in range(len(s) - SHINGLE + 1)}
    return sum(g in p for g in grams) / len(grams)


def cite(row):
    """(target with citations, problems)."""
    target, passages, question = row['target'], row['context'], row['question']
    allowed_q = nums(question)
    pnums = [nums(p) for p in passages]
    pieces, last, problems, cited = [], 0, [], 0
    for a, b in sentences(target):
        sent = target[a:b]
        need = nums(sent) - allowed_q
        refs = {i for i, p in enumerate(passages, 1) if support(sent, p) >= MIN_SUPPORT}
        refs |= {i for i, n in enumerate(pnums, 1) if n & need}
        missing = need - set().union(set(), *(pnums[i - 1] for i in refs))
        if missing:
            problems.append(f'ungrounded_number:{sorted(missing)[0]}')
            continue
        if refs:
            body = sent.rstrip()
            pieces.append(target[last:a + len(body)] + ' ' + ''.join(f'[D{i}]' for i in sorted(refs)))
            last = a + len(body)
            cited += 1
    pieces.append(target[last:])
    if not cited:
        problems.append('no_supported_sentence')
    return ''.join(pieces), problems


def strip(target):
    t = target[len(PREFIX):] if target.startswith(PREFIX) else target
    return re.sub(r' (?:\[D\d\])+', '', t)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pilot', required=True)
    p.add_argument('--pilot-sha256', required=True)
    p.add_argument('--tokenizer', required=True, help='tokenizer directory of the base model (chat template included)')
    p.add_argument('--max-answer-tokens', type=int, default=256)
    p.add_argument('--max-total-tokens', type=int, default=4096)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    import sys
    sys.path.insert(0, str(ROOT))
    from transformers import AutoTokenizer
    from trid.case_prompts import CASE_SYSTEM
    from trid.prompts import rag_prompt
    out = ROOT / args.out
    if out.exists():
        raise SystemExit(f'{out} exists; versions are never overwritten')
    pilot = ROOT / args.pilot
    raw = pilot.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.pilot_sha256:
        raise SystemExit(f'{pilot}: sha256 is not the frozen {args.pilot_sha256}')
    tok = AutoTokenizer.from_pretrained(str(ROOT / args.tokenizer), local_files_only=True)
    kept, held = defaultdict(list), []
    tokens = defaultdict(Counter)
    for line in raw.decode().splitlines():
        row = json.loads(line)
        context = list(row['context'])
        context[0] = provenance(row) + context[0]
        cited, problems = cite({**row, 'context': context})
        target = (PREFIX + cited) if row['behavior'] in NO_EVIDENCE else cited
        if strip(target) != row['target']:
            raise SystemExit(f"{row['id']}: transform is not reversible")
        n = len(tok(target + tok.eos_token, add_special_tokens=False)['input_ids'])
        if n > args.max_answer_tokens:
            problems.append('over_answer_budget')
        user = rag_prompt([(f'D{i}', t) for i, t in enumerate(context, 1)], row['question'])
        prompt = tok.apply_chat_template([{'role': 'system', 'content': CASE_SYSTEM}, {'role': 'user', 'content': user}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False)
        total = len(tok(prompt, add_special_tokens=False)['input_ids']) + n
        if total > args.max_total_tokens:
            problems.append('over_context_budget')
        out_row = {'id': row['id'], 'split': row['split'], 'split_group': row['split_group'], 'task_type': row['task_type'],
                   'behavior': row['behavior'], 'question': row['question'], 'context': context, 'target': target,
                   'answer_tokens': n, 'total_tokens': total, 'original_target_sha256': hashlib.sha256(row['target'].encode()).hexdigest(),
                   'content_sha256': row.get('content_sha256'), 'time_label': row.get('time_label'),
                   'institution': row.get('institution'), 'doc': row.get('doc'),
                   'transforms': ['provenance_line'] * bool(provenance(row)) + ['citations'] +
                                 ['no_evidence_prefix'] * (row['behavior'] in NO_EVIDENCE)}
        if problems:
            held.append({**out_row, 'held': sorted(set(problems))})
        else:
            kept[row['split']].append(out_row)
            tokens[row['split']][row['task_type']] += n
    out.mkdir(parents=True)
    files = {}
    for name, rows in (('train', kept['train']), ('dev', kept['pilot_dev']), ('held', held)):
        body = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode()
        (out / f'{name}.jsonl').write_bytes(body)
        files[name] = {'rows': len(rows), 'sha256': hashlib.sha256(body).hexdigest()}
    held_reasons = Counter(h for r in held for h in r['held'])
    held_by_task = Counter((r['split'], r['task_type']) for r in held)
    manifest = {
        'version': out.name, 'code_sha256': CODE_SHA256, 'source': {'pilot': args.pilot, 'sha256': args.pilot_sha256},
        'tokenizer': {'path': args.tokenizer, 'tokenizer_json_sha256': hashlib.sha256((ROOT / args.tokenizer / 'tokenizer.json').read_bytes()).hexdigest()},
        'max_total_tokens': args.max_total_tokens,
        'max_answer_tokens': args.max_answer_tokens, 'files': files,
        'kept_by_task': {s: dict(Counter(r['task_type'] for r in rows)) for s, rows in kept.items()},
        'answer_token_share': {s: {t: round(v / max(1, sum(c.values())), 3) for t, v in c.items()} for s, c in tokens.items()},
        'held_reasons': dict(held_reasons), 'held_by_split_task': {f'{s}:{t}': v for (s, t), v in held_by_task.items()},
        'contract': 'case adapter (CASE_SYSTEM + rag_prompt D1..Dn): every kept target cites [Dn]; numbers are in the cited '
                    'passage, the question or the provenance line; no-evidence behaviours start with 확인 불가; answer '
                    f'<= {args.max_answer_tokens} tokens. Development training only; AI review only; nothing is labelled '
                    'current law.'}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    a_manifest = {'id': f'{out.name}-a', 'source': f'{args.out} (B, MODEL-REPAIR-001)',
                  'train': [{'path': 'train.jsonl', 'sha256': files['train']['sha256'], 'adapter': 'case', 'split': 'train', 'epochs': 2}],
                  'valid': [{'path': 'dev.jsonl', 'sha256': files['dev']['sha256'], 'adapter': 'case', 'split': 'pilot_dev'}]}
    (out / 'a-train-manifest.json').write_text(json.dumps(a_manifest, ensure_ascii=False, indent=1))
    print(json.dumps({k: manifest[k] for k in ('files', 'kept_by_task', 'answer_token_share', 'held_reasons', 'held_by_split_task')},
                     ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
