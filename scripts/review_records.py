"""Review records for free-text evaluation outputs: export -> blinded packet -> verdict import -> aggregate.

export     One record per (model, item) from stored raw outputs (scripts/train_public_dialogue.py evaluate,
           scripts/evaluate_external_chat.py --freeform-prompts), the prompt file and a checklist fixed before
           generation. Writes records.jsonl (with model/run), a blinded packet.jsonl (random codes, shuffled, no model
           or run fields) and mapping.json (code -> record, file hashes). Every output id must match the prompt set.
import     Verdict rows keyed by packet code. Rejected: unknown code, a packet hash from another export, a reviewer
           not in the registry or declared with a different type (an AI reviewer can never be recorded as human or
           expert), and a second verdict by the same reviewer on the same code.
aggregate  Per model: execution status counts over ALL items, and per reviewer type the reviewed/unreviewed/uncertain
           counts and verdict distribution. Unreviewed and uncertain are never scored as wrong or right; the rate is
           shown only over reviewed items, next to the coverage.

Objective checks (cited [Dn] ids exist in the prompt, numbers absent from the prompt) are auxiliary evidence only;
they never set a semantic verdict.
"""
import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

REVIEWER_TYPES = ('ai', 'human', 'expert')
CONCLUSION = ('정답', '부분', '오답', '보류', '불확실')
NUMBER = re.compile(r'\d[\d,.]*')


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_text(text):
    return sha256_bytes(text.encode())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def normalize_output(path):
    """Stored output file -> (model descriptor, {id: (response, finish)})."""
    d = json.loads(Path(path).read_text())
    if 'weights_sha256' in d:   # trid evaluate
        model = dict(kind='trid', run=d['run'], weights_sha256=d['weights_sha256'])
        rows = {}
        for r in d['results']:
            if r.get('exception'):
                finish = 'exception'
            elif r.get('skipped_context'):
                finish = 'input_exceeds'
            elif r.get('stopped_on_eos'):
                finish = 'eos'
            else:
                finish = 'limit'
            rows[r['id']] = (r.get('answer'), dict(finish=finish, context_limited=bool(r.get('context_limited')),
                                                  max_tokens_requested=r.get('max_tokens_requested'),
                                                  max_tokens_used=r.get('max_tokens_used'),
                                                  generated_tokens=r.get('generated_tokens_including_eos')))
    else:                        # evaluate_external_chat freeform
        model = dict(kind='llama.cpp', model_id=d['model_id'], revision=d['revision'], weights_sha256=d['model_sha256'])
        rows = {}
        for r in d['results']:
            if r.get('exception'):
                finish = 'exception'
            elif r.get('skipped_context'):
                finish = 'input_exceeds'
            else:
                finish = 'eos' if r.get('finish_reason') == 'eos' else 'limit'
            rows[r['id']] = (r.get('generated'), dict(finish=finish, context_limited=False,
                                                     max_tokens_requested=d.get('max_output_tokens'),
                                                     max_tokens_used=d.get('max_output_tokens'),
                                                     generated_tokens=r.get('output_tokens')))
    return model, rows


def auxiliary(prompt, response):
    if not response:
        return dict(cited_ids=[], cited_ids_in_prompt=None, numbers_not_in_prompt=[])
    cited = sorted(set(re.findall(r'\[?D(\d)\]?', response)))
    ids = set(re.findall(r'\[D(\d)\]', prompt))
    # Citation markers like [D2] or D2 are source ids, not values.
    numbers = [n for n in NUMBER.findall(re.sub(r'\[?D\d\]?', ' ', response)) if n.rstrip('.,') not in prompt]
    return dict(cited_ids=[f'D{c}' for c in cited], cited_ids_in_prompt=all(c in ids for c in cited) if cited else None,
                numbers_not_in_prompt=sorted(set(numbers)))


def export(outputs, prompts_path, checklist_path, families, out_dir, seed):
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise ValueError('Use a new output directory')
    prompts = {p['id']: p for p in json.loads(Path(prompts_path).read_text())['results']}
    checklist = {c['id']: c for c in json.loads(Path(checklist_path).read_text())['items']}
    if set(checklist) != set(prompts):
        raise ValueError('Checklist and prompt ids differ')
    hashes = dict(prompts_sha256=sha256_bytes(Path(prompts_path).read_bytes()),
                  checklist_sha256=sha256_bytes(Path(checklist_path).read_bytes()))
    records = []
    for name, path in outputs:
        model, rows = normalize_output(path)
        if set(rows) != set(prompts):
            raise ValueError(f'{name}: output ids differ from the prompt set ({len(set(prompts) - set(rows))} missing, '
                             f'{len(set(rows) - set(prompts))} unexpected)')
        output_sha = sha256_bytes(Path(path).read_bytes())
        for item_id in prompts:
            response, finish = rows[item_id]
            records.append(dict(record_id=sha256_text(f'{name}|{output_sha}|{item_id}')[:16], item_id=item_id,
                                doc_families=families.get(item_id, []), model=name, model_descriptor=model,
                                output_path=str(path), output_sha256=output_sha,
                                prompt_sha256=sha256_text(prompts[item_id]['prompt']), **hashes,
                                response=response, **finish, auxiliary=auxiliary(prompts[item_id]['prompt'], response)))
    rng = random.Random(seed)
    codes = rng.sample(range(10 ** 5, 10 ** 6), len(records))
    for record, code in zip(records, codes):
        record['packet_code'] = f'R{code}'
    packet = [dict(packet_code=r['packet_code'], item_id=r['item_id'], question=checklist[r['item_id']]['question'],
                   prompt=prompts[r['item_id']]['prompt'], checklist={k: v for k, v in checklist[r['item_id']].items()
                                                                      if k not in ('prompt_tokens_trid', 'generation_room_trid')},
                   response=r['response'], finish=r['finish'], auxiliary=r['auxiliary']) for r in records]
    rng.shuffle(packet)
    out_dir.mkdir(parents=True)
    write_jsonl(out_dir / 'records.jsonl', records)
    write_jsonl(out_dir / 'packet.jsonl', packet)
    mapping = dict(packet_sha256=sha256_bytes((out_dir / 'packet.jsonl').read_bytes()),
                   records_sha256=sha256_bytes((out_dir / 'records.jsonl').read_bytes()), **hashes,
                   codes={r['packet_code']: dict(record_id=r['record_id'], model=r['model'], item_id=r['item_id'])
                          for r in records})
    (out_dir / 'mapping.json').write_text(json.dumps(mapping, ensure_ascii=False, indent=2))
    return mapping


def validate_verdicts(verdicts, mapping, registry):
    seen, errors = set(), []
    for n, v in enumerate(verdicts):
        where = f"verdict {n} ({v.get('packet_code')})"
        if v.get('packet_sha256') != mapping['packet_sha256']:
            errors.append(f'{where}: packet hash belongs to another export')
        if v.get('packet_code') not in mapping['codes']:
            errors.append(f'{where}: unknown packet code')
        kind = v.get('reviewer_type')
        registered = registry.get(v.get('reviewer_id'))
        if kind not in REVIEWER_TYPES:
            errors.append(f'{where}: reviewer_type must be one of {REVIEWER_TYPES}')
        elif registered is None:
            errors.append(f"{where}: reviewer {v.get('reviewer_id')!r} is not registered")
        elif registered != kind:
            errors.append(f"{where}: reviewer {v.get('reviewer_id')!r} is registered as {registered}, not {kind}")
        if v.get('conclusion') not in CONCLUSION:
            errors.append(f'{where}: conclusion must be one of {CONCLUSION}')
        key = (v.get('packet_code'), v.get('reviewer_id'))
        if key in seen:
            errors.append(f'{where}: duplicate verdict by the same reviewer')
        seen.add(key)
    if errors:
        raise ValueError('; '.join(errors[:10]) + (f' (+{len(errors) - 10} more)' if len(errors) > 10 else ''))


def aggregate(records, verdicts, mapping):
    by_code = defaultdict(list)
    for v in verdicts:
        by_code[v['packet_code']].append(v)
    out = {}
    for model in sorted({r['model'] for r in records}):
        rows = [r for r in records if r['model'] == model]
        result = dict(items=len(rows), execution=dict(Counter(r['finish'] for r in rows)),
                      context_limited=sum(r['context_limited'] for r in rows), reviews={})
        for kind in REVIEWER_TYPES:
            verdict = {r['packet_code']: next((v for v in by_code[r['packet_code']] if v['reviewer_type'] == kind), None)
                       for r in rows}
            done = [v for v in verdict.values() if v and v['conclusion'] != '불확실']
            result['reviews'][kind] = dict(
                reviewed=len(done), uncertain=sum(bool(v) and v['conclusion'] == '불확실' for v in verdict.values()),
                unreviewed=sum(v is None for v in verdict.values()),
                conclusion=dict(Counter(v['conclusion'] for v in done)),
                correct_rate_over_reviewed=round(sum(v['conclusion'] == '정답' for v in done) / len(done), 4) if done else None,
                false_added_claim=sum(bool(v.get('false_added_claim')) for v in done),
                condition_or_exception_missing=sum(str(v.get('condition_or_exception_missing')).startswith('예') for v in done))
        out[model] = result
    return out


def convert_ai(judgements_path, export_dir, reviewer_id):
    """C-003 style AI judgements ({models: {name: {items: [...]}}}) -> verdict rows typed 'ai' (never human)."""
    mapping = json.loads((Path(export_dir) / 'mapping.json').read_text())
    code_of = {(v['model'], v['item_id']): code for code, v in mapping['codes'].items()}
    data = json.loads(Path(judgements_path).read_text())
    rows = []
    for model, block in data['models'].items():
        for item in block['items']:
            rows.append(dict(packet_code=code_of[(model, item['id'])], packet_sha256=mapping['packet_sha256'],
                             reviewer_id=reviewer_id, reviewer_type='ai', conclusion=item['conclusion'],
                             evidence=item['evidence'], condition_or_exception_missing=item['condition_or_exception_missing'],
                             false_added_claim=item['false_added_claim'], truncated_observed=item['truncated'],
                             format_mismatch=item['format_mismatch'], rationale=item['note'],
                             source=f'{judgements_path} (AI review, checklist fixed before generation)'))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='stage', required=True)
    e = sub.add_parser('export')
    e.add_argument('--output', action='append', required=True, help='name=path to stored raw outputs')
    e.add_argument('--prompts', type=Path, required=True)
    e.add_argument('--checklist', type=Path, required=True)
    e.add_argument('--families', type=Path, help='JSON {item_id: [document family ids]}')
    e.add_argument('--out-dir', type=Path, required=True)
    e.add_argument('--seed', type=int, default=20261030)
    g = sub.add_parser('aggregate')
    g.add_argument('--export-dir', type=Path, required=True)
    g.add_argument('--verdicts', type=Path, action='append', default=[])
    g.add_argument('--registry', type=Path, required=True, help='JSON {reviewer_id: ai|human|expert}')
    g.add_argument('--report', type=Path, required=True)
    c = sub.add_parser('convert-ai')
    c.add_argument('--judgements', type=Path, required=True)
    c.add_argument('--export-dir', type=Path, required=True)
    c.add_argument('--reviewer-id', required=True)
    c.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.stage == 'export':
        families = json.loads(a.families.read_text()) if a.families else {}
        mapping = export([x.split('=', 1) for x in a.output], a.prompts, a.checklist, families, a.out_dir, a.seed)
        print(json.dumps(dict(records=len(mapping['codes']), packet_sha256=mapping['packet_sha256']), ensure_ascii=False))
    elif a.stage == 'convert-ai':
        if a.out.exists():
            raise ValueError('Use a new verdict file')
        rows = convert_ai(a.judgements, a.export_dir, a.reviewer_id)
        write_jsonl(a.out, rows)
        print(f'{len(rows)} AI verdicts -> {a.out}')
    else:
        mapping = json.loads((a.export_dir / 'mapping.json').read_text())
        records = read_jsonl(a.export_dir / 'records.jsonl')
        if sha256_bytes((a.export_dir / 'packet.jsonl').read_bytes()) != mapping['packet_sha256']:
            raise ValueError('Packet file changed after export')
        verdicts = [v for path in a.verdicts for v in read_jsonl(path)]
        validate_verdicts(verdicts, mapping, json.loads(a.registry.read_text()))
        summary = aggregate(records, verdicts, mapping)
        a.report.write_text(json.dumps(dict(protocol=__doc__, export_dir=str(a.export_dir),
                                            packet_sha256=mapping['packet_sha256'],
                                            verdict_files=[dict(path=str(x), sha256=sha256_bytes(x.read_bytes())) for x in a.verdicts],
                                            models=summary), ensure_ascii=False, indent=2))
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
