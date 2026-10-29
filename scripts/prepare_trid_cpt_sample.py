"""Token-budgeted continued-pretraining sample from data/mix-v3-dry for a given tokenizer (trid on Kanana base).

Budgets are counted in the target tokenizer's tokens (document text only; BOS/EOS added by the trainer).
Documents are visited in a seeded random order per group and taken whole until the group budget is met.
Evaluation-source documents (institution probe, public-document SFT/held-out sources) are excluded by id and
by exact text hash so continued pretraining never sees them.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {'press': ('press_release',), 'law': ('law', 'admrul', 'ordin'), 'wiki': ('wikipedia',)}
EXCLUDE_SOURCES = (ROOT / 'data/institution-qa-frozen-01/source-docs.jsonl',
                   ROOT / 'data/public-document-sft-source/documents.jsonl')


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def file_digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def excluded():
    ids, hashes = set(), set()
    for path in EXCLUDE_SOURCES:
        for line in path.read_text().splitlines():
            doc = json.loads(line)
            ids.add(doc['id'])
            hashes.add(digest(doc['text']))
    return ids, hashes


def sample(path, tok, budgets, rng, skip_ids, skip_hashes):
    by_group = {g: [] for g in budgets}
    kind_group = {k: g for g, kinds in GROUPS.items() for k in kinds}
    with open(path, 'rb') as f:
        offset = 0
        for line in f:
            group = kind_group.get(json.loads(line)['kind'])
            if group in by_group:
                by_group[group].append(offset)
            offset += len(line)
    out, stats = [], {}
    with open(path, 'rb') as f:
        for group, budget in budgets.items():
            order = by_group[group][:]
            rng.shuffle(order)
            tokens = docs = skipped = 0
            for offset in order:
                if tokens >= budget:
                    break
                f.seek(offset)
                doc = json.loads(f.readline())
                if doc['id'] in skip_ids or digest(doc['text']) in skip_hashes:
                    skipped += 1
                    continue
                n = len(tok.encode(doc['text']).ids)
                out.append({'id': doc['id'], 'kind': doc['kind'], 'text': doc['text'], 'tokens': n})
                tokens += n
                docs += 1
            stats[group] = {'budget': budget, 'tokens': tokens, 'documents': docs, 'excluded_eval_sources': skipped,
                            'available_documents': len(order)}
    rng.shuffle(out)
    return out, stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokenizer', type=Path, required=True, help='tokenizer.json of the base model')
    p.add_argument('--source', type=Path, default=ROOT / 'data/mix-v3-dry')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--press', type=int, default=35_000_000)
    p.add_argument('--law', type=int, default=40_000_000)
    p.add_argument('--wiki', type=int, default=25_000_000)
    p.add_argument('--valid-docs', type=int, default=600)
    p.add_argument('--seed', type=int, default=291029)
    args = p.parse_args()
    if args.output.exists():
        p.error('Use a new output directory')
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(args.tokenizer))
    rng = random.Random(args.seed)
    skip_ids, skip_hashes = excluded()
    train, stats = sample(args.source / 'train.jsonl', tok, {'press': args.press, 'law': args.law, 'wiki': args.wiki},
                          rng, skip_ids, skip_hashes)
    per_group = max(1, args.valid_docs // len(GROUPS))
    valid, valid_stats = sample(args.source / 'valid.jsonl', tok, {g: 10**12 for g in GROUPS}, rng, skip_ids, skip_hashes)
    kept = []
    for group, kinds in GROUPS.items():
        kept += [d for d in valid if d['kind'] in kinds][:per_group]
    args.output.mkdir(parents=True)
    for name, rows in (('train', train), ('valid', kept)):
        (args.output / f'{name}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    provenance = {
        'purpose': 'trid continued pretraining sample (Kanana base tokenizer budgets)',
        'source': str(args.source.relative_to(ROOT)), 'source_train_sha256': file_digest(args.source / 'train.jsonl'),
        'source_valid_sha256': file_digest(args.source / 'valid.jsonl'),
        'tokenizer': str(args.tokenizer), 'tokenizer_sha256': file_digest(args.tokenizer), 'seed': args.seed,
        'groups': GROUPS, 'train': stats, 'valid_documents': len(kept),
        'valid_tokens': sum(d['tokens'] for d in kept),
        'excluded_sources': [str(p.relative_to(ROOT)) for p in EXCLUDE_SOURCES],
        'hashes': {name: file_digest(args.output / f'{name}.jsonl') for name in ('train', 'valid')},
        'license': 'see source ATTRIBUTION.txt (law.go.kr public works, KOGL Type 1 press releases, Wikipedia CC BY-SA)',
    }
    (args.output / 'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: provenance[k] for k in ('train', 'valid_documents', 'valid_tokens')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
