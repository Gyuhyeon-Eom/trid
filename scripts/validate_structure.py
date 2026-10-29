"""Read-only bounded snapshot and reproducible actual-law structure diagnostics."""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trid.service.structure import H_JO, R_JO, R_EXT, R_ANNEX, is_regulation, parse_document


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def snapshot(source, output, per_kind=1000):
    counts = collections.Counter()
    size = source.stat().st_size
    seen = set()
    with source.open('rb') as stream, output.open('wb') as dest:
        while stream.tell() < size:
            line = stream.readline()
            if stream.tell() > size or not line.endswith(b'\n'):
                break
            row = json.loads(line)
            kind = row['kind']
            if kind not in ('law', 'admrul', 'ordin') or counts[kind] >= per_kind or row['id'] in seen:
                continue
            seen.add(row['id'])
            dest.write(line)
            counts[kind] += 1
    return dict(counts)


def analyze(rows):
    stats = collections.Counter(exceptions=0)
    lengths, findings = [], []
    for row in rows:
        text = row['text']
        stats['documents'] += 1
        stats['kind_' + row['kind']] += 1
        stats['detected_regulations'] += is_regulation(text)
        stats['detected_' + row['kind']] += is_regulation(text)
        try:
            passages, edges = parse_document(text, row['id'], row['title'])
        except Exception as exc:
            stats['exceptions'] += 1
            findings.append(dict(id=row['id'], category='exception', excerpt=str(exc)))
            continue
        stats['passages'] += len(passages)
        stats['resolved_local_edges'] += len(edges)
        stats['external_mentions'] += len(list(R_EXT.finditer(text)))
        labels = {x for p in passages for x in p.path[1:]}
        for p in passages:
            lengths.append(len(p.text))
            stats['invalid_spans'] += text[p.char_start:p.char_end] != p.text
            stats['over_limit'] += len(p.text) > 550
        # Candidate failures, not adjudicated semantic errors. Save source context for inspection.
        candidates = []
        if not is_regulation(text):
            candidates.append(('fallback', text[:350]))
        for m in R_JO.finditer(R_EXT.sub(lambda m: ' ' * len(m.group()), text)):
            label = '제' + m[1] + '조' + ('의' + m[2] if m[2] else '')
            stats['local_article_mentions'] += 1
            if label not in labels:
                stats['unresolved_article_mentions'] += 1
                candidates.append(('unresolved_article', text[max(0, m.start()-60):m.end()+120]))
        for line in text.splitlines():
            if '별표' in line and len(line) < 140:
                candidates.append(('appendix_review', line))
            if line.startswith(('2.', '3.', '1의', '2의')):
                candidates.append(('list_review', line[:220]))
            if '조의' in line:
                candidates.append(('subarticle_review', line[:220]))
            if '」' in line and (' 및 제' in line or '부터 제' in line):
                candidates.append(('external_chain_review', line[:300]))
        for category, excerpt in candidates:
            if sum(f['category'] == category for f in findings) < 10:
                findings.append(dict(id=row['id'], category=category, excerpt=excerpt, source=row['source']))
    lengths.sort()
    return dict(stats=stats, lengths=dict(min=min(lengths), median=lengths[len(lengths)//2], p95=lengths[int(len(lengths)*.95)], max=max(lengths)), findings=findings)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, default=Path('data/law-ko-source/docs.jsonl.part'))
    ap.add_argument('--snapshot', type=Path, default=Path('/tmp/trid-law-structure-3000.jsonl'))
    ap.add_argument('--reuse-snapshot', action='store_true')
    ap.add_argument('--output', type=Path, default=Path('/tmp/trid-structure-diagnostics.json'))
    ap.add_argument('--manifest', type=Path, help='Restore exact rows from original collected source using provenance.json')
    args = ap.parse_args()
    if args.manifest and not args.reuse_snapshot:
        manifest = json.loads(args.manifest.read_text())
        expected = {r['id']:r['sha256'] for r in manifest['document_manifest']}
        found = {}
        boundary = args.source.stat().st_size
        with args.source.open('rb') as stream:
            while stream.tell() < boundary:
                line = stream.readline()
                if stream.tell() > boundary or not line.endswith(b'\n'):
                    break
                row = json.loads(line)
                if row['id'] in expected and digest(row['text']) == expected[row['id']]:
                    found.setdefault(row['id'], line)
        if len(found) != len(expected):
            raise ValueError(f'Missing hash-matching source rows: {len(expected)-len(found)}')
        content = b''.join(found[r['id']] for r in manifest['document_manifest'])
        if hashlib.sha256(content).hexdigest() != manifest['snapshot_sha256']:
            raise ValueError('Source metadata/serialization changed; cannot reproduce exact snapshot')
        args.snapshot.write_bytes(content)
    elif not args.reuse_snapshot:
        print('snapshot', snapshot(args.source, args.snapshot), flush=True)
    rows = [json.loads(line) for line in args.snapshot.open()]
    result = analyze(rows)
    result['snapshot_sha256'] = hashlib.sha256(args.snapshot.read_bytes()).hexdigest()
    result['snapshot_path'] = str(args.snapshot)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != 'findings'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
