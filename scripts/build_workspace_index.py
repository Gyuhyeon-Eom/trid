"""Build the test site's shared search index ("workspace"): approved public sources only, verbatim text, with
per-document provenance (institution, document title, kind, issue/effective date, currency note, source URL).

Inputs (pinned by sha256 in the output manifest):
  - B's cleaned law.go.kr corpus (administrative rules and legal interpretations; KR Copyright Act Art. 7 works),
    `split == train` rows only (B's valid/test rows stay out of a public site). Quarantine and quality-held files are
    separate files and are not read.
  - The 20 KOGL Type 1 policy-briefing press releases already shown on the site.
Passages come from trid.service.structure.parse_document (조/항 aware, verbatim slices) and are indexed with the
service BM25 (trid.service.index.Index). No PII redaction is applied here: the texts are published public documents
whose B cleaning already recorded redactions; display must match the source verbatim.

  python scripts/build_workspace_index.py --out runs/workspace-index/v1
"""
import argparse
import hashlib
import html
import json
import re
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trid.service.index import Index  # noqa: E402
from trid.service.structure import parse_document  # noqa: E402
from trid.service.types import Document  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RULES = ROOT / 'data/internal-rules/v3/docs.jsonl'
PRESS = ROOT / 'data/public-document-sft-source/documents.jsonl'
PRACTICE = ROOT / 'data/search-corpus-v2'
KIND = {'administrative_rule': '행정규칙', 'legal_interpretation': '법령해석례'}


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

API_KIND = {'rule': '행정규칙', 'manual': '업무편람', 'law': '법령', 'interpretation': '법령해석·질의회신', 'faq': 'FAQ',
            'qa_board': '질의응답 게시판', 'statistic': '통계·공시', 'press': '정책브리핑 보도자료'}


def api_snapshot(directory, meta):
    """(documents, structured facts, snapshot info) of one local snapshot copy. Every file listed in manifest.json must
    match its sha256; records with usage restrictions are skipped; the issuer is the document's institution and
    `applies_to` is kept separately (an issuer is not necessarily the body the rule applies to).
    A record already indexed from another input is skipped (`duplicate`): its document id is
    `<source_id prefix>:<version>` — for law.go.kr DRF records source_id carries the rule ID and version the serial
    number (일련번호) that the law.go.kr corpus already uses as document id — or its source_id itself."""
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for f in manifest['files']:
        if sha256(directory / f['path']) != f['sha256']:
            raise ValueError(f"{directory / f['path']}: sha256 differs from the snapshot manifest")
    documents, facts, skipped, added = [], [], {'restricted': 0, 'empty': 0, 'duplicate': 0}, []
    updated = []
    for f in manifest['files']:
        if not f['path'].startswith('normalized/'):
            continue
        for line in (directory / f['path']).read_text().splitlines():
            row = json.loads(line)
            if (row.get('rights') or {}).get('restrictions'):
                skipped['restricted'] += 1
                continue
            doc_id = f"{row['source_id'].rsplit(':', 1)[0]}:{row['version']}"
            existing = doc_id if doc_id in meta else row['source_id'] if row['source_id'] in meta else None
            if existing:
                # Same text already indexed from another input: keep that document, but carry the snapshot's metadata
                # that the other input lacks (who the rule applies to, issuer code), so deduplication loses nothing.
                skipped['duplicate'] += 1
                issuer = row.get('issuer') or {}
                meta[existing].update({k: v for k, v in (('applies_to', row.get('applies_to')),
                                                          ('issuer_id', issuer.get('id')),
                                                          ('issuer_id_scheme', issuer.get('id_scheme')),
                                                          ('source_id', row['source_id']),
                                                          ('metadata_snapshot_id', manifest['snapshot_id'])) if v})
                updated.append(existing)
                continue
            for fact in row.get('structured_facts') or []:
                facts.append({**fact, 'source_id': row['source_id'], 'version': row['version'],
                              'source_url': (row.get('location') or {}).get('url')})
            if not (row.get('text') or '').strip():
                skipped['empty'] += 1
                continue
            doc = Document(doc_id, row['title'], row['content_hash'], {'public'}, row['text'])
            doc.passages, doc.refs = parse_document(row['text'], doc_id, row['title'])
            documents.append(doc)
            added.append(doc_id)
            dates, kinds, issuer = row.get('dates') or {}, row.get('date_kinds') or {}, row.get('issuer') or {}
            shown = 'published' if dates.get('published') else 'effective_from'
            meta[doc_id] = {'title': row['title'], 'kind': API_KIND.get(row['kind'], row['kind']),
                            'institution': issuer.get('name'), 'issuer_id': issuer.get('id'),
                            'issuer_id_scheme': issuer.get('id_scheme'), 'applies_to': row.get('applies_to'),
                            'date': dates.get(shown), 'date_type': kinds.get(shown, shown),
                            'effective_date': dates.get('effective_from'),
                            'currency': f"{dates.get('retrieved_at')} 수집 당시 자료이며 현행 여부는 다시 확인하지 않았음",
                            'source': (row.get('location') or {}).get('url'), 'license': (row.get('rights') or {}).get('license'),
                            'attribution': manifest.get('producer'), 'source_id': row['source_id'], 'version': row['version'],
                            'snapshot_id': manifest['snapshot_id']}
    info = {'snapshot_id': manifest['snapshot_id'], 'manifest_sha256': sha256(manifest_path), 'indexed': len(documents),
            'added_documents': added, 'metadata_updated': len(updated), 'skipped': skipped, 'facts': len(facts),
            'producer': manifest.get('producer')}
    return documents, facts, info


GUIDE_MIN_CHARS = 500


def figure_limit(row):
    """The extraction limit B records for figures it could not extract (flag figure_content_missing,
    meta.figures_omitted = number of [그림 생략] markers), or None."""
    n = (row.get('meta') or {}).get('figures_omitted') or 0
    if not n and 'figure_content_missing' not in (row.get('flags') or []):
        return None
    return (f"원본의 그림{f' {n:,}곳' if n else ''}은 본문에 [그림 생략]으로만 남아 있고 그림 내용은 추출되지 않았습니다. "
            '그림·표를 원본과 대조해 검증하지 않았습니다.')


def guides(path, meta, tables):
    """(documents, info) of B's cleaned practice guides (data/public-data/guides-vN/documents.jsonl). B's `documents`
    part already passed its rights / PII / reserved-overlap checks (held rows are in held.jsonl and never read here);
    rows with a status hold, near-empty extractions (covers, separators: < GUIDE_MIN_CHARS) and rows without an
    unrestricted or KOGL Type 1 licence are still left out. A figure-extraction limit (figure_limit) becomes a
    `limitations` note shown with every answer from the document. Table passages are flagged (B: table_check
    unverified)."""
    documents, skipped = [], {}
    for line in (path / 'documents.jsonl').read_text().splitlines():
        row = json.loads(line)
        m = row.get('meta') or {}
        license_ = row.get('license') or ''
        why = ('hold' if m.get('status_holds') else 'empty_or_cover' if len(row['text'].strip()) < GUIDE_MIN_CHARS else
               'license' if not any(k in license_ for k in ('제한 없음', '1유형', 'KOGL-Type1')) else None)
        if why:
            skipped.setdefault(why, []).append(row['id'])
            continue
        prov = row.get('provenance') or {}
        doc = Document(row['id'], row['title'], row['sha256'], {'public'}, row['text'])
        doc.passages, doc.refs = parse_document(row['text'], row['id'], row['title'])
        documents.append(doc)
        limit = figure_limit(row)
        meta[row['id']] = {'title': row['title'], 'kind': prov.get('doc_type') or '업무매뉴얼', 'institution': prov.get('institution'),
                           'date': prov.get('date'), 'date_type': prov.get('date_type'), 'effective_date': None,
                           'currency': '자료 기준일 당시 내용이며 현행 여부는 확인하지 않았음', 'source': row.get('source'),
                           'license': license_, 'attribution': row.get('collected_at'), 'extractor': m.get('extractor'),
                           'applies_to': {'scope': 'institution', 'note': prov.get('applies_to')}, 'table_check': m.get('table_check'),
                           'limitations': [limit] if limit else []}
        if m.get('table_check') == 'unverified':
            tables += [p.id for p in doc.passages if '|' in p.text or '<table' in p.text or '<tr' in p.text]
    manifest = path / 'manifest.json'
    return documents, {'path': str(path), 'manifest_sha256': sha256(manifest) if manifest.exists() else None,
                       'indexed': [d.doc_id for d in documents], 'skipped': skipped,
                       'limited': [d.doc_id for d in documents if meta[d.doc_id]['limitations']]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--practice', action='store_true', help=f'also index B search corpus {PRACTICE.relative_to(ROOT)}')
    p.add_argument('--api-snapshot', type=Path, action='append', default=[],
                   help='also index a local copy of a B API-S3-001 snapshot (reports/coordination/HYBRID-RAG-001-SCHEMA.md)')
    p.add_argument('--guides', type=Path, action='append', default=[],
                   help="also index B's cleaned practice guides (data/public-data/guides-vN); see guides()")
    a = p.parse_args()
    if a.out.exists():
        p.error('Use a new --out directory')
    started = time.time()
    documents, meta = [], {}
    for line in RULES.read_text().splitlines():
        row = json.loads(line)
        if row.get('split') != 'train':
            continue
        raw = row['meta'].get('raw_fields', {})
        if row['kind'] == 'legal_interpretation':  # the interpreting body answers; the querier only asked
            institution = raw.get('해석기관명', '법제처') + (f"(질의: {raw['질의기관명']})" if raw.get('질의기관명') else '')
        else:
            institution = raw.get('소관부처명') or row.get('provenance', {}).get('institution')
        doc = Document(row['id'], row['title'], row['sha256'], {'public'}, row['text'])
        doc.passages, doc.refs = parse_document(row['text'], row['id'], row['title'])
        documents.append(doc)
        meta[row['id']] = {'title': row['title'], 'kind': KIND.get(row['kind'], row['kind']), 'institution': institution,
                           'date': row.get('date'), 'date_type': row['meta'].get('date_type'),
                           'effective_date': raw.get('시행일자'), 'currency': row['meta'].get('currency'),
                           'source': row['source'], 'license': row['license'], 'attribution': row['attribution']}
    for line in PRESS.read_text().splitlines():
        row = json.loads(line)
        if row['license'] != 'KOGL-Type1':
            continue

        title = html.unescape(row['title'])
        doc = Document(row['id'], title, hashlib.sha256(row['text'].encode()).hexdigest(), {'public'}, row['text'])
        doc.passages, doc.refs = parse_document(row['text'], row['id'], title)
        documents.append(doc)
        agency = re.search(r'이 자료는 (.+?)의 보도자료', row.get('attribution') or '')
        meta[row['id']] = {'title': title, 'kind': '정책브리핑 보도자료', 'institution': agency[1] if agency else None,
                           'date': row['date'], 'date_type': '발표일', 'effective_date': None,
                           'currency': '발표 당시 자료이며 현재 적용 여부는 확인하지 않았음', 'source': row['source'],
                           'license': 'KOGL-Type1', 'attribution': row.get('attribution')}
    tables = []
    if a.practice:
        practice_docs = {}
        skipped = {'no_searchable_passage': 0, 'usage_restriction': 0}
        for line in (PRACTICE / 'documents.jsonl').read_text().splitlines():
            row = json.loads(line)
            if row['search_status'] != 'public_search_ok':
                skipped['no_searchable_passage'] += 1  # B: documents with no searchable passage are left out entirely
                continue
            if row.get('usage_restriction'):
                skipped['usage_restriction'] += 1  # B: not a basis for a present-tense answer (time-bound values, repeals)
                continue
            practice_docs[row['id']] = Document(row['id'], row['title'], row['text_sha256'], {'public'}, '')
            meta[row['id']] = {'title': row['title'], 'kind': row['doc_type'], 'institution': row['institution'],
                               'date': row['base_date'], 'date_type': row['date_type'], 'effective_date': None,
                               'currency': '기준일 당시 자료이며 현재 값·현행 여부는 확인하지 않았음',
                               'source': row['source_url'], 'license': row['license'],
                               'attribution': row.get('collection'), 'table_check': row.get('table_check'),
                               'applies_to': row.get('applies_to')}
        from trid.service.types import Passage
        import ast
        for line in (PRACTICE / 'passages.jsonl').read_text().splitlines():
            row = json.loads(line)
            doc = practice_docs.get(row['doc_id'])
            if doc is None:
                continue
            heading = row['heading_path']
            heading = ast.literal_eval(heading) if isinstance(heading, str) else heading
            doc.passages.append(Passage(row['passage_id'], row['doc_id'], row['text'], int(row['start']), int(row['end']),
                                        (doc.title, *heading)))
            if str(row.get('has_table')) == 'True' or meta[row['doc_id']]['table_check'] == 'unverified':
                tables.append(row['passage_id'])
        documents += [d for d in practice_docs.values() if d.passages]
    facts, snapshots = [], {}
    for snap in a.api_snapshot:
        docs, rows_facts, snapshots[str(snap)] = api_snapshot(snap, meta)
        documents += docs
        facts += rows_facts
    guide_info = []
    for path in a.guides:
        docs, info = guides(path, meta, tables)
        documents += docs
        guide_info.append(info)
    index = Index.build(documents)
    index.save(a.out)
    meta = {d.doc_id: meta[d.doc_id] for d in documents}  # documents actually indexed only
    counts = {}
    for m in meta.values():
        counts[m['kind']] = counts.get(m['kind'], 0) + 1
    dates = sorted(m['date'] for m in meta.values() if m['date'])
    inputs = {str(RULES.relative_to(ROOT)): sha256(RULES), str(PRESS.relative_to(ROOT)): sha256(PRESS)}
    if a.practice:
        inputs.update({str((PRACTICE / f).relative_to(ROOT)): sha256(PRACTICE / f) for f in ('documents.jsonl', 'passages.jsonl', 'manifest.json')})
    inputs.update({f'{s}/manifest.json': info['manifest_sha256'] for s, info in snapshots.items()})
    inputs.update({f"{g['path']}/manifest.json": g['manifest_sha256'] for g in guide_info})
    manifest = {'built_at_unix': time.time(), 'seconds': round(time.time() - started, 1), 'inputs': inputs,
                'documents': len(documents), 'passages': len(index.passages), 'by_kind': counts,
                'table_passages': len(tables), 'date_range': [dates[0], dates[-1]] if dates else None,
                'selection': 'law.go.kr rows with split == train (B valid/test excluded); KOGL Type 1 press releases'
                             + ('; B search-corpus-v2 rows with search_status public_search_ok and no usage_restriction, '
                                f'B passages as given (skipped {skipped})' if a.practice else '')
                             + ''.join(f"; API snapshot {info['snapshot_id']} ({info['indexed']} records, skipped "
                                       f"{info['skipped']})" for info in snapshots.values()),
                'api_snapshots': snapshots, 'guides': guide_info, 'structured_facts': len(facts),
                'note': 'Search-only corpus for the test site. Not a training-approval list; training holds are separate.'}
    if facts:
        (a.out / 'facts.jsonl').write_text(''.join(json.dumps(f, ensure_ascii=False) + '\n' for f in facts))
    (a.out / 'tables.json').write_text(json.dumps(sorted(tables)))
    (a.out / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False))
    (a.out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == '__main__':
    main()
