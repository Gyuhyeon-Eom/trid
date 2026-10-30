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
KIND = {'administrative_rule': '행정규칙', 'legal_interpretation': '법령해석례'}


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True)
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
    index = Index.build(documents)
    index.save(a.out)
    counts = {}
    for m in meta.values():
        counts[m['kind']] = counts.get(m['kind'], 0) + 1
    dates = sorted(m['date'] for m in meta.values() if m['date'])
    manifest = {'built_at_unix': time.time(), 'seconds': round(time.time() - started, 1),
                'inputs': {str(RULES.relative_to(ROOT)): sha256(RULES), str(PRESS.relative_to(ROOT)): sha256(PRESS)},
                'documents': len(documents), 'passages': len(index.passages), 'by_kind': counts,
                'date_range': [dates[0], dates[-1]] if dates else None,
                'selection': 'law.go.kr rows with split == train (B valid/test excluded); KOGL Type 1 press releases',
                'note': 'Search-only corpus for the test site. Not a training-approval list; training holds are separate.'}
    (a.out / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False))
    (a.out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == '__main__':
    main()
