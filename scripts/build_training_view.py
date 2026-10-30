"""Training view of fixed guide versions: what a training consumer may read. Source rows and raw files stay as they are.

From the exclusive candidate parts of guides-v6 / bojo-faq-v3 and the guides-v6-prep sidecar (bound by id + sha256):
  - rows whose sidecar has a training_hold are left out (listed in excluded.jsonl with their reasons);
  - spans that identify a person are replaced by a placeholder, located on the row text:
      reply_contact  a replying official named in a contact sentence: '조진석 전문위원, ☎[전화]', '김재진 전문위원(전화: …)'
                     -> '[회신 담당자]'
      asker_name     an asker naming himself: '… 청탁방지담당관 정학수입니다' -> '[질의자]입니다'
      petition_no    국민신문고 신청번호 '1AA-2506-0402461' -> '[민원 신청번호]'
    The sidecar's broader official_name spans also matched titles and grades ('행정사무관', '청탁방지담당관'); replacing
    those would change content, so they are not used here.
Consumer check (fails the build): no held id in the view; every row's sha256 binding matches; no reply_contact /
asker_name / petition_no pattern is left; outside the replaced spans the text is unchanged (so numbers, dates and
conditions are untouched, compared digit run by digit run).

  .venv/bin/python scripts/build_training_view.py --versions guides-v6 bojo-faq-v3 --prep guides-v6-prep --out guides-v6-train-view-v1
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()  # as loaded
SURNAMES = '김이박최정강조윤장임한오서신권황안송류홍전고문양손배백허유남심노하곽성차주우구민진나지엄채원천방공현함변염여추도소석선설마길연위표명기반왕금옥육인맹제모탁국어은편용예봉경'
TITLES = '전문위원|주무관|사무관|서기관|연구관|연구사|행정관|담당자'
REPLACE = {
    'reply_contact': (re.compile(rf'[가-힣]{{2,4}}\s?(?:{TITLES})(?=\s?(?:,|\()\s?(?:☎|☏|전화|\[전화\]|Tel|TEL))'), '[회신 담당자]'),
    # an official named by surname + given name with a title: '박정구 사무관님', '김선희 주무관 : ', '황성진 사무관입니다'
    'named_official': (re.compile(rf'(?<![가-힣])[{SURNAMES}][가-힣]{{1,2}}\s?(?:{TITLES})(?=님|께|에게|입니다|이라고|\s?:|\)|[\'’])'),
                       '[담당 공무원]'),
    'asker_name': (re.compile('(?:' + '|'.join(f'(?<={t} )' for t in ('담당관', '담당자', '주무관', '사무관', '과장', '팀장', '대리', '주임', '직원'))
                              + r')[가-힣]{2,4}(?=입니다)'), '[질의자]'),
    'petition_no': (re.compile(r'\b\d[A-Z]{2}-\d{4}-\d{6,7}\b'), '[민원 신청번호]'),
}
PHONE_FRAGMENT = re.compile(r'(?<![\d-])01[016789][-.\s]?\d{3,4}[-.\s]?[\d*＊]{1,4}')


def replace_spans(text):
    spans = sorted((m.start(), m.end(), kind) for kind, (rx, _) in REPLACE.items() for m in rx.finditer(text))
    merged = []
    for s, e, k in spans:
        if merged and s < merged[-1][1]:
            continue
        merged.append((s, e, k))
    out, pos = [], 0
    for s, e, k in merged:
        out.append(text[pos:s])
        out.append(REPLACE[k][1])
        pos = e
    out.append(text[pos:])
    return ''.join(out), merged


def outside(text, spans):
    """Text with the given spans cut out, for the unchanged-elsewhere check."""
    parts, pos = [], 0
    for s, e, _ in spans:
        parts.append(text[pos:s])
        pos = e
    parts.append(text[pos:])
    return parts


def main():
    from extract_public_guides import CANDIDATE_PARTS, OUT, check_version
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--versions', nargs='+', required=True)
    p.add_argument('--prep', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = OUT / args.out
    if (out / 'manifest.json').exists():
        raise SystemExit(f'{out} exists; versions are never overwritten')
    prep = {}
    for x in (OUT / args.prep / 'annotations.jsonl').read_text().split('\n'):
        if x:
            a = json.loads(x)
            prep[a['id']] = a
    kept, excluded, problems, counts = [], [], [], Counter()
    for v in args.versions:
        if check_version(v):
            raise SystemExit(f'{v} fails the consumer check')
        for part in CANDIDATE_PARTS:
            for x in (OUT / v / f'{part}.jsonl').read_text().split('\n'):
                if not x:
                    continue
                r = json.loads(x)
                a = prep.get(r['id'])
                if not a or a['sha256'] != r['sha256']:
                    problems.append(f"{r['id']}: no prep record bound to this sha256")
                    continue
                text, spans = replace_spans(r['text'])
                holds = list(a['training_hold'])
                # a partly masked mobile number the redactor did not catch ('010-3441-****'); checked after the
                # 신청번호 replacement so '1AA-2301-0162713' is not read as a phone number
                if PHONE_FRAGMENT.search(text):
                    holds.append('residual_phone_fragment')
                if holds:
                    excluded.append({'id': r['id'], 'sha256': r['sha256'], 'reasons': holds})
                    counts.update(f'excluded:{h}' for h in holds)
                    continue
                placeholders = '|'.join(re.escape(ph) for _, ph in REPLACE.values())
                src_parts, view_parts = outside(r['text'], spans), re.split(placeholders, text)
                if len(src_parts) != len(view_parts) or any(sp != vp for sp, vp in zip(src_parts, view_parts)):
                    problems.append(f"{r['id']}: text outside replaced spans changed")
                for kind, (rx, _) in REPLACE.items():
                    if rx.search(text):
                        problems.append(f"{r['id']}: {kind} left after replacement")
                counts.update(f'replaced:{k}' for _, _, k in spans)
                counts['rows_with_replacement'] += bool(spans)
                kept.append({'id': r['id'], 'source_sha256': r['sha256'], 'version': v, 'part': part, 'kind': r['kind'],
                             'title': r['title'], 'text': text, 'sha256': hashlib.sha256(text.encode()).hexdigest(),
                             'source': r['source'], 'license': r['license'], 'provenance': r['provenance'],
                             'faq_kind': r['meta'].get('faq_kind'), 'table_check': r['meta'].get('table_check'),
                             'replaced': [[s, e, k] for s, e, k in spans], 'prep': {k: a[k] for k in ('dup_answer', 'dup_near', 'year')}})
    if problems:
        raise SystemExit('training view check failed:\n' + '\n'.join(problems[:20]))
    out.mkdir(parents=True)
    body = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in kept).encode()
    (out / 'rows.jsonl').write_bytes(body)
    ex = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in excluded).encode()
    (out / 'excluded.jsonl').write_bytes(ex)
    manifest = {'version': args.out, 'code_sha256': CODE_SHA256, 'source_versions': args.versions, 'prep': args.prep,
                'prep_summary_sha256': hashlib.sha256((OUT / args.prep / 'summary.json').read_bytes()).hexdigest(),
                'rows': len(kept), 'excluded': len(excluded), 'rows_sha256': hashlib.sha256(body).hexdigest(),
                'excluded_sha256': hashlib.sha256(ex).hexdigest(), 'counts': dict(counts),
                'by_faq_kind': dict(Counter(r['faq_kind'] for r in kept)),
                'by_institution': dict(Counter(r['provenance']['institution'] for r in kept)),
                'documents_with_tables_unverified': sum(1 for r in kept if r['kind'] == 'guide_document'),
                'consumer_check': 'ok',
                'note': 'view for a training consumer; not a training approval; documents keep unverified tables'}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(json.dumps(manifest, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
