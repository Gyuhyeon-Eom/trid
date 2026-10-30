"""Training-preparation view of fixed guide versions, written beside them; the versions themselves are not touched.

For every candidate row of the given exclusive versions (guides-v6, bojo-faq-v3): a sidecar record bound to the row by
id + sha256 with
  dup_exact      cluster id for rows whose normalized question+answer is identical
  dup_answer     cluster id for rows that share one normalized answer (template answers to different questions)
  dup_near       cluster id for rows whose question+answer char 5-gram Jaccard >= 0.8 (MinHash 64, LSH 16x4, verified)
  spans          character spans (start, end, kind) of identifying text that redaction does not cover:
                 official_name (name + 주무관/사무관/전문위원/... in answers), petition_no (국민신문고 신청번호),
                 named_company ((주)/㈜/주식회사 + name, in questions), masked_contact (runs of '*' left by the source)
  personal_situation  question describes the asker's own family event, health or residence (heuristic)
  training_hold  reasons the row should not enter training as is (raw and the row stay as they are)
plus a summary: distribution by dataset, institution, institution type, year, faq_kind, length, and cluster sizes.

  .venv/bin/python scripts/prepare_guides_training_view.py --versions guides-v6 bojo-faq-v3 --out guides-v6-prep
"""
import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()  # as loaded
SPANS = {
    'official_name': re.compile(r'[가-힣]{2,3}\s?(?:주무관|사무관|전문위원|서기관|연구관|연구사|행정관|담당관)(?=[\s(,.]|$)'),
    'petition_no': re.compile(r'\b\d[A-Z]{2}-\d{4}-\d{6,7}\b'),
    'named_company': re.compile(r'(?:\(주\)|㈜|주식회사)\s?[가-힣A-Za-z0-9]{2,}'),
    'masked_contact': re.compile(r'\*{5,}'),
}
PERSONAL = re.compile(r'(?:제|저희|우리)\s?(?:아들|딸|자녀|배우자|남편|아내|부모님?|어머니|아버지|장인|장모|시부모)|'
                      r'(?:돌잔치|결혼식|장례|병원에 입원|수술|임신|출산)')
# board replies that answer nothing: scope notice, moderation notice, "answered through 국민신문고 instead"
NON_ANSWER = re.compile(r'본 게시판은 [^.]{0,40}답변드리는 곳|삭제될 수 있|국민신문고로 답변드리겠습니다|국민신문고를 이용하여 주시기')
SAME_ANSWER_KEEP = 3  # rows kept per same-answer cluster; later rows repeat one answer to different questions


def norm(text):
    return re.sub(r'[\W_]+', '', text).lower()


def year_of(r):
    m = r['meta']
    for v in (m.get('row_date'), (m.get('fields') or {}).get('자료발간연도'), m.get('updated'), m.get('registered'),
              r['provenance'].get('date')):
        if v and re.match(r'\d{4}', str(v)):
            return str(v)[:4]
    return 'unknown'


def minhash_clusters(rows, k=5, perms=64, bands=16, threshold=0.8):
    rnd = random.Random(19)
    salts = [rnd.getrandbits(64) for _ in range(perms)]
    shingles, sigs = [], []
    for r in rows:
        t = norm(r['text'])
        sh = {t[i:i + k] for i in range(max(1, len(t) - k + 1))}
        shingles.append(sh)
        hs = [int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), 'big') for s in sh]  # stable across runs
        sigs.append([min((h ^ salt) & 0xFFFFFFFFFFFFFFFF for h in hs) for salt in salts])
    rows_per_band = perms // bands
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = set()
    for b in range(bands):
        buckets = defaultdict(list)
        for i, sig in enumerate(sigs):
            buckets[tuple(sig[b * rows_per_band:(b + 1) * rows_per_band])].append(i)
        for members in buckets.values():
            for a in range(len(members)):
                for c in range(a + 1, len(members)):
                    i, j = members[a], members[c]
                    if (i, j) in seen:
                        continue
                    seen.add((i, j))
                    inter = len(shingles[i] & shingles[j])
                    if inter / len(shingles[i] | shingles[j]) >= threshold:
                        parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)
    return {i: g for g in groups.values() if len(g) > 1 for i in g}


def main():
    from extract_public_guides import CANDIDATE_PARTS, OUT, check_version
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--versions', nargs='+', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = OUT / args.out
    if (out / 'summary.json').exists():
        raise SystemExit(f'{out} exists; versions are never overwritten')
    rows = []
    for v in args.versions:
        if check_version(v):
            raise SystemExit(f'{v} fails the consumer check')
        for part in CANDIDATE_PARTS:
            for x in (OUT / v / f'{part}.jsonl').read_text().split('\n'):
                if x:
                    r = json.loads(x)
                    r['_version'], r['_part'] = v, part
                    rows.append(r)
    exact, answer = defaultdict(list), defaultdict(list)
    for i, r in enumerate(rows):
        q, _, a = r['text'].partition('\n답변: ')
        exact[hashlib.sha256(norm(r['text']).encode()).hexdigest()].append(i)
        if r['kind'] == 'qa_item' and len(norm(a)) >= 30:
            answer[answer_key(a)].append(i)
    near = minhash_clusters(rows)
    cluster_of = lambda groups: {i: f'{key[:12]}' for key, g in groups.items() if len(g) > 1 for i in g}
    ex_c, an_c = cluster_of(exact), cluster_of(answer)
    near_c = {i: f'n{min(g)}' for i, g in near.items()}
    records, summary = [], defaultdict(Counter)
    for i, r in enumerate(rows):
        q, _, a = r['text'].partition('\n답변: ')
        spans = []
        for kind, rx in SPANS.items():
            scope = r['text'] if kind in ('official_name', 'petition_no', 'masked_contact') else q
            for m in rx.finditer(scope):
                spans.append([m.start(), m.end(), kind])
        personal = r['kind'] == 'qa_item' and bool(PERSONAL.search(q))
        non_answer = r['kind'] == 'qa_item' and bool(NON_ANSWER.search(a))
        rank = sorted(answer[answer_key(a)]).index(i) if i in an_c else 0
        hold = sorted({s[2] for s in spans if s[2] in ('named_company', 'masked_contact')} |
                      ({'personal_situation'} if personal else set()) |
                      ({'non_answer'} if non_answer else set()) |
                      ({f'same_answer_over_{SAME_ANSWER_KEEP}'} if rank >= SAME_ANSWER_KEEP else set()) |
                      ({'dup_exact_non_first'} if i in ex_c and exact_first(exact, rows, i) != i else set()))
        rec = {'id': r['id'], 'sha256': r['sha256'], 'version': r['_version'], 'part': r['_part'],
               'dataset': r['raw'].split('/')[3], 'institution': r['provenance']['institution'],
               'institution_type': r['provenance'].get('institution_type'), 'year': year_of(r),
               'faq_kind': r['meta'].get('faq_kind', 'document'), 'chars': len(r['text']),
               'dup_exact': ex_c.get(i), 'dup_answer': an_c.get(i), 'dup_near': near_c.get(i),
               'spans': spans, 'personal_situation': personal, 'training_hold': hold}
        records.append(rec)
        for key in ('dataset', 'institution', 'institution_type', 'year', 'faq_kind'):
            summary[key][rec[key]] += 1
        summary['spans_by_dataset'].update(f"{rec['dataset']}:{kind}" for kind in {s[2] for s in spans})
        summary['holds_by_dataset'].update(f"{rec['dataset']}:{h}" for h in hold)
        summary['span_kinds'].update({s[2] for s in spans})
        summary['training_hold'].update(hold)
        summary['length'][('<200' if rec['chars'] < 200 else '<1000' if rec['chars'] < 1000 else '<3000' if rec['chars'] < 3000 else '>=3000')] += 1
    sizes = lambda c: Counter(len(g) for g in c.values() if len(g) > 1)
    out.mkdir(parents=True)
    body = ''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in records).encode()
    (out / 'annotations.jsonl').write_bytes(body)
    result = {'code_sha256': CODE_SHA256, 'versions': args.versions, 'rows': len(records),
              'annotations_sha256': hashlib.sha256(body).hexdigest(),
              'clusters': {'exact': {'rows': len(ex_c), 'size_hist': dict(sizes(exact))},
                           'same_answer': {'rows': len(an_c), 'size_hist': dict(sizes(answer))},
                           'near_0.8': {'rows': len(near_c), 'clusters': len(set(near_c.values()))}},
              'rows_with_hold': sum(1 for x in records if x['training_hold']),
              **{k: dict(v) for k, v in summary.items()}}
    (out / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(json.dumps(result, ensure_ascii=False, indent=1))


def answer_key(answer):
    return hashlib.sha256(norm(answer).encode()).hexdigest()


def exact_first(exact, rows, i):
    key = hashlib.sha256(norm(rows[i]['text']).encode()).hexdigest()
    return min(exact[key])


if __name__ == '__main__':
    main()
