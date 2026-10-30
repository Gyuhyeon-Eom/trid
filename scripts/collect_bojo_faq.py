"""국고보조금 FAQ (보조금통합포털 bojo.go.kr, e나라도움 자주묻는질문) via the list the FAQ page itself loads.

Source: data.go.kr dataset 15099563 (기획예산처_국고보조금 자주묻는질문 정보, 이용허락범위 제한 없음; 제공형태 '기관자체에서
다운로드') points to https://bojo.go.kr/hg/hg002/retrieveFaqList.do. Rights: the portal's 저작권 정책 (/ja/getJA011001View.do)
cites 저작권법 제24조의2 (free use of 기획예산처·행정안전부 works) and the public-data policy (commercial use included).
robots.txt: Allow: / (Disallow only /wisenut/). The page loads POST /ga/retrieveGA008100List.do with its own form
(10 per page); this repeats that request page by page through the shared HostLimiter (log logs/requests-bojo.go.kr.jsonl).
Pages are stored once under data/public-data/raw/15099563/pages/<n>.json.

  fetch    all pages (resumable)
  extract  one row per FAQ: title, answer as text (img alt text kept and flagged), source category path, dates
"""
import argparse
import hashlib
import html
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()  # as loaded
RAW = ROOT / 'data/public-data/raw/15099563/pages'
URL = 'https://bojo.go.kr/ga/retrieveGA008100List.do'
FORM = {'searchCtgry': 'SJ_CN', 'searchText': '', 'searchCode': '', 'searchSiteId': 'SITE_00036', 'perPage': '10',
        'cmmnDetailUserCode': '000', 'cmmnDetailPartCode': '000', 'cmmnDetailPartDtlCode': '000'}


def fetch(args):
    import collect_assembly_minutes as am
    client = am.Client(am.HostLimiter(args.interval, host='bojo.go.kr'), ROOT / 'logs/requests-bojo.go.kr.jsonl')
    metrics = am.Metrics()
    RAW.mkdir(parents=True, exist_ok=True)
    page, total = 1, None
    while total is None or (page - 1) * 10 < total:
        dest = RAW / f'{page:03d}.json'
        if dest.exists():
            body = dest.read_bytes()
        else:
            r = client.request('POST', URL, metrics, data={**FORM, 'curPage': str(page)})
            if r.status_code != 200:
                raise SystemExit(f'stopped: HTTP {r.status_code} on page {page}')
            body = r.content
            dest.write_bytes(body)
        items = json.loads(body)['ntbdList']
        if not items:
            break
        total = items[0]['totCnt']
        page += 1
    print(json.dumps({'pages': page - 1, 'total': total, 'metrics': metrics.snapshot()}, ensure_ascii=False))


def to_text(fragment):
    alts = re.findall(r'<img[^>]*\balt="([^"]*)"', fragment)
    text = re.sub(r'</t[dh]>', ' | ', fragment)  # keep cell boundaries; rows end at </tr>
    text = re.sub(r'<br\s*/?>|</p>|</li>|</tr>', '\n', text)
    text = re.sub(r'<img[^>]*\balt="([^"]*)"[^>]*>', r'\n\1\n', text)
    text = html.unescape(re.sub(r'<[^>]+>', '', text)).replace('\xa0', ' ')
    text = '\n'.join(re.sub(r'[ \t]+', ' ', x).strip() for x in text.split('\n'))
    return re.sub(r'\n{3,}', '\n\n', text).strip(), bool(''.join(alts).strip())


def extract(args):
    from trid.korean_quality import PUBLIC_DOCS, assess, redact
    from extract_public_guides import faq_signals
    out = ROOT / 'data/public-data' / args.version
    if (out / 'manifest.json').exists():
        raise SystemExit(f'{out} exists; versions are never overwritten')
    rows, seen = [], set()
    for path in sorted(RAW.glob('*.json')):
        raw_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        for it in json.loads(path.read_bytes())['ntbdList']:
            if it['nttId'] in seen:
                continue
            seen.add(it['nttId'])
            answer, from_alt = to_text(it.get('nttCn') or '')
            text, pii = redact(f"질문: {it['sjCn']}\n답변: {answer}")
            ok, why = assess(text, t=PUBLIC_DOCS)
            # text taken from <img alt> was written for screen readers; it is not checked against the image itself
            holds = ['answer_from_img_alt'] if from_alt else []
            if not answer.strip():
                holds.append('no_answer_content')
            rows.append({'id': f"bojo.go.kr:faq:{it['nttId']}", 'group': 'bojo.go.kr:faq', 'kind': 'qa_item', 'title': it['sjCn'],
                         'text': text, 'sha256': hashlib.sha256(text.encode()).hexdigest(),
                         'source': 'https://bojo.go.kr/hg/hg002/retrieveFaqList.do',
                         'license': '저작권법 제24조의2 + 공공데이터 이용정책 (bojo.go.kr 저작권 정책); data.go.kr 15099563 이용허락범위 제한 없음',
                         'raw': str(path.relative_to(ROOT)), 'raw_sha256': raw_sha,
                         'flags': (['pii_redacted'] if pii else []) + [f'hold:{h}' for h in holds],
                         'provenance': {'institution': '기획예산처·행정안전부 (운영: 한국재정정보원)', 'institution_type': '중앙행정기관',
                                        'area': '국고보조금 집행·정산', 'doc_type': 'FAQ',
                                        'applies_to': 'e나라도움 국고보조사업(중앙관서·지자체·민간보조사업자)',
                                        'date': it.get('updtDt') or it.get('registDdCn'), 'date_type': '게시물 수정일(없으면 등록일)'},
                         'meta': {'registered': it.get('registDdCn'), 'updated': it.get('updtDt'), 'category_path': it.get('codeNm'),
                                  'notice': it.get('noticeAt') == 'Y', 'answer_from_img_alt': from_alt,
                                  **faq_signals(answer, it.get('codeNm') or ''), 'quality': 'ok' if ok else why,
                                  'pii_redactions': dict(pii), 'table_check': 'unverified' if '<table' in (it.get('nttCn') or '') else 'no_tables',
                                  'clean_version': args.version, 'code_sha256': CODE_SHA256}})
    out.mkdir(parents=True)
    body = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode()
    (out / 'qa_items.jsonl').write_bytes(body)
    with open(out / 'overlap-input.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps({'id': r['id'], 'group': r['group'], 'source': r['source'], 'text': r['text']}, ensure_ascii=False) + '\n')
    man = {'version': args.version, 'code_sha256': CODE_SHA256, 'rows': len(rows), 'characters': sum(len(r['text']) for r in rows),
           'sha256': hashlib.sha256(body).hexdigest(), 'faq_kind': dict(Counter(r['meta']['faq_kind'] for r in rows)),
           'answer_from_img_alt': sum(r['meta']['answer_from_img_alt'] for r in rows),
           'table_check': dict(Counter(r['meta']['table_check'] for r in rows)),
           'pii_rows': sum(1 for r in rows if r['meta']['pii_redactions']), 'quality': dict(Counter(r['meta']['quality'] for r in rows))}
    (out / 'manifest.json').write_text(json.dumps(man, ensure_ascii=False, indent=1))
    print(json.dumps(man, ensure_ascii=False, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['fetch', 'extract'])
    p.add_argument('--interval', type=float, default=2.0)
    p.add_argument('--version', default='bojo-faq-v1')
    args = p.parse_args()
    {'fetch': fetch, 'extract': extract}[args.action](args)


if __name__ == '__main__':
    main()
