"""Throwaway B-011: for every press item in the collector cache, was the body taken from the press-release file or from
another attachment (별첨, 보고서, 통계 ...)? Uses only stored raw (data/korea-kr-briefing-source/raw/items)."""
import json
import os
import re
from collections import Counter
from multiprocessing import Pool

DIR = '/Users/eomgyuhyeon/사업/trid/data/korea-kr-briefing-source/raw/items'
PRESS = re.compile(r'보도자료|보도 자료|조간|석간|즉시\s*보도|배포|\[\d{1,2}\.\d{1,2}|^\(?\d{2}[.-]\d{2}|보도참고')
ATTACH = re.compile(r'별첨|붙임|첨부|참고자료|부록|백서|보고서|통계|연보|Award|판정|가이드|매뉴얼|안내서|자료집')


def one(name):
    with open(os.path.join(DIR, name)) as f:
        d = json.load(f)
    if d.get('error'):
        return None
    chosen = (d.get('file') or {}).get('name') or ''
    names = [a['name'] for a in d.get('attachments') or []]
    press_like = [n for n in names if PRESS.search(n)]
    body = d.get('body') or ''
    if d.get('body_source') in (None, 'html', 'view', 'page'):
        kind = 'html_body'
    elif PRESS.search(chosen):
        kind = 'press_file'
    elif press_like:
        kind = 'other_attachment_while_press_file_exists'
    elif ATTACH.search(chosen):
        kind = 'attachment_like_name_no_press_file'
    else:
        kind = 'unclear_name'
    ctrl = sum(1 for c in body[:200000] if ord(c) < 32 and c not in '\n\t\r')
    return d['news_id'], kind, len(body), ctrl / max(1, min(len(body), 200000)), d.get('body_source')


if __name__ == '__main__':
    names = os.listdir(DIR)
    c, chars, garbled, src = Counter(), Counter(), Counter(), Counter()
    flagged = []
    with Pool(8) as pool:
        for r in pool.imap_unordered(one, names, chunksize=200):
            if not r:
                c['error_item'] += 1
                continue
            nid, kind, n, ctrl, bs = r
            c[kind] += 1
            chars[kind] += n
            src[bs] += 1
            if ctrl > 0.002:
                garbled[kind] += 1
            if kind == 'other_attachment_while_press_file_exists' or ctrl > 0.002:
                flagged.append({'news_id': nid, 'kind': kind, 'chars': n, 'control_char_ratio': round(ctrl, 4)})
    out = {'items': len(names), 'by_kind': dict(c), 'chars_by_kind': dict(chars), 'control_char_ratio_over_0.2pct': dict(garbled),
           'body_source': dict(src)}
    json.dump({**out, 'flagged': sorted(flagged, key=lambda x: -x['chars'])}, open('/tmp/trid-corpus/qa/press-attachment-scan.json', 'w'),
              ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))
