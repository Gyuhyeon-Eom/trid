"""Internal-work training-case candidates for the specialised small model (TRID-BASE-CUTOVER-001), CPU only.

Two sources, both from material B verified (B-012/B-013); no teacher-model generation, no institution data:

  official   법령해석례 rows of data/internal-rules/v3/docs.jsonl. Question = the official question in the
             interpretation title ("…하는지"), conclusion = 【회답】, ground = the 【이유】 paragraph right before the
             "따라서" paragraph. All three are verbatim official text; the only added text is a fixed line stating
             the reply date and that the interpretation reflects the law at that time.
  pack       behaviour cases written by C (AI, not human-reviewed) on the B-013 evidence-pack articles, in
             reports/coordination/C-011/pack-cases.json: answer with condition / exception / effective
             date / source, ask back when facts are missing, say a live value must be looked up, hold when the
             basis is not secured. Every quoted span is checked to be verbatim in the pack article.

Exclusions: rows whose document is in a B-013 reserve-overlap list (full match or partial review) are written
with training_hold; C-010 development documents are excluded so earlier development measurements stay separate.
Nothing here is an evaluation item.

usage: build_internal_cases.py --out DIR
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / 'data/internal-rules/v3/docs.jsonl'
PACKS = ROOT / 'reports/coordination/B-013/evidence-packs.json'
PACK_CASES = ROOT / 'reports/coordination/C-011/pack-cases.json'
OVERLAPS = [ROOT / 'reports/coordination/B-013/overlap-law-admrul-expc-all-148.json',
            ROOT / 'reports/coordination/B-013/overlap-pack-documents-148.json']
C010_DEV = ROOT / 'reports/coordination/C-010/dev60-source-review.jsonl'
REVIEW_HOLD = ROOT / 'reports/coordination/C-011/review-hold.json'   # id -> reason, from AI review; held until a person checks
HOLD_REASON = 'reserve-overlap (B-013, unresolved): training input on hold; never an evaluation item'


def overlap_ids():
    ids = set()
    for p in OVERLAPS:
        r = json.loads(p.read_text())
        ids |= {m['corpus_id'] for m in r.get('matches', [])} | {m['corpus_id'] for m in r.get('partial_review', [])}
    return ids


def official_question(row):
    """The official question: the title when it is phrased as a question ('…하는지(… 관련)'), otherwise the last
    question of 【질의요지】 (before '< 질의 배경 >'), cut after its last premise ('…있는바,' or a sentence end '다. ';
    dates such as '2014. 1. 1.' are not boundaries). A summary question that names no act gets the act from the
    title as a 'law 관련: question' prefix (template join). None when no clean question is found."""
    title = row['title']
    q = re.sub(r'^[^「」\-]{1,40}?\s*-\s*', '', title)          # '감사원 - …', '경찰청- …'
    q = re.sub(r'\s*\([^()]*관련\)\s*$', '', q).strip()
    if re.search(r'(지|여부)$', q):
        return q
    summary = row['text'].split('【질의요지】', 1)[1].split('【회답】', 1)[0].split('< 질의 배경 >', 1)[0]
    summary = re.sub(r'\s+', ' ', summary).strip()
    m = re.search(r'(?:지\?|지 여부\??|여부\?)\s*$', summary)
    if not m:
        return None
    head = summary[:m.end()]
    cuts = [x.end() for x in re.finditer(r'(?:있는바|하는바|되어 있는바|규정하고 있는바|있으므로),\s*|[가-힣]다\.\s+', head)]
    q = head[max(cuts) if cuts else 0:].strip()
    if not q or not re.match(r'[가-힣「“"]', q):
        return None                                             # starts mid-phrase (e.g. ') 제7조를 …')
    if '「' not in q:
        law = re.search(r'「[^」]+」(?:\s*제\d+조(?:의\d+)?(?:제\d+항)?)?', title)
        if not law:
            return None
        q = f'{law.group(0)} 관련: {q}'
    return q if 15 <= len(q) <= 320 else None


def stated_ground(reason):
    """The paragraph right before the '따라서' conclusion of 【이유】, verbatim. Older replies mark paragraphs with ○,
    newer ones use line breaks; a paragraph broken by the extraction ('…있' / '인데, …') is joined back verbatim."""
    lines = [l for l in reason.split('\n') if l.strip()]
    k = next((i for i, l in enumerate(lines) if l.lstrip('○ ').startswith('따라서')), None)
    if not k:
        return None
    start = k - 1
    while start > 0 and (re.match(r'\s*(인데|로서|이고|으로서|,)', lines[start]) or not re.search(r'[다.]\s*$', lines[start - 1])):
        start -= 1
    return '\n'.join(lines[start:k]).strip()


def bigram_overlap(a, b):
    grams = lambda s: {s[i:i + 2] for i in range(len(s) - 1)}
    ga, gb = grams(re.sub(r'\s+', '', a)), grams(re.sub(r'\s+', '', b))
    return len(ga & gb) / max(1, min(len(ga), len(gb)))


def official_case(row):
    text = row['text']
    if not all(k in text for k in ('【질의요지】', '【회답】', '【이유】')):
        return None, 'no sections'
    question = official_question(row)
    if not question:
        return None, 'no official question sentence'
    answer = text.split('【회답】', 1)[1].split('【이유】', 1)[0].strip()
    if re.search(r'질의\s*[가나다1-9]', answer):
        return None, 'multi-part answer'
    ground = stated_ground(text.split('【이유】', 1)[1])
    if not answer or not ground or len(ground.strip('○ .')) < 40:
        return None, 'no stated ground'
    ground = ground.lstrip('○ ').strip()
    if bigram_overlap(answer, ground) > .6:
        return None, 'ground only restates the answer'
    if len(answer) > 600 or len(ground) > 700:
        return None, 'too long for the training budget'
    if re.search(r'\d\)\d\)', question + ground):
        return None, 'footnote text leaked into question or ground (extraction artifact)'
    if re.match(r'(여기에서|이 경우|위\s)', question):
        return None, 'question starts with a referent outside the question'
    if ground not in text:
        return None, 'ground not verbatim'
    date = row.get('date') or '미상'
    context = f"{row['title']}\n【회답】\n{answer}\n【이유】\n{ground}"
    target = (f"결론: {answer}\n근거: {ground}\n"
              f"시점: 이 해석은 {date} 회신(날짜 필드는 회답일로 보이나 미확인) 당시의 법령을 기준으로 한 것이므로, 지금 적용하려면 현행 조문을 확인해야 합니다.\n"
              f"출처: 법제처 법령해석례 {row['id']}")
    return dict(kind='official_interpretation', behavior='answer_with_basis', question=question, context=[context],
                target=target, evidence=dict(conclusion=answer, ground=ground), date=date, date_field='date (회답일 추정)',
                authored='question, conclusion and ground verbatim from the official interpretation; date/source lines are a fixed template'), None


def pack_cases():
    packs = {(a['id'], a['article']): a for arts in json.loads(PACKS.read_text()).values() for a in arts}
    rows = []
    for c in json.loads(PACK_CASES.read_text())['cases']:
        a = packs[(c['doc'], c['article'])]
        for span in c['evidence']:
            if span not in a['text']:
                raise ValueError(f"{c['id']}: evidence not verbatim in {c['article']}: {span[:40]!r}")
        rows.append(dict(kind='pack_behavior', behavior=c['behavior'], question=c['question'], context=[a['text']],
                         target=c['target'], evidence=dict(spans=c['evidence']), doc=c['doc'], article=c['article'],
                         effective=a['raw_dates'].get('시행일자'), date_fields=a['raw_dates'], source=a['source'],
                         text_sha256=a['text_sha256'], authored='question and explanation written by C (AI assistant); not human-reviewed',
                         id=c['id'], missing=c.get('missing', [])))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    held = overlap_ids()
    dev = {json.loads(l)['doc'] for l in C010_DEV.open()}
    out, skipped = [], Counter()
    for line in V3.open():
        row = json.loads(line)
        if row['kind'] != 'legal_interpretation':
            continue
        if row['id'] in dev:
            skipped['C-010 development document'] += 1
            continue
        case, why = official_case(row)
        if not case:
            skipped[why] += 1
            continue
        case.update(id=f"official:{row['id']}", doc=row['id'], source=row['source'], license=row['license'],
                    text_sha256=row['sha256'], split_hint=row.get('split'))
        out.append(case)
    out += pack_cases()
    for c in out:
        review_hold = json.loads(REVIEW_HOLD.read_text()) if REVIEW_HOLD.exists() else {}
        c['training_hold'] = HOLD_REASON if c['doc'] in held else (f"review hold: {review_hold[c['id']]}" if c['id'] in review_hold else None)
        if not c['training_hold'] and c.get('split_hint') in ('valid', 'test'):
            c['training_hold'] = f"B v3 split '{c['split_hint']}': kept out of training to keep evaluation-side rows separate"
        c['review'] = 'unreviewed'
        c['synthetic_parts'] = 'template lines' if c['kind'] == 'official_interpretation' else 'question and explanation (AI-written)'
    batch = [c for c in out if not c['training_hold']]
    excluded = [c for c in out if c['training_hold']]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'candidates.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in batch))
    (a.out / 'excluded.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in excluded))
    meta = dict(purpose='C-011 internal-work SFT candidates (question / context / target / evidence); not evaluation items; '
                        'AI review only, not human-verified; tokenizer/template left to A',
                candidates=dict(Counter(c['behavior'] for c in batch)), candidates_total=len(batch),
                excluded=dict(Counter(c['training_hold'][:40] for c in excluded)), skipped_before_case=dict(skipped),
                evidence_docs=sorted({c['doc'] for c in batch}),
                inputs={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (V3, PACKS, PACK_CASES, *OVERLAPS, C010_DEV)},
                reserve_note='148 reserve families checked by B (v3 quarantine) and B-013 full scan; the other 852 families are not checked')
    (a.out / 'manifest.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != 'evidence_docs'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
