"""Collect open-weight baseline reports into one comparison table; never re-runs models.

RAG310: external reports are read as stored; stored trid answers are rescored with the same
product/content scorer so content_* columns are comparable. Institution probe: external models use
content_strict (value+source, order-free); trid's extractive reader uses its own `exact` metric, so the
two are listed side by side, not ranked as the same measurement. Missing reports are marked pending.

grounded_mention is a verbosity-tolerant reading check computed here from stored answers: every gold
value (or the abstention phrase) appears AND the set of cited [Dn] equals the gold source set
(no citation for abstentions). It does not penalize extra unsupported sentences, so it is an
upper-bound diagnostic, not a product metric.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_external_chat import SCORED, product_score, summarize, wilson

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'reports/open-weight-baseline'
MODELS = ('qwen3-1.7b-q8', 'kanana-1.5-2.1b-q8')
CONDITION = 'grammar_instructed'
TRID_RAG = {'trid-medium-sft-v1 (free)': 'reports/rag310-medium-sft-v1-free.json',
            'trid-medium-sft-v1 (evidence)': 'reports/rag310-medium-sft-v1-evidence.json',
            **{f'{run} ({decoding})': f'reports/open-weight-baseline/rag310-{run}-{decoding}.json'
               for run in ('trid-public-dialogue-01', 'trid-public-dialogue-gpu-01') for decoding in ('evidence', 'free')}}
TRID_PROBE = {'trid reader cpt-v1/v2/v4': 'reports/reader-factorial-01-cpt-v1-v2-v4-probe.json',
              'trid reader cpt-v1/v2/v2': 'reports/reader-factorial-01-cpt-v1-v2-v2-probe.json'}

CITATION = re.compile(r'\[(D[1-9])\]')
ABSTAIN = '확인 불가'


def rows_for(data):
    return [json.loads(line) for line in (ROOT / data / 'test.jsonl').read_text().splitlines()]


def grounded_rate(rows, results):
    """All / answerable / unanswerable: abstention-heavy models otherwise look like readers."""
    if [r['id'] for r in rows] != [r['id'] for r in results]:
        raise ValueError('Stored answers do not match the evaluation rows')
    hit = [bool(res.get('mentions_gold')) and set(CITATION.findall(res.get('generated') or '')) == set(row['sources'])
           for row, res in zip(rows, results, strict=True)]
    split = {}
    for name, keep in (('all', lambda row: True), ('answerable', lambda row: bool(row['gold'])),
                       ('unanswerable', lambda row: not row['gold'])):
        chosen = [h for row, h in zip(rows, hit) if keep(row)]
        split[name] = cell(sum(chosen), len(chosen))
    answerable = [res for row, res in zip(rows, results) if row['gold']]
    split['answerable_abstained'] = cell(sum(ABSTAIN in (res.get('generated') or '') for res in answerable), len(answerable))
    return split


def load(path):
    path = ROOT / path if not Path(path).is_absolute() else Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def cell(correct, total):
    low, high = wilson(correct, total)
    return dict(correct=correct, total=total, rate=round(correct / total, 4), wilson95=[round(low, 4), round(high, 4)])


def external(kind, model):
    report = load(BASE / f'{kind}-{model}.json')
    if report is None:
        return None
    s = report['conditions'][CONDITION]['summary']['all']
    # Gold value present but value+source check failed: separates format/length loss from reading errors.
    lost = [r for r in report['conditions'][CONDITION]['results'] if r['mentions_gold'] and not r['content_strict']]
    return dict(report=f'reports/open-weight-baseline/{kind}-{model}.json', model_sha256=report['model_sha256'],
                test_sha256=report['test_sha256'], strict_product=cell(s['strict_product'], s['total']),
                content_strict=cell(s['content_strict'], s['total']), mentions_gold=cell(s['mentions_gold'], s['total']),
                grounded_mention=grounded_rate(rows_for(report['data']), report['conditions'][CONDITION]['results']),
                truncated_output=s['truncated_output'], errors=s['errors'],
                gold_mentioned_but_not_scored=dict(total=len(lost), truncated=sum(bool(r.get('truncated_output')) for r in lost)))


def trid_rag(path, test_sha256):
    report = load(path)
    if report is None or report['test_sha256'] != test_sha256:
        return None
    rows = rows_for(report['data'])
    if [r['id'] for r in rows] != [r['id'] for r in report['results']]:
        raise ValueError(f'Stored answers do not match the evaluation rows: {path}')
    results = []
    for row, old in zip(rows, report['results'], strict=True):
        result = {**old, **product_score(row, old.get('generated') or '')}
        if old.get('skipped_context') or old.get('exception'):
            result.update({m: False for m in SCORED})
        results.append(result)
    s = summarize(rows, results)['all']
    return dict(report=path, model_sha256=report['model_sha256'], test_sha256=test_sha256,
                strict_product=cell(s['strict_product'], s['total']), content_strict=cell(s['content_strict'], s['total']),
                mentions_gold=cell(s['mentions_gold'], s['total']), grounded_mention=grounded_rate(rows, results))


def build(trid_rag_reports=()):
    rag = {m: external('rag310', m) for m in MODELS}
    known = {r['test_sha256'] for r in rag.values() if r}
    if len(known) > 1:
        raise ValueError('External RAG310 reports used different test files')
    if known:
        sha = known.pop()
        for name, path in {**TRID_RAG, **dict(trid_rag_reports)}.items():
            rag[name] = trid_rag(path, sha)
            if rag[name] is None and (ROOT / path).exists():
                raise ValueError(f'trid report used a different RAG310 test file: {path}')
    probe = {m: external('probe', m) for m in MODELS}
    for name, path in TRID_PROBE.items():
        r = load(path)
        s = r and r['summary'].get('all', r['summary'])
        probe[name] = r and dict(report=path, model_sha256=r['model_sha256'], metric='extractive exact',
                                 exact=cell(s['exact'], s['total']))
    public = load(BASE / 'public-dialogue-assessment.json')
    return dict(protocol=__doc__, condition=CONDITION, rag310=rag, institution_probe=probe,
                public_dialogue=public and {m: dict(heldout=f"{a['heldout_public_qa']['correct']}/{a['heldout_public_qa']['total']}",
                                                    regressions=sum(v['pass'] for v in a['user_regressions'].values()),
                                                    termination=a['termination']['pass'], counterfactual=a['counterfactual']['pass'])
                                            for m, a in public['models'].items()} | {'trid reference': public['trid_reference']},
                pending=[f'{k}:{m}' for k, table in (('rag310', rag), ('probe', probe)) for m, v in table.items() if v is None])


def pct(c):
    return f"{c['correct']}/{c['total']} ({c['rate']:.1%}, 95% {c['wilson95'][0]:.1%}-{c['wilson95'][1]:.1%})"


def short(c):
    return f"{c['correct']}/{c['total']} ({c['rate']:.1%})"


def markdown(summary):
    lines = ['# 공개 가중치 기준선 비교', '', '조건: `grammar_instructed`, greedy, 사고 비활성화. 자동 생성 문서이며 수치는 각 보고서에서 읽는다.', '',
             '## RAG310 (근거 제공 질의응답)', '', '| 모델 | 형식까지 엄격 | 값·출처 (순서 무관) | 출처 일치 + 정답 언급 | └ 답 있음 | └ 답 없음(보류·충돌) | 답 있는데 보류 |', '|---|---|---|---|---|---|---|']
    for name, r in summary['rag310'].items():
        g = r and r['grounded_mention']
        lines.append(f'| {name} | 진행 중 | | | | | |' if r is None else
                     f"| {name} | {pct(r['strict_product'])} | {pct(r['content_strict'])} | {pct(g['all'])} | {short(g['answerable'])} | {short(g['unanswerable'])} | {short(g['answerable_abstained'])} |")
    lines += ['', '## 기관 문서 probe 40', '', '| 모델 | 지표 | 결과 |', '|---|---|---|']
    for name, r in summary['institution_probe'].items():
        if r is None:
            lines.append(f'| {name} | | 진행 중 |')
        elif 'exact' in r:
            lines.append(f"| {name} | 추출 exact | {pct(r['exact'])} |")
        else:
            lines.append(f"| {name} | 값·출처 (순서 무관) | {pct(r['content_strict'])} · 출처 일치 + 정답 언급 {pct(r['grounded_mention']['all'])} · 64토큰 잘림 {r['truncated_output']} |")
    lines += ['', '추출 reader의 exact와 생성 모델의 값·출처 채점은 같은 측정이 아니다.',
              '"출처 일치 + 정답 언급"은 장문을 허용하는 독해 상한 진단치다: 정답 값이 들어 있고 인용한 [Dn] 집합이 정답 출처와 같으면 맞음. 덧붙인 근거 없는 문장은 감점하지 않는다.', '',
              '## 공개 문서 8문항 · 회귀 3 · 반사실', '', '| 모델 | 분리 문서 8 | 회귀 통과 | 128토큰 종료 | 반사실 |', '|---|---|---|---|---|']
    for name, r in (summary['public_dialogue'] or {}).items():
        if name == 'trid reference':
            continue
        lines.append(f"| {name} | {r['heldout']} | {r['regressions']}/3 | {'통과' if r['termination'] else '실패'} | {'통과' if r['counterfactual'] else '실패'} |")
    for name, r in ((summary['public_dialogue'] or {}).get('trid reference') or {}).items():
        lines.append(f"| {name} | {r['heldout_public_qa']} | | | {r.get('counterfactual', '')} |")
    lines += ['', '판정 근거·경계 사례: `reports/open-weight-baseline/public-dialogue-assessment.json` (AI 검토, 독립 사람 평가 아님).']
    if summary['pending']:
        lines += ['', '미완료: ' + ', '.join(summary['pending'])]
    return '\n'.join(lines) + '\n'


def candidate(value):
    name, sep, path = value.partition('=')
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError('expected NAME=REPORT_PATH')
    return name, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=BASE / 'summary.json')
    parser.add_argument('--trid-rag', type=candidate, action='append', default=[],
                        help='Extra stored trid RAG310 report (scripts/evaluate_rag_eval.py output), NAME=PATH')
    args = parser.parse_args()
    summary = build(args.trid_rag)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    args.output.with_suffix('.md').write_text(markdown(summary))
    print(markdown(summary))


if __name__ == '__main__':
    main()
