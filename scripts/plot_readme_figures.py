"""README figures from stored reports; never re-runs a model.

  python scripts/plot_readme_figures.py            # writes docs/figures/{benchmark,training-data}.svg
  rsvg-convert -z 2 docs/figures/benchmark.svg -o docs/figures/benchmark.png   (same for training-data)

benchmark: strict product scoring (value + source + `값 [Dn]` format). trid-91M RAG310 is the best generative SFT
run with evidence-constrained decoding; trid-91M probe is the extractive reader's `exact`. Open-weight models are
reference baselines only (not trid) and use the `evidence_constrained` condition of scripts/evaluate_external_chat.py.
training-data: pretraining runs' own metrics and the mix-v3 token composition measurement.
"""
import json
import re
import math
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'docs/figures'
FONT = "Nanum Pen Script, NanumPen, sans-serif"
INK, MUTED, GRID = '#111827', '#6b7280', '#e5e7eb'

MODELS = (  # label, colour, RAG310 report, probe report; first entry is trid and is highlighted
    ('trid-91M (자체 학습)', '#2563eb', 'open-weight-baseline/rag310-trid-public-dialogue-gpu-05-mix04-s3000-evidence.json',
     'reader-factorial-01-two-complete.json'),
    ('Kanana-1.5-2.1B + LoRA', '#4b5563', 'open-weight-lora/rag310-kanana-lora.json', 'open-weight-lora/probe-kanana-lora.json'),
    ('Qwen3-1.7B + LoRA', '#9ca3af', 'open-weight-lora/rag310-qwen3-lora.json', 'open-weight-lora/probe-qwen3-lora.json'),
    ('Kanana-1.5-2.1B', '#64748b', 'open-weight-lora/rag310-kanana-base-constrained.json',
     'open-weight-lora/probe-kanana-base-constrained.json'),
    ('Qwen3-1.7B', '#a16207', 'open-weight-lora/rag310-qwen3-base-constrained.json',
     'open-weight-lora/probe-qwen3-base-constrained.json'),
)
READER = 'reader-factorial-01-cpt-v1-v2-v4'


def load(name):
    return json.loads((ROOT / 'reports' / name).read_text())


def split_scores(report):
    """(all, answerable, unanswerable) as (correct, total)."""
    if 'conditions' in report:
        s = report['conditions']['evidence_constrained']['summary']
        key = 'strict_product'
    else:
        s, key = report['summary'], 'strict'
    a, b = s['all'], s['answerable']
    return (a[key], a['total']), (b[key], b['total']), (a[key] - b[key], a['total'] - b['total'])


def probe_score(report):
    if 'conditions' in report:
        s = report['conditions']['evidence_constrained']['summary']['all']
        return s['strict_product'], s['total']
    row = next(r for r in report['results'] if r['name'] == READER)
    return row['probe_correct'], row['probe_total']


def text(x, y, s, size=13, anchor='start', weight='normal', fill=INK):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size + 3}" text-anchor="{anchor}" font-weight="{weight}" '
            f'fill="{fill}">{escape(s)}</text>')


def svg(width, height, body):
    """Match the architecture plates: pen lettering, double outlines and pale hatching."""
    patterns = {}

    def sketch(match):
        attrs = dict(re.findall(r'([\w-]+)="([^"]*)"', match.group(0)))
        x, y, w, h = (float(attrs[k]) for k in ('x', 'y', 'width', 'height'))
        color = attrs['fill']
        key = patterns.setdefault(color, f'hatch{len(patterns)}')
        # Exact rectangle retains the data extent; only the pen outline is irregular.
        return (
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{color}" fill-opacity=".06"/>'
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="url(#{key})"/>'
            f'<path d="M{x},{y+h} L{x+.4},{y} L{x+w},{y+.3} L{x+w-.3},{y+h} Z" '
            f'fill="none" stroke="{color}" stroke-width="1.5"/>'
            f'<path d="M{x+.8},{y+h-.5} L{x-.3},{y+.7} L{x+w-.4},{y-.4} L{x+w+.5},{y+h-.7}" '
            f'fill="none" stroke="{color}" stroke-width=".7" opacity=".65"/>'
        )

    content = re.sub(r'<rect\b[^>]+/>', sketch, '\n'.join(body))
    defs = ''.join(
        f'<pattern id="{key}" width="7" height="7" patternUnits="userSpaceOnUse">'
        f'<path d="M-2,7L7,-2M0,9L9,0" stroke="{color}" opacity=".25" stroke-width=".7"/>'
        '</pattern>' for color, key in patterns.items()
    )
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            f'font-family="{FONT}"><rect width="{width}" height="{height}" fill="#ffffff"/>'
            f'<defs>{defs}</defs>\n{content}\n</svg>\n')


def benchmark():
    panels = [('RAG310 전체', []), ('RAG310 답 있음', []), ('RAG310 답 없음·충돌', []), ('기관 문서 probe', [])]
    for _, _, rag, probe in MODELS:
        for i, score in enumerate(split_scores(load(rag)) + (probe_score(load(probe)),)):
            panels[i][1].append(score)
    width, height, top, plot_h = 1240, 560, 120, 300
    left, panel_w, gap = 60, 270, 20
    bar_w, bar_gap = 34, 12
    body = [text(40, 48, 'trid — 근거를 얼마나 정확히 읽는가', 24),
            text(40, 76, '기록된 기준선 비교 · 정답률(%) · greedy / 근거 제한 · 공개 모델은 trid 가중치에 포함되지 않음', 14, fill=MUTED)]
    base = top + plot_h
    for p, (title, scores) in enumerate(panels):
        x0 = left + p * (panel_w + gap)
        for tick in range(0, 101, 25):
            y = base - plot_h * tick / 100
            body.append(f'<path d="M{x0},{y:.1f}h{panel_w}" stroke="{GRID}" stroke-width="1"/>')
            if p == 0:
                body.append(text(x0 - 8, y + 4, str(tick), 12, 'end', fill=MUTED))
        span = len(scores) * bar_w + (len(scores) - 1) * bar_gap
        bx = x0 + (panel_w - span) / 2
        for (label, colour, *_), (correct, total) in zip(MODELS, scores):
            pct = 100 * correct / total
            h = plot_h * pct / 100
            body.append(f'<rect x="{bx:.1f}" y="{base - h:.1f}" width="{bar_w}" height="{max(h, 1):.1f}" rx="3" fill="{colour}"/>')
            body.append(text(bx + bar_w / 2, base - h - 7, f'{pct:.1f}', 12, 'middle', 'bold' if colour == MODELS[0][1] else 'normal'))
            bx += bar_w + bar_gap
        n = scores[0][1]
        body.append(text(x0 + panel_w / 2, base + 30, title, 15, 'middle', 'bold'))
        subtitle = f'{n}문항' if p < 3 else '40문항 · trid는 별도 추출 헤드'
        body.append(text(x0 + panel_w / 2, base + 50, subtitle, 12, 'middle', fill=MUTED))
    lx, ly = left, base + 92
    for label, colour, *_ in MODELS:
        body.append(f'<rect x="{lx}" y="{ly - 12}" width="14" height="14" rx="3" fill="{colour}"/>')
        body.append(text(lx + 20, ly, label, 12))
        lx += 225
    body.append(text(40, height - 18, 'RAG310: gpu-05-mix04 생성 모델 / 기관 probe: cpt-v1·head-v2·data-v4 추출 모델(exact). '
                     '서로 다른 체크포인트이며, probe는 반복 사용한 진단 자료입니다.', 12, fill=MUTED))
    return svg(width, height, body)


def training_data():
    base = json.loads((ROOT / 'runs/trid-korean-medium-base-v1/metrics.json').read_text())[-1]
    cpt_metrics = json.loads((ROOT / 'runs/trid-korean-medium-cpt-gpu-complete/metrics.json').read_text())
    cpt0, cpt = cpt_metrics[0], cpt_metrics[-1]
    comp = json.loads((ROOT / 'data/mix-v3-dry/measurement.json').read_text())['composition_tokens']
    parts = (('법령·행정규칙·자치법규', '#2563eb', comp['law-ko-source']),
             ('정책브리핑 보도자료', '#f59e0b', comp['korea-kr-briefing-source']),
             ('한국어 위키백과', '#10b981', comp['wiki-ko-full-source']))
    totals = [sum(c.values()) for _, _, c in parts]
    whole = sum(totals)
    width, height = 1240, 360
    body = [text(40, 48, 'trid-91M 사전학습 데이터', 24, weight='bold'),
            text(40, 76, '출처·라이선스를 추적할 수 있는 공개 한국어 자료만 사용 · 외부 사전학습 가중치 없음', 14, fill=MUTED)]
    rows = (('1단계  위키 사전학습', f'{base["tokens_seen"] / 1e6:,.0f}M 토큰 · {base["step"]:,} step · val loss {base["valid_loss"]:.3f}',
             ((parts[2][1], 1.0),)),
            ('2단계  혼합 이어 학습', f'{cpt["tokens_seen"] / 1e6:,.0f}M 토큰 · {cpt["step"]:,} step · 혼합 val loss '
                                 f'{cpt0["valid_loss"]:.3f} → {cpt["valid_loss"]:.3f}',
             tuple((colour, total / whole) for (_, colour, _), total in zip(parts, totals))))
    y, bar_x, bar_w = 120, 440, 760
    for title, detail, segments in rows:
        body.append(text(40, y + 16, title, 16, weight='bold'))
        body.append(text(40, y + 38, detail, 11, fill=MUTED))
        x = bar_x
        for colour, share in segments:
            w = bar_w * share
            body.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="34" fill="{colour}"/>')
            body.append(text(x + w / 2, y + 22, f'{100 * share:.1f}%', 14, 'middle', 'normal', INK))
            x += w
        y += 80
    lx = bar_x
    for label, colour, _ in parts:
        body.append(f'<rect x="{lx}" y="{y - 12}" width="14" height="14" rx="3" fill="{colour}"/>')
        body.append(text(lx + 20, y, label, 13))
        lx += 60 + 13 * len(label)
    body.append(text(40, height - 30, f'혼합 막대: mix-v3 준비 자료 {whole / 1e6:,.1f}M 토큰의 구성. '
                     '실제 CPT와 다른 토크나이저로 측정한 비율이며 학습 노출 비율이 아닙니다.', 12, fill=MUTED))
    return svg(width, height, body)


def summary(path):
    report = load(path)
    if 'conditions' in report:
        return report['conditions']['evidence_constrained']['summary'], 'strict_product'
    return report['summary'], 'strict'


def task_radar():
    axes = [('rag_value', '값 찾기'), ('rag_name', '이름·기관'),
            ('rag_noise', '방해 근거'), ('rag_revision', '개정 구분'),
            ('rag_table', '표 읽기'), ('rag_combine', '두 근거 결합'),
            ('rag_conflict', '근거 충돌'), ('rag_klue', 'KLUE 독해')]
    cx, cy, radius = 490, 310, 185
    body = [text(35, 40, '과제별 독해 프로필 — RAG310 안에서 무엇이 어려운가', 25),
            text(35, 70, '각 축은 해당 과제의 엄격 정답률(%) · 같은 근거 제한 조건 · 면적은 종합 점수가 아님', 15, fill=MUTED)]

    def point(i, r):
        angle = -math.pi / 2 + i * math.tau / len(axes)
        return cx + r * math.cos(angle), cy + r * math.sin(angle)

    def points(values):
        return ' '.join(f'{x:.2f},{y:.2f}' for x, y in values)

    for level in (25, 50, 75, 100):
        body.append(f'<polygon points="{points(point(i, radius * level / 100) for i in range(8))}" '
                    f'fill="none" stroke="{GRID}" stroke-width="1"/>')
        body.append(text(cx + 6, cy - radius * level / 100 + 5, str(level), 11, fill=MUTED))
    reference, _ = summary(MODELS[0][2])
    for i, (key, label) in enumerate(axes):
        x, y = point(i, radius)
        body.append(f'<path d="M{cx},{cy}L{x},{y}" stroke="{GRID}"/>')
        x, y = point(i, radius + 37)
        body.append(text(x, y, f'{label} (n={reference[key]["total"]})', 16, 'middle'))
    for index in (1, 2, 0):
        label, color, report, _ = MODELS[index]
        s, metric = summary(report)
        vertices = [point(i, radius * s[key][metric] / s[key]['total']) for i, (key, _) in enumerate(axes)]
        body.append(f'<polygon points="{points(vertices)}" fill="{color}" fill-opacity=".04" stroke="{color}" stroke-width="2"/>')
        for x, y in vertices:
            body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"/>')
    for i in range(3):
        label, color, *_ = MODELS[i]
        x = 110 + 300 * i
        body.append(f'<path d="M{x},575h25" stroke="{color}" stroke-width="2"/>')
        body.append(text(x + 33, 580, label, 15))
    body.append(text(35, 625, 'trid: gpu-05-mix04-s3000 · 공개 LoRA 두 모델은 비교 기준선. 작은 과제 표본과 평가 데이터 노출 차이를 고려해야 합니다.', 13, fill=MUTED))
    return svg(1000, 650, body)


def answer_errors():
    body = [text(35, 40, '답이 있는데 보류하는가 — 답 있음 228문항의 출력 분해', 25),
            text(35, 70, '엄격 정답 / 잘못된 보류 / 그 밖의 비정답 · 서로 겹치지 않는 세 범주', 15, fill=MUTED)]
    colors = ('#2563eb', '#b45309', '#64748b')
    for i, (label, _, report, _) in enumerate(MODELS[:3]):
        s, metric = summary(report)
        a = s['answerable']
        counts = (a[metric], a['abstained'], a['total'] - a[metric] - a['abstained'])
        assert min(counts) >= 0 and sum(counts) == 228
        y = 115 + i * 95
        body.append(text(35, y + 25, label, 17))
        x = 320
        for count, color in zip(counts, colors):
            width = 630 * count / 228
            body.append(f'<rect x="{x}" y="{y}" width="{width}" height="45" fill="{color}"/>')
            body.append(text(x + width / 2, y + 29, f'{count} ({count / 228:.1%})', 14, 'middle'))
            x += width
    for i, (label, color) in enumerate(zip(('엄격 정답', '답 있음에도 보류', '나머지 비정답'), colors)):
        x = 220 + i * 230
        body.append(f'<rect x="{x}" y="415" width="18" height="18" fill="{color}"/>')
        body.append(text(x + 27, 430, label, 16))
    body.append(text(35, 480, '비정답에는 값·출처·형식 오류 등이 포함됩니다. 보류 자체가 오류는 아니며, 여기서는 답이 있는 문항만 집계합니다.', 13, fill=MUTED))
    return svg(1000, 505, body)


def experiment_comparison():
    runs = [('gpu-05-mix04', '실제 QA 확장 / 기준선'),
            ('gpu-06-mix05', '문서별 teacher QA'),
            ('gpu-07-shared-ctx-01', '공유 문맥 QA'),
            ('gpu-07-mix06', '공유 문맥 + 기관 QA')]
    body = [text(35, 40, '데이터를 바꿨을 때 — 자체 91M의 3,000단계 실험', 25),
            text(35, 70, 'RAG310 답 있음 228문항 · 엄격 정답률(%) · 같은 단계 수가 모든 학습 조건의 동일성을 뜻하지는 않음', 14, fill=MUTED)]
    for i, (run, label) in enumerate(runs):
        s, metric = summary(f'open-weight-baseline/rag310-trid-public-dialogue-{run}-s3000-evidence.json')
        a = s['answerable']; rate = a[metric] / a['total']
        y = 115 + i * 65
        body.append(text(35, y + 22, label, 17))
        body.append(f'<rect x="340" y="{y}" width="{rate * 570 / .25}" height="30" fill="#2563eb"/>')
        body.append(text(350 + rate * 570 / .25, y + 22, f'{rate:.1%} · {a[metric]}/{a["total"]}', 16))
    for tick in (0, 5, 10, 15, 20, 25):
        x = 340 + 570 * tick / 25
        body.append(text(x, 400, f'{tick}%', 12, 'middle', fill=MUTED))
    body.append(text(35, 450, '이 그림은 엄격 정답입니다. 실험 판정 보고서의 “출처 일치 + 정답 언급” 점수·유의확률과 혼용하지 않습니다.', 13, fill=MUTED))
    return svg(1000, 475, body)


def main():
    (FIG / 'benchmark.svg').write_text(benchmark())
    (FIG / 'training-data.svg').write_text(training_data())
    for name, draw in [('task-radar', task_radar), ('answer-errors', answer_errors),
                       ('experiment-comparison', experiment_comparison)]:
        (FIG / f'{name}.svg').write_text(draw())


if __name__ == '__main__':
    main()
