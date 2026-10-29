"""README figures from stored reports; never re-runs a model.

  python scripts/plot_readme_figures.py            # writes docs/figures/{benchmark,training-data}.svg
  rsvg-convert -z 2 docs/figures/benchmark.svg -o docs/figures/benchmark.png   (same for training-data)

benchmark: strict product scoring (value + source + `값 [Dn]` format). trid-91M RAG310 is the best generative SFT
run with evidence-constrained decoding; trid-91M probe is the extractive reader's `exact`. Open-weight models are
reference baselines only (not trid) and use the `evidence_constrained` condition of scripts/evaluate_external_chat.py.
training-data: pretraining runs' own metrics and the mix-v3 token composition measurement.
"""
import json
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'docs/figures'
FONT = "Apple SD Gothic Neo, Noto Sans KR, Malgun Gothic, sans-serif"
INK, MUTED, GRID = '#111827', '#6b7280', '#e5e7eb'

MODELS = (  # label, colour, RAG310 report, probe report; first entry is trid and is highlighted
    ('trid-91M (자체 학습)', '#2563eb', 'open-weight-baseline/rag310-trid-public-dialogue-gpu-05-mix04-s3000-evidence.json',
     'reader-factorial-01-two-complete.json'),
    ('Kanana-1.5-2.1B + LoRA', '#4b5563', 'open-weight-lora/rag310-kanana-lora.json', 'open-weight-lora/probe-kanana-lora.json'),
    ('Qwen3-1.7B + LoRA', '#9ca3af', 'open-weight-lora/rag310-qwen3-lora.json', 'open-weight-lora/probe-qwen3-lora.json'),
    ('Kanana-1.5-2.1B', '#cbd5e1', 'open-weight-lora/rag310-kanana-base-constrained.json',
     'open-weight-lora/probe-kanana-base-constrained.json'),
    ('Qwen3-1.7B', '#e2e8f0', 'open-weight-lora/rag310-qwen3-base-constrained.json',
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
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" '
            f'fill="{fill}">{escape(s)}</text>')


def svg(width, height, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            f'font-family="{FONT}"><rect width="{width}" height="{height}" fill="#ffffff"/>\n' + '\n'.join(body) + '\n</svg>\n')


def benchmark():
    panels = [('RAG310 전체', []), ('RAG310 답 있음', []), ('RAG310 답 없음·충돌', []), ('기관 문서 probe', [])]
    for _, _, rag, probe in MODELS:
        for i, score in enumerate(split_scores(load(rag)) + (probe_score(load(probe)),)):
            panels[i][1].append(score)
    width, height, top, plot_h = 1240, 560, 120, 300
    left, panel_w, gap = 60, 270, 20
    bar_w, bar_gap = 34, 12
    body = [text(40, 48, 'trid 근거 기반 질의응답 성능', 24, weight='bold'),
            text(40, 76, '정답률(%) · 값과 출처, 답변 형식까지 모두 맞아야 정답 · greedy · 근거 복사 제한 디코딩 · '
                         '회색은 공개 모델 비교 기준선(trid 아님)', 14, fill=MUTED)]
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
        body.append(text(x0 + panel_w / 2, base + 50, f'{n}문항', 12, 'middle', fill=MUTED))
    lx, ly = left, base + 92
    for label, colour, *_ in MODELS:
        body.append(f'<rect x="{lx}" y="{ly - 12}" width="14" height="14" rx="3" fill="{colour}"/>')
        body.append(text(lx + 20, ly, label, 13, weight='bold' if colour == MODELS[0][1] else 'normal'))
        lx += 20 + sum(13 if ord(c) > 0x3000 else 7.5 for c in label) + 36
    body.append(text(40, height - 18, '기관 문서 probe의 trid-91M은 추출 독해기(exact) 결과입니다. 공개 모델에도 같은 근거 제한 문법을 적용했습니다. '
                     'probe는 반복 사용한 진단 자료입니다.', 12, fill=MUTED))
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
        body.append(text(40, y + 38, detail, 13, fill=MUTED))
        x = bar_x
        for colour, share in segments:
            w = bar_w * share
            body.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="34" fill="{colour}"/>')
            body.append(text(x + w / 2, y + 22, f'{100 * share:.1f}%', 14, 'middle', 'bold', '#ffffff'))
            x += w
        y += 80
    lx = bar_x
    for label, colour, _ in parts:
        body.append(f'<rect x="{lx}" y="{y - 12}" width="14" height="14" rx="3" fill="{colour}"/>')
        body.append(text(lx + 20, y, label, 13))
        lx += 60 + 13 * len(label)
    body.append(text(40, height - 30, f'2단계 혼합 자료: 정제·개인정보 치환·MinHash 중복 제거 후 {whole / 1e6:,.1f}M 토큰(비율은 측정 토크나이저 기준). '
                     '10억 토큰은 이 자료를 약 2회 반복 학습한 양입니다.', 12, fill=MUTED))
    return svg(width, height, body)


def main():
    (FIG / 'benchmark.svg').write_text(benchmark())
    (FIG / 'training-data.svg').write_text(training_data())


if __name__ == '__main__':
    main()
