# trid

**맥북 한 대로 처음부터 만드는 한국어 언어모델.** 토크나이저, Transformer, 학습 루프, 평가셋까지 전부 이 저장소 안에 있고, 외부 사전학습 가중치는 쓰지 않는다.

하려는 건 하나다. 공공기관이나 기업의 폐쇄망 안에서, 규정·공고·보도자료 같은 문서를 근거로 받아 그 안의 값을 **지어내지 않고 그대로 찾아서** 답하는 작은 모델.

<p align="center"><img src="docs/figures/pipeline.png" width="1020" alt="trid pipeline"></p>

## 어떻게 답하나

질문이 들어오면 문서에서 근거 문단 몇 개를 찾아 번호를 붙여 모델에 준다. 모델은 세 가지 중 하나로 답한다.

```
8만 원 [D3]        값과 출처
확인 불가           근거에 없다
근거 충돌           근거끼리 값이 다르다
```

값 부분은 디코딩 단계에서 **인용한 근거 문단에 실제로 있는 글자열만** 생성하도록 묶어 둔다. 출처 번호도 프롬프트에 준 번호만 쓸 수 있다. 그래서 숫자나 이름을 꾸며내는 일은 구조적으로 없고, 틀린다면 엉뚱한 자리를 골랐거나 보류를 못 한 것이다. 어느 쪽인지 평가에서 따로 센다.

## 왜 처음부터 만드나

폐쇄망에 들어가는 모델은 가중치가 어디서 왔고 무슨 데이터로 학습했는지 끝까지 설명할 수 있어야 한다. 공개 가중치를 받아 이름만 바꾸면 그 설명이 안 된다. 그래서 라이선스가 분명한 한국어 자료만 모아 작은 모델을 직접 학습한다. 성능은 공개 대형 모델보다 낮다. 대신 모든 단계가 재현되고, 틀린 답의 원인을 데이터까지 거슬러 올라갈 수 있다.

## 모델

| | |
|---|---|
| 구조 | decoder-only Transformer. RoPE, SwiGLU, RMSNorm, 임베딩 공유. GQA·QK-norm 옵션 |
| 크기 | 91M (dim 768 · 12층 · 문맥 1,024). 이전 세대 17M |
| 토크나이저 | byte-level BPE 8,192 |
| 학습 | PyTorch MPS. bf16, `torch.compile`, gradient accumulation, WSD 스케줄 지원 |
| 추론 | KV cache, 근거 복사 제한 디코딩. CPU에서도 돈다 |

## 지금 어디까지 왔나

<p align="center"><img src="docs/figures/pretrain-loss.svg" width="760" alt="pretraining loss"></p>

91M 사전학습이 돌고 있다 (검증 손실 9.17 → 3.36, 32,750 step). 그 전 세대인 17M은 근거 읽기에 실패했다. 자체 평가셋 310문항에서 값과 출처를 모두 맞힌 게 0개, 제한 디코딩을 걸어도 3개였다. 질문이 뭐든 학습 데이터에 있던 값을 그대로 뱉었는데, 학습 데이터가 금액 10개·이름 8개로만 돼 있어 복사가 아니라 암기를 배운 탓이다. 데이터를 다시 만들었고(16개 영역, 넓은 범위의 무작위 값) 91M으로 비교하는 중이다.

검색 쪽은 괜찮다. 같은 평가셋에서 BM25에 형태소 분석을 붙이면 Recall@3 0.95, 외부 임베딩 모델(KURE-v1)과 차이가 3pp 안이다. 병목은 생성이다.

숫자와 실패 기록은 전부 [KOREAN_RAG.md](KOREAN_RAG.md)와 `reports/`에 있다. 실패한 시도도 지우지 않고 남겨 뒀다.

## 실행

```sh
uv sync --frozen --python 3.12
.venv/bin/python -m unittest discover -s tests        # 108 tests, CPU

# 평가 (근거 복사 제한 디코딩)
.venv/bin/python scripts/evaluate_rag_eval.py --run runs/<run> --decoding evidence --output reports/<name>.json
```

사전학습·SFT·데이터 수집 방법은 [PRETRAINING.md](PRETRAINING.md), [KOREAN_RAG.md](KOREAN_RAG.md)에 있다.

## 저장소

```
trid/        모델 · 학습 · 추론 · 제한 디코딩 · 품질 필터
scripts/     수집 · 정제 · 학습 파이프라인 · 평가
tests/       108 tests
reports/     실험 결과
docs/        그림, 이전 README
```

## 라이선스

코드는 Apache License 2.0. 학습 데이터는 위키백과(CC BY-SA), 법제처 법령·판례(저작권법 제7조), korea.kr 보도자료(공공누리 1유형), KLUE(CC BY-SA)만 쓴다. 학습된 가중치는 아직 배포하지 않는다.
