# trid

**대한민국 공공기관 폐쇄망 전용 한국어 언어모델.** 인터넷이 끊긴 기관 내부 서버에서, 규정·공고·보도자료 같은 문서를 근거로 받아 그 안의 값을 **지어내지 않고 그대로 찾아서** 답하는 작은 모델이다. 맥북 한 대로 처음부터 학습하고, 토크나이저·Transformer·학습 루프·평가셋까지 전부 이 저장소 안에 있다. 외부 사전학습 가중치, 외부 API, 외부 임베딩 모델은 제품에 들어가지 않는다.

폐쇄망이라는 조건이 설계를 정한다. 학습 데이터는 출처와 라이선스를 끝까지 설명할 수 있는 공공 자료만 쓰고, 추론은 CPU 한 대에서 300 MB 안에 돌며, 답은 근거 문단의 글자열만 복사하고 출처 번호를 남겨 감사 로그로 추적할 수 있어야 한다.

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
| 구조 | decoder-only Transformer. RoPE, SwiGLU, RMSNorm(pre-norm), 입·출력 임베딩 공유. GQA·QK-norm 옵션 |
| 크기 | 91,245,312 파라미터 (dim 768 · 12층 · 12 head · FFN 2,048 · 문맥 1,024). 이전 세대 17M |
| 파라미터 분포 | 임베딩 6.3M (7%) · attention 28.3M (31%) · FFN 56.6M (62%) |
| 토크나이저 | byte-level BPE 8,192. 한국어 1.6~1.8자/토큰 |
| 학습 | PyTorch MPS. bf16 autocast, `torch.compile`, gradient accumulation, WSD 스케줄, z-loss 옵션 |
| 추론 | KV cache. 근거 복사 제한 디코딩 또는 구간 단위 점수화 |

### 내부

<p align="center"><img src="docs/figures/architecture.png" width="920" alt="trid model architecture"></p>

- GQA(`kv_heads`)는 KV를 압축 저장하고 SDPA 직전에만 head 수로 펼친다. 12→4 KV head면 캐시가 72→24 MiB(FP32, 1,024 토큰).
- QK-norm은 RoPE 전에 head 차원 RMSNorm. 두 옵션 모두 기본은 꺼져 있고, 켜도 기존 체크포인트와 키·순서가 유지된다.
- 구조는 llama.cpp의 `llama` 아키텍처와 텐서 이름만 다르다. GGUF로 내보내 CPU에서 돌릴 수 있고, Q8_0에서 greedy 출력이 PyTorch와 동일함을 확인했다 (`reports/oss-model-size-and-inference.md`).

### 답을 만드는 두 가지 방식

둘 다 **근거에 없는 값은 나올 수 없다**는 보장은 같다. 차이는 모델이 어디서 결정하느냐다.

| | 근거 복사 제한 디코딩 (`trid/constrained.py`) | 구간 단위 점수화 (`trid/span_scoring.py`) |
|---|---|---|
| 방식 | 토큰마다 "지금까지 생성한 바이트열 + 이 토큰"이 인용 근거의 부분 문자열인지 검사. 아니면 다음 점수 토큰 | 근거의 단어 경계에서 시작하는 모든 구간(최대 8어절·64자)을 `값 [Dn]`으로 만들어 전체 로그우도로 비교 |
| 결정 단위 | 토큰 (탐욕) | 답 전체 |
| 보류 | `확인 불가`·`근거 충돌`도 문법의 일부 | 두 보류 문구도 후보에 포함 |
| 비용 | 토큰당 수 ms | 문단 550자 기준 후보 650개, KV 캐시 공유로 1~2초(17M, CPU) |
| 한계 | 첫 토큰에서 잘못 들어가면 못 돌아온다 | 두 값 결합 답은 열거 안 함. 짧은 답에 유리한 길이 편향 |

## 어디로 가나: 목표 구조

91M 생성 모델로 값을 "쓰게" 하는 대신, 같은 몸통에 **추출 헤드**를 얹어 값을 "고르게" 한다. 근거는 두 가지다. 한국어 소형 인코더는 이 크기에서 이미 추출형 독해를 잘 하고(RoBERTa-small 68M, KLUE-MRC EM 57; KoELECTRA-small, KorQuAD EM 82), 우리 17M 디코더는 구간을 강제로 고르게 해도 출처는 62/80 맞히면서 값은 0/80이다. 출처를 가리키는 능력과 값을 쓰는 능력이 분리돼 있다는 뜻이고, 후자를 생성에 맡길 이유가 없다.

<p align="center"><img src="docs/figures/system.png" width="920" alt="closed-network QA system"></p>

- `[Dn]`은 고른 구간이 속한 문단에서 결정되므로 번호를 생성하지 않는다.
- 보류 임계값은 검증 split의 risk–coverage 곡선에서 정하고 평가셋은 쓰지 않는다.
- 디코더 헤드는 설명·대화에만 쓴다.
- 권한 필터와 감사 로그는 모델 밖, 서비스 계층이다. 어떤 문서가 검색 대상이었고 답이 어느 문서의 몇 번째 글자에서 왔는지 남긴다.

### GraphRAG는?

GraphRAG·LightRAG는 대형 LLM이 문서에서 개체·관계를 뽑아 그래프와 요약을 만든다. 폐쇄망에 그런 모델이 없고, 91M으로는 안 된다. 그리고 GraphRAG-Bench 결과를 보면 단일 사실 조회에서는 기본 RAG보다 자주 못하고, 이득은 다중 홉·요약형 질문에만 있다.

그런데 공공기관 문서는 LLM 없이도 그래프가 나온다. 법령·규정은 **법 → 장 → 조 → 항 → 호** 계층이 있고, 별표·서식, 개정 이력, "제12조에 따른", "「○○법 시행령」 제3조" 같은 참조가 명시돼 있다. 이걸 파싱하면 결정적인 구조 그래프가 되고, BM25로 찾은 조문에서 참조 조문과 별표로 홉을 넓힐 수 있다. "출장비 상한은?"에 조문 하나가 답하지만 "그 상한의 예외는?"은 단서 조항과 별표까지 가야 하는데, 이 확장을 그래프가 맡는다. 모델은 여전히 그래프가 모아 온 근거에서 값만 고른다. 색인이 재현 가능하고 모델이 작아도 되는 이유다.

순서는 추출 헤드 결과를 본 뒤다. 평가셋에 다중 홉 문항을 따로 만들어 BM25 단독 대비 이득이 있을 때만 0.2에 넣는다.

### 다음 사전학습 (설계)

| 항목 | 지금 | 다음 |
|---|---|---|
| 데이터 | 위키 1억 토큰 | 위키 + 법령·행정규칙·판례 + 보도자료 약 10억 토큰, FineWeb-2 ko 필터·PII 제거·MinHash |
| 토크나이저 | 위키로만 학습 | 혼합 말뭉치로 재학습 (법령 문체에서 토큰 6~13% 절감) |
| step당 토큰 | 2,048 | 262,144 (batch 2 × accumulate 128) |
| lr / 스케줄 | 3e-4 cosine | 파일럿 {1e-3, 2e-3, 4e-3} 후 선택, WSD 디케이 20% |
| 정밀도 | FP32 | bf16 autocast + `torch.compile` (실측 +79%) |
| 사전학습 목표 | causal | causal + masked 혼합 (추출 헤드용) |
| 실험 | — | Muon, 24×576 deep-thin, logit softcap·U-net skip 묶음 (각 1일) |

### 배포 형태 (측정 기준)

91M · Q8_0 GGUF · llama.cpp CPU. 메모리 300 MB 미만, 질의당 p90 약 0.4초(M5, 4 스레드). Q8_0은 perplexity +0.1%로 greedy 출력이 PyTorch와 같았고, Q4는 출처 번호가 바뀌는 경우가 있어 평가셋 통과 전엔 안 쓴다.

조사 보고서: [폐쇄망 구조](reports/closed-network-architecture-survey.md) · [학습 기법](reports/oss-training-techniques.md) · [크기·추론](reports/oss-model-size-and-inference.md) · [한국어 OSS](reports/korean-oss-opportunities.md) · [사전학습 자료](reports/korean-pretraining-sources-survey.md)

## 지금 어디까지 왔나

<p align="center"><img src="docs/figures/pretrain-loss.svg" width="760" alt="pretraining loss"></p>

91M 사전학습이 돌고 있다 (검증 손실 9.17 → 3.36, 32,750 step). 그 전 세대인 17M은 근거 읽기에 실패했다. 자체 평가셋 310문항에서 값과 출처를 모두 맞힌 게 0개, 제한 디코딩을 걸어도 3개, 구간 단위 점수화로는 보류 22개만 맞히고 답 있는 문항은 0/228이다. 질문이 뭐든 학습 데이터에 있던 값을 그대로 뱉었는데, 학습 데이터가 금액 10개·이름 8개로만 돼 있어 복사가 아니라 암기를 배운 탓이다. 데이터를 다시 만들었고(16개 영역, 넓은 범위의 무작위 값) 91M으로 비교하는 중이다.

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

## 버전

0.1에서 시작한다. 고치거나 보태면 0.11, 0.12처럼 소수점 둘째 자리를 올리고, 새로운 것이 들어오면 0.2로 넘어간다. 1.0은 폐쇄망에 넣어도 되겠다 싶을 때 붙인다. 지금은 0.1이다.
