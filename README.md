<h1 align="center">trid</h1>

<p align="center">
  <b>대한민국 공공기관 폐쇄망용 한국어 sLLM</b><br>
  기관 문서를 근거로 받아, 값을 지어내지 않고 원문에서 찾아 출처와 함께 답합니다.
</p>

<p align="center">
  📑 <a href="docs/RESEARCH_LOG.md">연구 기록</a> &nbsp;|&nbsp;
  📊 <a href="reports/">평가 보고서</a> &nbsp;|&nbsp;
  ⚙️ <a href="docs/SERVICE.md">서비스·설치</a> &nbsp;|&nbsp;
  🛡️ <a href="docs/SECURITY.md">보안 검증</a> &nbsp;|&nbsp;
  🧪 <a href="docs/SHARING.md">테스트 데모</a>
</p>

<p align="center">
  <img alt="license" src="https://img.shields.io/badge/code-Apache%202.0-green">
  <img alt="python" src="https://img.shields.io/badge/python-3.12-informational">
  <img alt="weights" src="https://img.shields.io/badge/weights-not%20released-lightgrey">
</p>

## 소개

trid는 인터넷이 끊긴 기관 내부 서버에서 규정·공고·보도자료를 읽고 질문에 답하는 한국어 소형 언어모델입니다. 검색한 근거 문단에 번호를 붙여 모델에 주면, 모델은 다음 세 가지 중 하나로만 답합니다.

```
8만 원 [D3]      값과 출처
확인 불가         근거에 없음
근거 충돌         근거끼리 값이 다름
```

- **외부 가중치 없음**: Llama·Qwen 계열과 같은 decoder 구조를 쓰되, 토크나이저와 가중치를 무작위 초기화부터 직접 학습합니다. 모델이 무엇을 배웠는지 데이터까지 거슬러 확인할 수 있습니다.
- **지어내지 않는 답**: 값은 인용한 근거 문단에 실제로 있는 글자열만, 출처는 프롬프트에 준 번호만 출력할 수 있도록 디코딩 단계에서 제한합니다.
- **답이 없으면 멈춤**: 근거가 없으면 `확인 불가`, 근거끼리 값이 다르면 `근거 충돌`로 답하도록 학습합니다.
- **폐쇄망 그대로 동작**: 추론에 외부 API, 외부 임베딩 모델, 네트워크 호출이 없습니다. HWP·HWPX·PDF 문서를 바로 색인합니다.
- **추적 가능한 학습 데이터**: 학습 자료마다 출처 URL, 라이선스, SHA256을 `provenance.json`에 남깁니다.

## 소식

- **2026-10-29** 개발 방향을 **외부 가중치 없는 자체 sLLM**으로 확정했습니다. 목표는 대한민국 공공기관·공기업 문서 전반에 대응하는 모델입니다. 공개 가중치 모델은 비교 기준선으로만 씁니다.
- **2026-10-29** Qwen3-1.7B와 Kanana-1.5-2.1B를 튜닝 전후로 같은 평가셋에서 측정해, 자체 모델이 도달해야 할 기준선을 세웠습니다. [기준선](reports/open-weight-baseline/README.md) · [LoRA 측정](reports/open-weight-lora/)
- **2026-10-29** trid-91M의 법령·보도·위키 혼합 자료 **10억 토큰 이어 사전학습**을 마쳤습니다. [기록](docs/experiments/PRETRAINING.md)
- **2026-10-28** 법령·행정규칙·자치법규 3,000건으로 조문 구조 그래프 검색을 측정했습니다. [측정](reports/graph-retrieval-v1.md)

## 성능

<p align="center"><img src="docs/figures/benchmark.png" width="960" alt="trid benchmark"></p>

- **RAG310**: 근거가 주어진 질의응답 310문항입니다. 자체 작성 기관 문서 160문항과 KLUE MRC dev 150문항으로 구성되며, 답이 있는 문항 228개와 답이 없거나 근거가 충돌하는 문항 82개가 섞여 있습니다.
- **기관 문서 probe**: 실제 정책브리핑 보도자료에서 만든 40문항입니다(답 있음 30, 답 없음 10).
- 값, 출처, `값 [Dn]` 형식을 모두 맞혀야 정답으로 칩니다. 회색 막대는 비교용 공개 모델이며 trid가 아닙니다.

<details>
<summary><b>상세 수치</b></summary>

| 모델 | RAG310 전체 | 답 있음 228 | 답 없음·충돌 82 | 기관 probe 40 |
|---|---:|---:|---:|---:|
| **trid-91M** | **71** | **29** | **42** | **20** |
| Kanana-1.5-2.1B + LoRA (기준선) | 223 | 161 | 62 | 35 |
| Qwen3-1.7B + LoRA (기준선) | 180 | 126 | 54 | 30 |
| Kanana-1.5-2.1B (기준선, 튜닝 전) | 63 | 52 | 11 | 15 |
| Qwen3-1.7B (기준선, 튜닝 전) | 20 | 6 | 14 | 0 |

- trid-91M의 RAG310은 생성 SFT 모델(`public-dialogue-gpu-05-mix04`), 기관 probe는 추출 독해기(`cpt-v1/head-v2/data-v4`, exact) 결과입니다.
- 공개 모델 LoRA는 trid 학습 커리큘럼 2,000예제로 250단계 학습했습니다. 모든 공개 모델에 같은 근거 제한 문법을 적용했습니다.
- RAG310은 튜닝이나 체크포인트 선택에 쓰지 않았습니다. 자체 작성 160문항은 학습 커리큘럼과 같은 개발자가 만든 합성 문서라 실제 기관 문서 성능을 대표하지 않습니다. 기관 probe는 여러 번 사용한 진단 자료입니다.
- 원본 출력과 채점: [`reports/open-weight-baseline/`](reports/open-weight-baseline/), [`reports/open-weight-lora/`](reports/open-weight-lora/), [`reports/reader-factorial-01-two-complete.json`](reports/reader-factorial-01-two-complete.json)

</details>

검색 단계는 병목이 아닙니다. RAG310 질의에서 BM25 + Kiwi 형태소 분석의 Recall@3은 0.947로, 외부 임베딩 모델(KURE-v1)과 3pp 이내입니다([보고서](reports/retrieval-benchmark-v1.md)). 남은 격차는 근거를 읽고 값을 고르는 모델의 독해력입니다.

## 모델 구성

### 전체 흐름

<p align="center"><img src="docs/figures/pipeline.png" width="960" alt="trid pipeline"></p>

공개 한국어 자료를 정제해 토크나이저와 모델을 처음부터 학습하고, 근거 질의응답으로 지도학습합니다. 서비스에서는 기관 문서를 검색해 근거 문단을 번호와 함께 프롬프트에 넣고, 근거 복사 디코딩으로 답을 만듭니다.

### 모델 구조

<p align="center"><img src="docs/figures/architecture.png" width="820" alt="trid model architecture"></p>

| | trid-91M |
|---|---|
| 파라미터 | 91,245,312 |
| 구조 | decoder-only Transformer. RoPE, SwiGLU, RMSNorm(pre-norm), 입·출력 임베딩 공유 |
| 크기 | 12층 · hidden 768 · attention head 12 · FFN 2,048 |
| 파라미터 분포 | 임베딩 6.3M (7%) · attention 28.3M (31%) · FFN 56.6M (62%) |
| 문맥 길이 | 1,024 토큰 |
| 토크나이저 | byte-level BPE, 어휘 8,192. 한국어 1.6~1.8자/토큰 |
| 옵션 | GQA(`kv_heads`), QK-norm. 기본은 꺼져 있음 |
| 독해 헤드 | 추출 헤드: 답 있음/없음/충돌 3분류 + 근거 내 글자 구간 시작·끝 |
| 실행 | llama.cpp `llama` 아키텍처 호환 GGUF. Q8_0에서 PyTorch와 greedy 출력 동일 |

### 답을 만드는 방식

세 방식 모두 근거에 없는 값은 나올 수 없습니다. 차이는 모델이 어디서 결정하느냐입니다.

| 방식 | 코드 | 동작 |
|---|---|---|
| 근거 복사 제한 디코딩 | `trid/constrained.py` | 토큰마다 "지금까지 생성한 글자열 + 이 토큰"이 인용 근거의 부분 문자열인지 검사합니다. 출처 번호도 프롬프트에 있는 것만 허용합니다. |
| 구간 점수화 | `trid/span_scoring.py` | 근거의 어절 경계에서 시작하는 후보 구간을 모두 `값 [Dn]`으로 만들고, 전체 로그우도로 비교해 고릅니다. |
| 추출 헤드 | `trid/extractive.py` | 생성하지 않고 근거 안의 시작·끝 위치를 직접 고릅니다. 출처는 고른 구간이 속한 문단에서 정해집니다. |

### 학습 데이터

<p align="center"><img src="docs/figures/training-data.png" width="960" alt="trid pretraining data"></p>

- **사전학습**: 한국어 위키백과(CC BY-SA), 국가법령정보센터 법령·행정규칙·자치법규(저작권법 제7조), 정책브리핑 보도자료(공공누리 제1유형)만 씁니다. 품질 필터, 개인정보 치환, MinHash 근사 중복 제거를 거칩니다.
- **지도학습**: 합성 기관 문서 기반 근거 질의응답 커리큘럼, KLUE MRC, TyDi QA 한국어, 정책브리핑 실제 보도자료에서 뽑은 근거 질의응답을 씁니다. 평가셋 문서와 겹치는 학습 지문은 제외합니다.
- **공개 모델의 역할**: 기관 문서 독해 예제 일부는 공개 모델이 질문 초안을 쓰고 다른 모델이 검증합니다(`scripts/build_teacher_qa.py`). 답은 모두 원문 글자열을 그대로 복사하며, 공개 모델의 가중치나 생성 답변은 trid에 들어가지 않습니다.

### 폐쇄망 서비스

<p align="center"><img src="docs/figures/system.png" width="960" alt="closed-network QA system"></p>

| 구성 요소 | 역할 |
|---|---|
| 수집 (`trid/service/ingest.py`) | HWP·HWPX·PDF·txt 텍스트 추출, 개인정보 마스킹 |
| 조문 구조 그래프 | LLM 없이 법→장→조→항→호 계층과 "제12조에 따른" 같은 명시 참조를 파싱 |
| 검색 | Kiwi 형태소 분석 + BM25로 문단을 찾고, 구조 그래프로 참조 조문·별표까지 확장 |
| 권한 필터 | 사용자가 볼 수 없는 문서를 검색 대상에서 제외 |
| 모델 | 근거 문단을 `[D1]…[Dn]`으로 받아 값과 출처를 선택하거나 보류 |
| 감사 로그 | 답이 어느 문서의 몇 번째 글자에서 왔는지, 누가 질의했는지 기록 |

## 빠른 시작

```sh
# 설치 (Python 3.12, uv)
uv sync --frozen --python 3.12 --extra service

# 문서 폴더 색인: HWP·HWPX·PDF·txt, 조문 구조 그래프, 개인정보 마스킹
.venv/bin/trid ingest <문서폴더>/ --index index/

# 질의
.venv/bin/trid ask --index index/ --run runs/<run> --user kim '숙박비 상한은 1박당 얼마인가?'

# 내부망 JSON API (127.0.0.1, bearer token, 감사 로그)
.venv/bin/trid serve --index index/ --run runs/<run>

# CPU 배포용 GGUF 내보내기 (llama.cpp)
uv sync --extra gguf
.venv/bin/python scripts/export_gguf.py --run runs/<run> --output trid.q8_0.gguf --quant q8_0
```

`.hwp`는 [rhwp](https://github.com/edwardkim/rhwp) CLI, `.pdf`는 poppler `pdftotext`를 사용합니다. 오프라인 설치와 서비스 설정은 [docs/SERVICE.md](docs/SERVICE.md), 평가 재현은 [docs/experiments/KOREAN_RAG.md](docs/experiments/KOREAN_RAG.md)에 있습니다.

## 라이선스

코드는 [Apache License 2.0](LICENSE)입니다. 학습 데이터는 위키백과(CC BY-SA), 법제처 법령·판례(저작권법 제7조), korea.kr 보도자료(공공누리 제1유형), KLUE(CC BY-SA), TyDi QA(Apache 2.0)를 사용하며 자료별 출처는 각 `provenance.json`에 있습니다. 비교 기준선으로 측정한 Qwen3와 Kanana-1.5는 Apache 2.0이며 trid 가중치에는 포함되지 않습니다. 학습된 가중치는 아직 배포하지 않습니다.

## 인용

```bibtex
@misc{trid2026,
  title  = {trid: A Korean small language model for grounded QA in air-gapped public-sector networks},
  author = {Eom, Gyuhyeon},
  year   = {2026},
  url    = {https://github.com/Gyuhyeon-Eom/trid}
}
```
