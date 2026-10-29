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
  <img alt="version" src="https://img.shields.io/badge/version-v0.1-blue">
  <img alt="license" src="https://img.shields.io/badge/code-Apache%202.0-green">
  <img alt="python" src="https://img.shields.io/badge/python-3.12-informational">
  <img alt="status" src="https://img.shields.io/badge/weights-not%20released-lightgrey">
</p>

<p align="center"><img src="docs/figures/pipeline.png" width="960" alt="trid pipeline"></p>

## 소개

trid는 인터넷이 끊긴 기관 내부 서버에서 규정·공고·보도자료를 근거로 질의에 답하는 한국어 소형 언어모델과 그 서비스 스택입니다. 근거 문단에 번호를 붙여 모델에 주면, 모델은 세 가지 중 하나로만 답합니다.

```
8만 원 [D3]      값과 출처
확인 불가         근거에 없음
근거 충돌         근거끼리 값이 다름
```

값은 디코딩 단계에서 **인용한 근거 문단에 실제로 있는 글자열만** 나오도록 제한하고, 출처 번호도 프롬프트에 준 번호만 허용합니다. 근거에 없는 숫자나 이름은 출력할 수 없습니다. 근거 안의 엉뚱한 구간을 고를 위험은 남아 있어 학습·색인 마스킹·권한 필터로 줄이고 보안 검증셋으로 측정합니다([SECURITY](docs/SECURITY.md)).

**외부 사전학습 가중치를 쓰지 않습니다.** trid는 Llama·Qwen 계열과 같은 구조를 따르되, 가중치는 무작위 초기화부터 출처가 기록된 데이터로 직접 학습합니다. 그래야 모델이 무엇을 배웠는지 데이터까지 추적할 수 있습니다. Qwen3·Kanana 같은 공개 모델은 같은 평가셋에서 **비교 기준선**으로만 측정합니다.

## 소식

- **2026-10-29** 개발 방향을 **외부 가중치 없는 자체 sLLM**으로 확정했습니다. 목표는 대한민국 공공기관·공기업 문서 전반에 대응하는 모델입니다. 앞으로 공개 가중치 모델은 비교 기준선으로만 씁니다.
- **2026-10-29** 비교 기준선으로 Qwen3-1.7B와 Kanana-1.5-2.1B를 튜닝 전후로 측정했습니다. 공개 모델 + LoRA는 RAG310에서 180~223/310입니다. 자체 모델이 도달해야 할 참고 수준이며 trid 모델이 아닙니다. [기준선](reports/open-weight-baseline/README.md) · [LoRA 측정](reports/open-weight-lora/)
- **2026-10-29** 자체 91M 몸통의 법령·보도·위키 혼합 자료 **10억 토큰 이어 사전학습** 완료를 확인했습니다. 추출 독해기 8조합 비교 중 2조합이 끝났습니다. [기록](docs/experiments/PRETRAINING.md)
- **2026-10-28** 법령·행정규칙·자치법규 3,000건에서 조문 구조 그래프 검색을 측정했습니다. 다중 근거 회수율은 BM25 단독 6.25%, 그래프 확장 16.25%입니다(12문단 기준). [측정](reports/graph-retrieval-v1.md)

## 모델

자체 개발 모델 하나만 있습니다. 가중치는 아직 배포하지 않습니다.

| 모델 | 파라미터 | 구조 | 상태 |
|---|---:|---|---|
| **trid-91M** | 91.2M | decoder-only, 12층, dim 768, 문맥 1,024, BPE 8,192. 무작위 초기화부터 직접 학습 | 실험. 추출 독해 헤드를 테스트 데모에 연결 |

모델 크기 확대와 공공기관 문서 수집 확대를 다음 단계로 진행합니다.

<details>
<summary><b>trid-91M 구조</b></summary>

<p align="center"><img src="docs/figures/architecture.png" width="860" alt="trid model architecture"></p>

| | |
|---|---|
| 구조 | RoPE, SwiGLU, RMSNorm(pre-norm), 입·출력 임베딩 공유. GQA·QK-norm 옵션 |
| 크기 | 91,245,312 (dim 768 · 12층 · 12 head · FFN 2,048 · 문맥 1,024) |
| 파라미터 분포 | 임베딩 6.3M (7%) · attention 28.3M (31%) · FFN 56.6M (62%) |
| 토크나이저 | byte-level BPE 8,192. 한국어 1.6~1.8자/토큰 |
| 사전학습 | 위키 1억 토큰(val loss 3.2064) → 법령·보도·위키 혼합 10억 토큰 이어 학습(혼합 val loss 3.3262 → 1.9629) |
| 학습 | PyTorch MPS/CUDA, bf16 autocast, `torch.compile`, WSD 스케줄 |
| 추론 | KV cache. 근거 복사 제한 디코딩, 구간 점수화, 추출 헤드(3분류 + 글자 구간) |
| 배포 형태 | llama.cpp `llama` 아키텍처 호환 GGUF. Q8_0에서 PyTorch와 greedy 출력 동일 |

</details>

## 주요 특징

- **폐쇄망 전제**: 추론 시 외부 API, 외부 임베딩 모델, 네트워크 호출이 없습니다. 형태소 분석기 모델 파일까지 패키지에 포함됩니다.
- **복사 보장 디코딩**: 값은 인용 문단의 부분 문자열로만 생성합니다(`trid/constrained.py`, `trid/span_scoring.py`). 비교 기준선 측정에도 같은 제약을 llama.cpp GBNF 문법으로 적용합니다.
- **보류 응답**: 근거가 없으면 `확인 불가`, 근거끼리 다르면 `근거 충돌`로 멈추도록 학습하고 평가합니다.
- **조문 구조 그래프 검색**: LLM 없이 법→장→조→항→호 계층과 명시 참조("제12조에 따른")를 파싱해 BM25 결과를 참조 조문·별표로 확장합니다.
- **감사 가능성**: 권한 필터, 감사 로그, 답이 어느 문서의 몇 번째 글자에서 왔는지 기록합니다. HWP·HWPX·PDF를 수집하고 색인 단계에서 개인정보를 마스킹합니다.
- **출처 추적 가능한 학습 데이터**: 학습 자료마다 출처 URL, 라이선스, SHA256을 `provenance.json`에 남깁니다.
- **공개 모델은 데이터 작성 보조로만**: 기관 문서 독해 예제 일부는 공개 모델(Kanana·Qwen)이 질문 초안을 쓰고 다른 모델이 검증합니다(`scripts/build_teacher_qa.py`). 답은 모두 원문 글자열을 그대로 복사하며, 공개 모델의 가중치나 생성 답변은 trid에 들어가지 않습니다. 해당 행에는 작성·검증 모델이 기록됩니다.

## 성능

모든 수치는 greedy 디코딩 기준입니다. **형식까지 엄격**은 값과 출처를 모두 맞히고 `값 [Dn]` 형식까지 지킨 경우만 정답으로 봅니다.

### RAG310: 근거가 주어진 질의응답

자체 작성 문서 160문항과 KLUE MRC dev 150문항으로 구성됩니다. 답이 있는 문항 228개, 답이 없거나 근거가 충돌하는 문항 82개입니다.

| 모델 | 디코딩 | 전체 310 | 답 있음 228 | 답 없음·충돌 82 |
|---|---|---:|---:|---:|
| trid-91M 생성 SFT (`public-dialogue-gpu-05-mix04`) | 근거 제한 | 71 (22.9%) | 29 | 42 |
| Qwen3-1.7B (튜닝 전) | 지시만 | 2 (0.6%) | — | — |
| Qwen3-1.7B (튜닝 전) | 근거 제한 문법 | 20 (6.5%) | 6 | 14 |
| Kanana-1.5-2.1B (튜닝 전) | 지시만 | 1 (0.3%) | — | — |
| Kanana-1.5-2.1B (튜닝 전) | 근거 제한 문법 | 63 (20.3%) | 52 | 11 |
| 참고: Qwen3-1.7B + LoRA (외부 가중치, trid 아님) | 근거 제한 문법 | 180 (58.1%) | 126 | 54 |
| 참고: Kanana-1.5-2.1B + LoRA (외부 가중치, trid 아님) | 근거 제한 문법 | 223 (71.9%) | 161 | 62 |

### 실제 기관 보도자료 probe (40문항)

| 모델 | 전체 40 | 답 있음 30 | 답 없음 10 |
|---|---:|---:|---:|
| trid-91M 추출 독해기 (`cpt-v1/head-v2/data-v4`) | 20 | 14 | 6 |
| Qwen3-1.7B (튜닝 전, 근거 제한 문법) | 0 | 0 | 0 |
| Kanana-1.5-2.1B (튜닝 전, 근거 제한 문법) | 15 | 15 | 0 |
| 참고: Qwen3-1.7B + LoRA (외부 가중치, trid 아님) | 30 | 20 | 10 |
| 참고: Kanana-1.5-2.1B + LoRA (외부 가중치, trid 아님) | 35 | 26 | 9 |

> [!NOTE]
> - RAG310은 모델 선택이나 튜닝에 사용하지 않았습니다. 다만 자체 작성 160문항은 학습 커리큘럼과 같은 개발자가 만든 합성 문서라, 실제 기관 문서 성능을 대표하지 않습니다.
> - 기관 probe는 여러 번 사용한 진단 자료이며 독립 최종 평가가 아닙니다. 추출 독해기의 `exact`와 생성 모델의 값·출처 채점은 같은 측정이 아닙니다.
> - 공개 기반 모델은 사전학습 단계에서 KLUE나 공개 보도자료를 이미 봤을 수 있습니다.
>
> 원본 출력과 채점: [`reports/open-weight-lora/`](reports/open-weight-lora/), [`reports/open-weight-baseline/`](reports/open-weight-baseline/), [`reports/reader-factorial-01-two-complete.json`](reports/reader-factorial-01-two-complete.json)

### 검색

같은 RAG310 질의에서 BM25 + Kiwi 형태소 분석의 Recall@3은 0.947로, 외부 임베딩 모델(KURE-v1)과 3pp 이내입니다. [보고서](reports/retrieval-benchmark-v1.md)

## 빠른 시작

### 설치

```sh
uv sync --frozen --python 3.12 --extra service
.venv/bin/python -m unittest discover -s tests      # CPU 회귀 테스트
```

`.hwp`는 [rhwp](https://github.com/edwardkim/rhwp) CLI, `.pdf`는 poppler `pdftotext`를 사용합니다. 오프라인 설치 절차는 [docs/SERVICE.md](docs/SERVICE.md)에 있습니다.

### 문서 색인과 질의

```sh
# 문서 폴더 → 색인 (HWP·HWPX·PDF·txt, 조문 구조 그래프, 개인정보 마스킹)
.venv/bin/trid ingest <문서폴더>/ --index index/

# 단일 질의
.venv/bin/trid ask --index index/ --run runs/<run> --user kim '숙박비 상한은 1박당 얼마인가?'

# 127.0.0.1 JSON API (bearer token, 감사 로그)
.venv/bin/trid serve --index index/ --run runs/<run>
```

### 평가

```sh
# trid 자체 모델: RAG310, 보안 검증셋(84건)
.venv/bin/python scripts/evaluate_rag_eval.py --run runs/<run> --decoding evidence --output reports/<name>.json
.venv/bin/python scripts/evaluate_security.py --run runs/<run> --decoding span --output reports/<name>.json

# 비교 기준선(공개 모델): llama-server를 띄운 뒤 같은 평가셋으로 측정
.venv/bin/python scripts/evaluate_external_chat.py --model-file <model.gguf> --model-id <id> --revision <rev> \
    --family qwen3 --output reports/<name>.json
```

## 학습

| 단계 | trid (자체) |
|---|---|
| 데이터 수집·정제 | `scripts/collect_*.py`, `scripts/build_pretraining_mix.py` |
| 사전학습 | `trid pretrain` ([기록](docs/experiments/PRETRAINING.md)) |
| 지도학습 | `scripts/train_public_dialogue.py`, 추출 헤드 `trid/extractive.py` |
| 기관 문서 독해 예제 | `scripts/build_teacher_qa.py` (원문 복사 답, 공개 모델은 질문 초안·검증만) |

`scripts/prepare_trid_cpt_sample.py`, `scripts/train_trid_cpt.py`, `scripts/train_trid_sft.py`, `scripts/distill_trid_chat.py`는 공개 가중치 모델을 학습하는 도구입니다. 위 비교 기준선(LoRA) 측정을 재현하는 용도로만 남겨 두며 trid 학습에는 쓰지 않습니다.

## 배포

자체 91M은 llama.cpp GGUF로 내보내 CPU에서 실행할 수 있습니다.

```sh
uv sync --extra gguf
.venv/bin/python scripts/export_gguf.py --run runs/<run> --output trid.q8_0.gguf --quant q8_0
```

Q8_0의 perplexity 변화는 약 +0.1%였고 f32·Q8_0 모두 PyTorch와 greedy 출력이 같았습니다([검증](reports/gguf-export-parity.json)). 현재 Python 서비스의 구간 점수화 경로는 가상 문서 질의 1건에 10.9~11.9초가 걸려 지연 목표를 아직 충족하지 못합니다([SERVICE](docs/SERVICE.md)).

## 저장소 구조

```
trid/                 모델 · 학습 · 추론 · 제한 디코딩 · 추출/검색 헤드 · 품질 필터
trid/service/         수집 · 조문 구조 · 색인 · 권한 · 감사 · 엔진 · 서버 · CLI
scripts/              수집기 · 정제 · 학습 파이프라인 · 평가 · GGUF 내보내기
tests/                모델·데이터·서비스 회귀 테스트
configs/              모델 구성 (91M, GQA/QK-norm 실험)
data/                 자체 작성 평가셋 · 보안 검증셋 (나머지 수집 자료는 git 제외)
reports/              실험 결과 원본 (JSON) 과 분석 (MD)
docs/                 서비스 · 보안 · 시연 · 연구 기록
docs/experiments/     실험별 상세 기록
docs/archive/         이전 버전 README · 모델 카드
deploy/               테스트 데모 배포 설정
```

## 문서

| 문서 | 내용 |
|---|---|
| [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md) | 설계 근거, 목표 구조, 실험별 수치와 실패 기록 (2026-10-29까지의 README) |
| [docs/SERVICE.md](docs/SERVICE.md) | 서비스 구성, 오프라인 설치, 실행 기록 |
| [docs/SECURITY.md](docs/SECURITY.md) | 위협 모델, 보안 검증셋 84건 |
| [docs/KOREAN.md](docs/KOREAN.md) | 한국어 전용 개발 기준 |
| [docs/SHARING.md](docs/SHARING.md) · [docs/SHOWCASE.md](docs/SHOWCASE.md) | 테스트 데모 공유, 기관 시연 |
| [docs/experiments/PRETRAINING.md](docs/experiments/PRETRAINING.md) | 사전학습, GPU 실행·복구 기록 |
| [docs/experiments/KOREAN_RAG.md](docs/experiments/KOREAN_RAG.md) | 근거 기반 질의응답 학습·평가 프로토콜 |
| [docs/experiments/](docs/experiments/) | 대화, 독해, 구간 추출, 최적화 실험 |
| [reports/](reports/) | 폐쇄망 구조·학습 기법·한국어 OSS 조사와 모든 실험 결과 |

## 버전과 Git 규칙

- 제품 버전은 **v0.1**입니다. 수정은 0.1x, 새 기능은 0.x로 올리고, 배포 기준을 충족할 때 1.0을 붙입니다.
- `cpt-v1`, `head-v2`, `curriculum-v4`, `gpu-05` 같은 이름은 학습 실행·데이터의 실험 식별자이며 제품 버전이 아닙니다.
- 검증을 마친 작업 단위마다 커밋합니다. 실험 결과는 원본 JSON을 `reports/`에 남기고, 실패한 실험도 지우지 않습니다.
- 학습 산출물(`runs/`)과 대용량 수집 자료(`data/`)는 git에 넣지 않습니다. 평가셋 원본만 `.gitignore` 예외로 관리합니다.

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
