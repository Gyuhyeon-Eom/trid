# 검색 병목 진단 v1 (korean-rag-eval-v1)

`reports/korean-oss-opportunities.md` 5절의 첫 단계. 생성 전에 **근거 문서를 찾는 단계**가 병목인지 확인한다.
재현: `.venv/bin/python scripts/benchmark_retrieval.py --output reports/retrieval-benchmark-v1.json` (모델 없는 기계: `--skip-dense`). 수치 원본은 `reports/retrieval-benchmark-v1.json`.

## 설정

- 코퍼스: `data/korean-rag-eval-v1/test.jsonl`(읽기 전용, sha256 검증) 310문항의 근거 문단 전체를 텍스트 기준으로 중복 제거한 **421개 문단**.
- 질의: `gold`가 있는 **228문항**의 "질문:" 문장만 사용. 정답 문단은 해당 문항 `sources`의 문단(12문항은 정답 문단 2개).
- 검색기
  1. `bm25_kiwi`: bm25s + kiwipiepy 형태소. 색인 태그 = NNG·NNP·NNB·NR·NP(명사류), VV·VA(동사·형용사 어간), SN(숫자), SL(알파벳), SH(한자). 조사·어미·기호 제외.
  2. `bm25_space`: bm25s + 공백 어절(대조군).
  3. `kure`: nlpai-lab/KURE-v1 dense(코사인), sentence-transformers, CPU 4스레드.
  4. `hybrid`: (1)+(3)의 reciprocal rank fusion(k=60).
- 지표: Recall@k = 정답 문단 **전부**가 상위 k에 포함(정답 2개 문항은 @1이 구조적으로 0), MRR = 첫 정답 문단 역순위. 지연시간은 질의당 CPU(인덱스 시간 제외, 사전학습 작업과 동시 실행 상태라 절대값은 보수적).

## 결과 (전체, 228문항)

| 검색기 | R@1 | R@3 | R@5 | MRR | 질의당 평균/p90 (ms) | 색인 (s) |
|---|---|---|---|---|---|---|
| bm25_space (대조) | 0.518 | 0.654 | 0.702 | 0.624 | 0.04 / 0.07 | 0.02 |
| bm25_kiwi | 0.794 | **0.947** | 0.969 | 0.906 | 0.86 / 1.34 | 1.58 |
| kure (상한선) | **0.851** | **0.974** | 0.974 | **0.942** | 173 / 261 | 102.9 |
| hybrid | 0.833 | **0.974** | **0.987** | 0.935 | 174 / 262 | — |

### 출처별

| 검색기 | authored R@1 / R@3 / MRR (128) | klue_dev R@1 / R@3 / MRR (100) |
|---|---|---|
| bm25_space | 0.406 / 0.594 / 0.562 | 0.660 / 0.730 / 0.703 |
| bm25_kiwi | 0.680 / 0.922 / 0.865 | 0.940 / 0.980 / 0.958 |
| kure | 0.766 / 0.961 / 0.918 | 0.960 / 0.990 / 0.973 |
| hybrid | 0.727 / 0.961 / 0.897 | 0.970 / 0.990 / 0.982 |

### 과제별 (R@1 / R@3)

| 과제 (n) | bm25_space | bm25_kiwi | kure | hybrid |
|---|---|---|---|---|
| rag_klue (100) | 0.66 / 0.73 | 0.94 / 0.98 | 0.96 / 0.99 | 0.97 / 0.99 |
| rag_value (32) | 0.41 / 0.44 | 0.78 / 0.97 | 0.81 / 0.94 | 0.88 / 0.97 |
| rag_name (24) | 0.50 / 0.58 | 0.92 / 0.92 | 0.96 / 1.00 | 0.92 / 1.00 |
| rag_noise (24) | 0.38 / 0.83 | 0.67 / 1.00 | 0.96 / 1.00 | 0.75 / 1.00 |
| rag_table (20) | 0.65 / 0.70 | 0.95 / 0.95 | 0.90 / 1.00 | 0.95 / 1.00 |
| rag_revision (16) | 0.31 / 0.63 | 0.31 / 0.94 | 0.50 / 1.00 | 0.38 / 0.94 |
| rag_combine (12, 정답 2개) | 0.00 / 0.33 | 0.00 / 0.58 | 0.00 / 0.75 | 0.00 / 0.75 |

## 해석

- **형태소 토큰화가 결정적**이다. 공백 어절 BM25(R@3 0.65)에서 Kiwi 형태소 BM25(R@3 0.95)로 +0.29. "출장비는/출장비를"이 한 토큰이 되는 효과가 그대로 나타난다.
- **검색은 병목이 아니다.** 외부 가중치 없이 BM25(kiwi)만으로 상위 3개 안에 정답 문단이 94.7% 들어간다. KURE-v1 상한선과의 차이는 R@3 기준 2.6%p(6문항), R@1 기준 5.7%p. 같은 평가셋에서 **정답 문단이 프롬프트에 이미 주어진 상태**의 생성 strict 정확도가 310문항 중 3(`rag-eval-v1-baseline-v0.3-candidate-evidence.json`)이므로, 손실은 검색이 아니라 읽기·복사 단계에서 난다.
- BM25(kiwi)가 상위 3개에서 놓친 12문항 중 4개(C-005, C-038, D-038, D-039)는 KURE도 놓친다. `rag_revision`(같은 조문의 개정 전후 두 판본)과 `rag_combine`(정답 문단 2개)은 어느 검색기든 R@1이 낮아 **top-k≥3을 생성기에 넘기는 설계**가 필요하다.
- 하이브리드는 R@5에서만 소폭 우세(0.987)이고 R@1·MRR은 KURE 단독보다 낮다. 421문단 규모에서 RRF의 이득은 없다.
- 지연: BM25(kiwi) 질의당 1ms 미만, KURE 173ms(CPU, 동시 부하 중). 폐쇄망 CPU 제품에서 dense는 200배 비용에 R@3 +2.6%p다.

## 권고

정책 선택지 2(외부 가중치 불허, BM25 kiwi 단독)로 시작해도 검색 재현율 손실은 R@3 기준 약 2–3%p다. 임베딩 정책 결정은 소유자에게 남기되, 이 수치로는 외부 임베딩을 제품에 넣을 근거가 약하다. 다음 단계는 BM25(kiwi) top-3를 프롬프트에 넣었을 때의 생성 strict 정확도(`evaluate_rag_eval.py`)와 oracle 문단 대비 차이를 재는 것이다.

## 라이선스·사용 범위

- **KURE-v1은 개발 중 상한선 측정에만 사용했고 제품에 포함하지 않는다.** 가중치는 HF 캐시(`~/.cache/huggingface/hub/models--nlpai-lab--KURE-v1`)에만 있으며 저장소에 복사하지 않았다.
- bm25s: MIT — https://github.com/xhluca/bm25s
- kiwipiepy: Apache-2.0 — https://github.com/bab2min/kiwipiepy
- sentence-transformers: Apache-2.0 — https://github.com/UKPLab/sentence-transformers
- KURE-v1: MIT — https://huggingface.co/nlpai-lab/KURE-v1
- 의존성은 `pyproject.toml`의 `retrieval` extra에 고정(bm25s 0.3.15, kiwipiepy 0.24.1, sentence-transformers 6.2.0).
