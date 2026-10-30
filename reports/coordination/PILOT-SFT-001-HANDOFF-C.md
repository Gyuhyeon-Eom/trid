# PILOT-SFT-001 → C 테스트 인계 (PIPELINE-3-001)

- **작성:** A (01a18739), 2026-10-30 14:59 KST
- **소유 경계:**
  - **C가 소유하는 것:** 고정 평가 실행, 채점, 회귀 판정을 C가 단독으로 맡는다. 다음 평가의 새 실행도 C 소유다.
  - **A가 마무리하는 것:** 이미 진행 중인 pilot_dev 진단 생성 1건은 A가 끝내고 결과를 넘긴다. C는 이것을 다시 돌리지 않는다.
- **블라인드 원칙:** 모델명과 매핑은 A만 갖는다. C에게는 보이지 않는다.
- **금지 사항:** 학습이 끝난 뒤 정답이나 평가셋을 바꾸지 않는다.

## 1. 학습 산출물 (고정)

| 항목 | 값 |
|---|---|
| 튜닝 모델 | `pilot-sft-001-a-qwen3-1p7b`. Qwen3-1.7B + LoRA를 병합한 모델 |
| 기반 | `Qwen/Qwen3-1.7B` HF rev `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, Apache-2.0. 로컬 사본은 `runs/open-weight-base/qwen3-1.7b-hf-rev70d244cc/`이고 safetensors 두 개의 sha가 HF LFS oid와 일치한다(`169ad53e…`, `912becff…`). |
| 병합 가중치 | `runs/pilot-sft-001-a-qwen3-1p7b/train/merged/model.safetensors`, sha256 `82f09136206a7be79a014581ce1d550e399742259598cfffd2a828415b8150a8` |
| adapter | `runs/pilot-sft-001-a-qwen3-1p7b/train/adapter/adapter_model.safetensors`, sha256 `3990cde0…9177`. config `90add4db…` |
| 튜닝 GGUF (평가·서빙용) | `runs/pilot-sft-001-a-qwen3-1p7b/pilot-sft-001-a-qwen3-1p7b.q8_0.gguf`, sha256 `0c78b0c2ace93fda8b3ce1de53441ec006afcb7f89638d79d8ef896b5a9432af` |
| 원본 GGUF (같은 변환기) | `runs/open-weight-base/Qwen3-1.7B-rev70d244cc-localconv-Q8_0.gguf`, sha256 `f2ba848c65c2e135495656fc91774b82a10d1f9267671609fad54684c9db1c38` |
| 공식 GGUF (기존 기준선) | `runs/open-weight-base/Qwen3-1.7B-Q8_0.gguf`, rev `90862c4b…`, sha256 `061b54da…cb1a` |
| 변환 | llama.cpp `8ca4060`의 `convert_hf_to_gguf.py --outtype q8_0`. 원본과 튜닝본에 같은 변환기를 썼다. |
| tokenizer / template | tokenizer.json sha256 `aeb13307…dae4`(원본과 같음). chat template은 원본 `tokenizer_config.json`과 바이트 단위로 같고, `chat_template.jinja` sha256은 `a55ee1b1…74d8`. GGUF에도 내장돼 있다. |
| 학습 manifest | `reports/coordination/PILOT-SFT-001-A-manifest.json`, sha256 `8f0c41cd…b95b`. C017 `pilot.jsonl`(`6bd0bdda…e291`)에서 split=train 57행만 2 epoch 학습했다. pilot_dev 18행은 검증 손실 기록만 하고 학습하지 않았다. |
| 학습 조건 | 15단계, 배치 8(micro 1), lr 2e-4, warmup 5% + cosine, LoRA r16·α32·dropout 0.05(q,k,v,o,gate,up,down), bf16, max_tokens 6144(잘림·탈락 0), seed 20261030, 답 토큰에만 loss |
| 시스템 지시 | C 사례 학습에는 `trid.case_prompts.CASE_SYSTEM`을 썼다. 값 형식 평가 지시(`evaluate_external_chat.SYSTEM`)는 학습에 쓰지 않았다. |
| 학습 기록 | `runs/pilot-sft-001-a-qwen3-1p7b/train/training.json`(`e6083090…`), `train.log`. S3 `corpus/runs/pilot/pilot-sft-001-a-qwen3-1p7b/`, `DONE.json` |
| 학습 job | `trid-pt-pilot-sft-001-a-qwen3-1p7b-1030133312`, ml.g6e.xlarge, Billable 345초. 코드 `e6a95a1`, code sha `603d86b9` |
| 마지막 검증 손실 (task별, 기록만) | official 0.162, plain 0.916, code 1.118, table 1.691. 원본의 검증 손실은 측정하지 않았다. |

## 2. 서빙·평가 실행 명령 (기준선과 같은 조건)

- **서버:** `runs/tools/llama.cpp-8ca4060/llama-server -m <gguf> --host 127.0.0.1 --port <port> -c 4096 -np 1 -t 4 -tb 4 --jinja`
  - 바이너리 sha256: `f7b5362c…9bdc`
  - pilot_dev처럼 긴 문항은 `-c 8192`를 쓴다.
- **고정 평가:** `.venv/bin/python scripts/evaluate_external_chat.py --url http://127.0.0.1:<port> --model-file <gguf> --model-id <id> --revision <rev> --family qwen3 --data <set> --output <new file>`
  - 조건: prompt_only, grammar_instructed, evidence_constrained (기본 3조건)
  - 대상: RAG310 `data/korean-rag-eval-v1`, probe `data/institution-qa-frozen-01`
  - 자유 응답: `--freeform-prompts reports/public-dialogue-01-after.json`, `reports/open-weight-baseline/counterfactual-prompts.json`
  - 순서대로 한 번에 실행하는 스크립트: `scripts/eval_open_weight_suite.sh`(A 작성)
- **주의 1: 지표를 섞지 않는다.** 기존 자체 모델 `strict`(evidence 디코딩)와 공개 가중치 기준선의 `content_strict`(출처 표기 순서를 허용)는 서로 다른 지표다.
- **주의 2: 동시 실행 금지.** 같은 Mac에서 llama.cpp 평가를 동시에 돌리면 문항당 시간이 약 4~8배 늘어난다. 실측: 단독 1.6초 → 동시 2개 6~14초. 순차로 실행한다.

## 3. 평가 진행 상태 (14:59 기준)

| 실행 | 상태 | 산출물 |
|---|---|---|
| pilot_dev 진단: 원본(localconv), `-c 8192`, CASE_SYSTEM, 512토큰 | **완료** 18/18 | `reports/open-weight-pilot/pilot-dev-qwen3-1.7b-base-localconv.json` |
| pilot_dev 진단: 튜닝본 | **A 진행 중**. port 51916, pid 89476, 4/18 | `reports/open-weight-pilot/pilot-dev-qwen3-1.7b-pilot-sft-001.json`(완료 시 생성). 예상 약 15분 |
| 고정 평가(RAG310·probe·설명·반사실): 원본 localconv와 튜닝본 | **미실행. C 소유 새 실행** | — |
| 고정 평가: 공식 GGUF 원본 | 기존 실측 있음 | `reports/open-weight-baseline/rag310-qwen3-1.7b-q8.json`, `reports/open-weight-lora/rag310-qwen3-base-constrained.json`, `probe-*`, `public-dialogue-assessment.json` |

**고정 평가를 하지 못한 원인:**
- A가 13:34~14:34에 원본·튜닝본 고정 평가를 동시에 돌렸다. 그러다 다음 두 문제로 산출물 없이 끝났다:
  - 셸 1시간 제한에 걸려 원본 평가가 강제 종료됐다.
  - 동시 실행 경합 때문에 너무 느려서 튜닝본 평가를 A가 직접 정지했다.
- 이 실행들은 산출물이 없으므로 "이미 실행분"에 해당하지 않는다. PIPELINE-3-001에 따라 새 실행은 C 소유다.
- pilot_dev 진단이 끝나면 A의 감시 스크립트가 남은 고정 평가 단계를 시작 전에 멈춘다. C와 중복 실행되는 일은 없다.

## 4. 블라인드 dev 패키지 (C-019 8절)
- **준비:** 튜닝본 pilot_dev가 끝나는 즉시 A가 만든다. 생성 도구: `scripts/make_blind_pair_package.py`
  - 쌍 순서와 답 순서를 모두 무작위로 섞는다.
  - 모델명을 지운다.
  - 매핑 sha256 commitment를 패키지에 넣는다.
- **위치 (15:12 전달 완료):** `reports/coordination/PILOT-SFT-001-blind/pilot-dev-blind-pairs.json`
  - 18쌍, sha256 `85ea1c4ba730c1e8ac9055ed9d4dc1fc50516519ea94b2e4a85c890cdd8eafb6`
  - 매핑 commitment `f39df736ef1e2079ddc6b1e1cef7a95a0cc732db1f5ff8cce30d534e9364d15e`
  - `pilot_sha256`은 고정 pilot v1 `6bd0bdda…`과 같다.
- **pilot_dev 진단 생성 상태:** 두 모델 모두 끝났다(15:11).
  - 이름이 붙은 원본 결과(`reports/open-weight-pilot/pilot-dev-*.json`)는 C 채점이 끝날 때까지 커밋하지 않는다. C는 열지 않는다.
  - A의 남은 고정 평가 단계는 시작 전에 멈췄다.
- **매핑:** A만 보관한다(`runs/`, Git 밖). **C는 열지 않는다.** C가 채점표를 넘기면 A가 공개하고 집계한다.
- **참고 진단:** 숫자 커버리지와 근거 없는 숫자 같은 결정적 지표는 진단 전용이다. 블라인드를 지키기 위해 C 채점이 끝날 때까지 공개하지 않는다.

## 5. 실험 한계 (판정할 때 반드시 반영)
- 답 토큰의 89%가 공식 해석례 결론·이유 추출이다. 효과는 주로 이 형식에 대한 것이다.
- 근거 충돌 사례 0건, 조문 dev 0건, 독립 검증 0건이다. 852 예약 계열도 미확인이다.
- 해석례는 해석일 기준이다. 현행 법령 여부는 확인하지 않았다.
- 15단계짜리 짧은 pilot이다. 상용 성능 목표가 아니라 효과·회귀를 진단하는 실험이다.

## 6. 맥미니
- 맥미니는 S1 감시 전용(root 임시 login 세션)이다. root 발행 가드를 유지한다(MACMINI-DEPLOY-002).
- 이 pilot과 평가에는 관여하지 않는다.
