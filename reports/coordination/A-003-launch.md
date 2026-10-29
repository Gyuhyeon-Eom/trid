# A-003 시작 보고: 91M / 390M MHA 1B 토큰 본 비교

## 1. 세션 / 역할 / 작업 / 상태

- 세션 01a18739-bcc6-77fc-b2a0-14d216f1becb / A / A-003 / **두 job 모두 실제 학습 중.**
- 확인 근거: CloudWatch 로그에 첫 평가 행이 찍혔다(아래). job 생성 요청만 성공한 상태가 아니다.
- 지시: A-002-review(A-003 승인).
- 지킨 사항:
  - 파일럿 20M 가중치는 쓰지 않았다. 두 arm 모두 새 run 이름, random 초기화다.
  - C·B 자원은 변경하지 않았다.
  - 최종 시험 적합성은 unknown으로 유지한다.

## 2. 실행 중인 job

| arm | job | 인스턴스 | 학습 시작 확인 (로그) |
|---|---|---|---|
| 91M MHA | `trid-pt-a003-1b-91m-mha-1030035759` | ml.g6e.xlarge (L40S 1장) | `pretrain: 91,245,312 parameters \| cuda \| bf16 compile \| 0/15259 steps \| 65,536 tokens/step \| epoch` → `step 100/15259 \| train 5.4983 \| valid 5.1919 \| 110,212 tokens/s` |
| 390M MHA | `trid-pt-a003-1b-390m-mha-1030035815` | ml.g6e.2xlarge (L40S 1장) | `pretrain: 390,128,640 parameters \| … \| 0/15259 steps …` → `step 100/15259 \| train 5.2683 \| valid 4.9724 \| 34,482 tokens/s` |

- 100단계 처리량은 compile·warmup 구간이 포함된 세션 평균이다. 지속 처리량은 A-002 기준 약 17만 / 4.5만 토큰/s다.
- 예상 학습 시간은 91M 약 1.7~2시간, 390M 약 6.3~6.5시간이다. 평가와 세션 재시작(compile)을 포함한 추정치다.
- 두 job의 공통 조건:
  - **종료 조건:** 15,259단계 완료(MAX_SESSIONS=0, 전체 진행), 또는 MaxRuntime 43,200초(12시간, 작업 안전 기한). job이 끝나면 인스턴스가 종료된다.
  - **세션:** `--stop-after 2000` 단위. 세션이 끝날 때마다 체크포인트를 남기고, 같은 프로세스 안에서 로컬 체크포인트로 재개해 이어간다. 로컬 디렉터리는 S3와 동기화되므로, job이 중단되면 같은 run 이름으로 새 job을 띄워 S3에서 재개한다.
  - **기록:** `reports/coordination/sm-jobs-A.jsonl`
- 코드:
  - `s3://trid-training-148761639313/corpus/code/pretrain/fe74049b6803e20c/`
  - code_sha `fe74049b6803e20c`, git HEAD `0da169b`, 작업 트리 변경 없음
  - CUDA fused AdamW 재개 수정이 포함된 코드다.
- 체크포인트: S3 `corpus/runs/pretrain/a003-1b-91m-mha/`, `corpus/runs/pretrain/a003-1b-390m-mha/` (CheckpointConfig `/opt/ml/checkpoints`)
  - 1,000단계마다 체크포인트를 남긴다.
  - S3 동기화가 로컬 삭제를 반영하지 않아 지난 체크포인트가 S3에 누적된다. 390M은 1개당 약 4.7GB다.

## 3. 고정한 공통 설정 (두 run의 `training.json`에서 확인)

| 항목 | 값 |
|---|---|
| 데이터 | `mix-v3-basetok-tokens`, manifest sha256 `514e0f93ae6d5429546b8c73e70bf85efbf004550d7e208ab20011f31165e904` (S3 `corpus/tokens/mix-v3-basetok-tokens/`, 시작 시 bin sha256 검증) |
| 토크나이저 / 문맥 | 8,192 BPE / 1,024 |
| 초기화 / seed | random / 291101 |
| 샘플링 | epoch, 출처 가중치 없음. epoch 창 474,056개 → 1B 토큰은 약 2.06 epoch |
| 배치 | 16 × 누적 4 × 1,024 = 65,536 토큰/단계, 15,259단계 = 1,000,013,824 토큰 |
| optimizer | AdamW β(0.9, 0.95), weight decay 0.1 (1차원 파라미터·norm 제외), fused (CUDA), grad clip 1.0 |
| 학습률 | 3e-4 |
| 스케줄 | WSD: 1~100단계 선형 warmup, 101~12,207단계 일정, 12,208~15,259단계(3,052단계, 20%) 0까지 선형 감쇠 |
| 정밀도 | bf16 autocast, torch.compile |
| 평가 / 체크포인트 | 100단계마다 검증 8 batch / 1,000단계마다 + 세션 끝(2,000단계 단위) |
| 기록 | 평가 시점마다 grad_norm, 구간 최대 grad_norm, 최대 CUDA 메모리 |

## 4. 사전 등록: 학습 이후 규칙 (결과를 보기 전에 기록)

1. **사전학습 비교 대상:** 두 arm 모두 **15,259단계 최종 체크포인트** `checkpoints/step-00015259/model.safetensors`를 쓴다. 검증 손실 기준 best 파일(`model.safetensors`)이나 중간 step은 쓰지 않는다.
2. **SFT:** `scripts/sm_sft_eval.sh` → `scripts/train_public_dialogue.py train`.
   - 데이터: `data/public-dialogue-mix-04` (S3 `corpus/eval-inputs/a003/`)
   - 학습: 1,200단계, batch 8, micro batch 2, lr 1e-4, seed 271030, 검증 100단계마다 64행, 문맥 1,024, example 정규화
   - 두 arm에 완전히 같은 설정을 쓴다. 기존 gpu-05-s1200과도 같은 설정이다.
   - **SFT 체크포인트 선택:** 양쪽 모두 기존 `trid.training.train` 규칙(검증 손실 best, 0~1,200단계 중 100단계 간격 평가)으로 저장된 `sft/model.safetensors`를 쓴다.
   - 실행: 두 arm을 서로 다른 GPU job으로 동시에 돌린다(L40S).
3. **개발 평가:** SFT 체크포인트로 같은 CUDA 장치에서 실행한다.
   - RAG310 `evaluate_rag_eval.py`: evidence(근거 제한 디코딩)와 free, greedy, 64토큰
   - KLUE dev 질문 민감도 `evaluate_question_sensitivity.py`
4. **주지표:** RAG310 답 있음 228문항의 `strict` (값·출처 모두 정답, `<값> [Dn]` 형식 파싱), **evidence 디코딩**.
   - 보조 지표: free strict, 출처 일치 + 정답 언급(`summarize_open_weight_baseline.grounded_rate`), "답 있는데 보류", "답 없는데 답함", KLUE dev 정답·보류·같은 답 반복.
5. **개발 후보 판정 (A-001-review 기준, 390M vs 91M):** 아래를 모두 만족하면 390M을 개발 후보로 본다.
   - 주지표 +5%p 이상(답 있음 228문항 기준 12문항 이상)
   - 같은 문항 짝지은 정확 McNemar p<0.05
   - "답 있는데 보류"와 "답 없는데 답함" 각각 5문항 초과 악화 없음
6. **해석 범위:** seed 1개이고 RAG310은 반복해서 써 온 개발셋이다. 이 결과로 최종 공공업무 우위를 주장하지 않는다. 결과를 본 뒤 합격선을 바꾸지 않는다.

## 5. 이후 작업과 관찰

- **진행 확인:** 주요 체크포인트(2,000단계 세션 경계)마다 진행 상황, 예상 완료 시각, 사용 자원을 A-003.md 작성 시점까지 기록한다.
- **중단 기준:**
  - finite한 일시적 기울기 peak만으로는 중단하지 않는다.
  - NaN/Inf, 반복 실패, 체크포인트 손상이 나오면 원인을 고친 뒤 같은 조건으로 재개하고 실패 기록을 보존한다.
- **91M이 먼저 끝나면** 그 arm의 SFT·평가 job을 먼저 시작한다. 두 arm의 조건은 같다.
- **이번에 추가한 코드** (진행 중인 사전학습에는 영향 없음, 아래 커밋):
  - `scripts/sm_sft_eval.sh`: 최종 step 가중치 다운로드와 state.json 단계 검증, SFT, 평가, 가중치 sha256 기록
  - `scripts/launch_sm_pretrain.py`: `--entry`와 SFT·평가 파일을 코드 업로드에 포함
- **평가 입력:** S3 `corpus/eval-inputs/a003/`
  - `public-dialogue-mix-04` train/valid/provenance
  - `korean-rag-eval-v1` test/provenance/ATTRIBUTION
  - `klue-mrc-source/dev.json`

SUPERVISION_REPORT_READY A-003-launch
