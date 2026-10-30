# PILOT-SFT-001 (A): 1.7B pilot LoRA 사전 기록, 비용 사전 검산, 제한 권한 인증 절차

- **세션:** 01a18739 / A
- **상태:** 준비 완료, **C017 pilot manifest 대기**
- **입력 추정 금지:** manifest를 받기 전에는 행 수, 단계 수, 비용을 추정하지 않는다. 아래 수치는 모두 상한이다.
- **승인 범위:** 개발용 소규모 특화 실험만 조건부 승인됐다. 제품 품질 승인이 아니다.

## 1. 입력 조건 (C017 manifest가 갖춰야 할 것)
- **포함:** 관련성 높음·중간, 조문·표·코드 사례 중 AI 검토가 끝나고 해시가 맞는 것
- **필수 표시:** 중복 제거, 평가 제외 표시, 시점·기관 표시
- **제외:**
  - 미검토 276건
  - 관련성 미판정·낮음
  - 폐지 의심
- **같은 문서의 원본 자료와 쉬운 설명은 같은 split에 둔다.**
- **시점 표시:** 2015년 이후 해석례도 현행 여부는 확인되지 않았다. 학습 데이터에 "현행"이라고 표시하지 않는다.
- **형식:** A의 `scripts/train_open_weight_lora.py` manifest(JSON)
  - 파일마다 `path`(manifest 기준 상대 경로), `sha256`, `adapter: case`, `epochs`
  - C 사례 행에는 `question`, `context`, `target`, `task_type`, `id`가 있어야 한다.
  - A는 C 파일을 수정하지 않는다. 형식이 맞지 않으면 메시지로 요청한다.
- **토큰 길이:**
  - `max_tokens` 4096을 넘는 행은 학습 스크립트가 버리고 그 수를 기록한다.
  - C-014 기준으로 표 사례 9건 중 3건이 4,096을 넘었다(TRID-BASE-CUTOVER-001 10절).

## 2. 사전 고정 설정 (Qwen3-1.7B 우선)

| 항목 | 값 |
|---|---|
| 기반 | `Qwen/Qwen3-1.7B` HF rev `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, Apache-2.0. job 안에서 이 revision으로 내려받는다. |
| LoRA | r 16, alpha 32, dropout 0.05, q/k/v/o/gate/up/down |
| 최적화 | AdamW, lr 2e-4, warmup 5% + cosine, grad clip 1.0, bf16, gradient checkpointing |
| 배치 | 8 (micro 1, 누적 8) |
| 문맥 | `max_tokens` 4096, 답 토큰에만 loss |
| seed | 20261030 |
| 노출 (반복 금지) | 파일별 `epochs`는 최대 2다. **작은 자료를 추가로 반복하지 않는다.** 단계 수 = ⌈(행 수 × epoch) / 8⌉로 고정하고 평가 점수로 고르지 않는다. 파일별 답 토큰 점유율을 기록한다. |
| 검증 | manifest의 valid 파일로 task별 검증 손실을 기록만 한다. 선택에 쓰지 않는다. |
| 중단 기준 | ① 손실이 유한하지 않으면 즉시 중단(저장 안 함). ② 5단계 시점의 예상 학습 시간이 `--max-train-seconds` 2,400초를 넘으면 중단(저장 안 함). ③ MaxRuntime 3,600초가 하드 정지다. ④ 재시도 0회, 자동 재개 없음. |

## 3. 실행 형태 (클라우드 단일 완결 job)
- **진입점:** `scripts/sm_lora_pilot.sh`(커밋 `71cfeed`). 한 job 안에서 다음을 모두 끝낸다:
  - 기반 모델 다운로드(고정 revision)
  - 학습
  - 병합
  - `DONE.json` 기록(manifest 해시, 병합 가중치 sha256)
- **출력 위치:** `s3://trid-training-148761639313/corpus/runs/pilot/<run>/`
- **인스턴스:** ml.g6e.xlarge 온디맨드 1대
- **실행 주체:**
  - 기동은 맥북에서 기존 A 경로(`launch_sm_pretrain.py --entry sm_lora_pilot.sh`)로 한 번만 한다. 맥북 장시간 MPS 학습은 기본 운영으로 쓰지 않는다.
  - 맥미니 root 세션으로는 발행하지 않는다.
- **로컬 검증:** 맥북에서 학습 → 병합 → GGUF → llama.cpp → 평가 경로를 확인했다. MPS 4단계, 학습 데이터 아님.

## 4. 평가 (원본 대비 동일 평가, 필수)
- **비교 대상과 경로:**
  - 원본: 같은 HF rev `70d244cc…`
  - 튜닝본: 병합 모델
  - 둘 다 **같은 변환기로 Q8_0 GGUF로 만든다**(맥북 llama.cpp `8ca4060`, 각 약 30초).
  - 둘 다 기준선과 같은 llama.cpp 런타임과 `evaluate_external_chat.py`를 쓴다.
- **평가 항목:**
  - RAG310 전체 × 3조건
  - 기관 probe 40
  - 공개 설명 8 + 회귀 3, 반사실
  - C가 provenance를 분리해 넘긴 held-out(있으면)
- **변환 경로 일치:** 공식 GGUF(rev `90862c4b…`)로 낸 기준선 수치와 로컬에서 변환한 원본 수치를 함께 보고한다. 변환 경로 차이가 결과를 바꾸지 않는지 확인하기 위해서다.
- **관문:** TRID-BASE-CUTOVER-001 11절의 채택 관문을 그대로 쓴다. 보류·허위 응답 개선이 정답 수 개선보다 우선한다.
- **평가 시간 [INFERENCE]:** 맥북 CPU llama.cpp로 모델당 약 15~25분이다. 기준선 실측(0.6B 10분 42초, 1.7B 문항당 1.6초)에서 추정했다.

## 5. 비용 사전 검산 (A 예약 90,000원 안)

| 항목 | 원 |
|---|---|
| 기존 지출: 스모크 + 관측기 | 612 |
| S1-91M 실제: Billable 2,676초 × $3.2032/h × 1,650 | 3,929 |
| S1-390M 최악 보전: MaxRuntime 18,000초 | 28,427 |
| S1 스토리지·S3 예비 | 1,021 |
| **pilot 최악:** ml.g6e.xlarge 3,600초 + ML 스토리지 1시간 + S3(병합 3.4GB + adapter) 1개월 | 5,285 + 36 + 140 = **5,461** |
| **pilot 이후 잔여 (최악 기준)** | **약 50,550** |

- 80% 선(72,000원)에 한참 못 미친다.
- S1-390M이 끝나면 실제 Billable로 다시 정산한다(BUDGET-RUN-A).
- 이 pilot 1회 외의 추가 job, 재시도, 반복 실행은 다시 예약을 받는다.

## 6. 맥미니 무인 발행에 필요한 제한 권한 인증 절차 (부모 보고용)
- **현재:** 맥미니는 root `aws login` 세션이다. 감시 전용으로만 쓴다. driver의 `identity_guard`가 root에서는 발행 캠페인을 거부한다.
- **root는 권한을 좁힐 수 없다.** 따라서 무인 발행에는 별도 IAM 주체가 필요하다.
- 아래 절차는 장기 액세스 키 없이 `aws login`을 쓴다(공식 문서: [AWS CLI console credentials login](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sign-in.html)). 문서에 따르면 IAM 사용자도 `aws login`을 쓸 수 있고, 세션은 최대 12시간이며 그 뒤 다시 로그인해야 한다.

1. **IAM 사용자 생성:** 사용자(루트 아님)가 콘솔에서 `trid-controller`를 만든다.
   - 콘솔 비밀번호와 MFA를 설정한다.
   - **액세스 키는 만들지 않는다.**
2. **정책 연결:**
   - AWS 관리형 `SignInLocalDevelopmentAccess`: `aws login`에 필요하다.
   - 인라인 감시 정책 `deploy/macmini-controller/iam-policy-trid-controller.json`: S1 Describe·List, `corpus/runs/s1/*` 읽기, status 한 객체 쓰기. 삭제, 생성, PassRole 없음.
3. **발행 권한 (별도 승인 후에만):**
   - `iam-policy-trid-controller-publish-FUTURE.json`의 `<campaign>`을 승인된 prefix로 바꿔 붙인다.
   - 내용: `trid-pt-*` 생성·중지, `trid-sagemaker-lora`만 PassRole, 그 캠페인의 checkpoint만 삭제.
   - 승인 전에는 붙이지 않는다.
4. **맥미니에서 로그인 전환:**
   - 실행: `/opt/homebrew/bin/aws login --profile trid-controller --region ap-northeast-2`
   - 브라우저에서 root가 아니라 **IAM 사용자 `trid-controller`**로 로그인한다. 기존 root 세션 덮어쓰기를 묻는 질문에는 y로 답한다.
5. **확인:** `bash ~/trid-controller/repo/deploy/macmini-controller/preflight.sh`
   - identity가 `arn:aws:iam::148761639313:user/trid-controller`이고 root 경고가 사라져야 한다.
   - 6개 항목이 모두 PASS여야 한다.
   - 다음 주기에 driver `state.json`의 `identity.kind`가 `iam`이 된다.
6. **세션 만료:** 12시간마다 사람이 다시 로그인해야 한다. 그 사이 제어기는 `reauth_required` 상태로 대기하며 job을 만들지 않는다.
   - 영구 무인 운영은 아니다.
   - 그 이상이 필요하면 장기 키와 IAM Roles Anywhere 같은 다른 경로가 필요하고, 이는 별도 보안 결정이다.

## 7. 다음
- C017 manifest를 받으면 다음 순서로 진행한다:
  1. 해시와 형식을 확인한다.
  2. 토큰 길이 분포와 행 수를 확인하고, 그에 따른 단계 수·예상 시간·비용을 이 문서에 기록한다(기동 전).
  3. job을 1회 기동한다.
  4. 원본과 튜닝본을 같은 조건으로 평가한다.
  5. 관문 판정을 보고한다.

SUPERVISION_REPORT_READY PILOT-SFT-001-A
