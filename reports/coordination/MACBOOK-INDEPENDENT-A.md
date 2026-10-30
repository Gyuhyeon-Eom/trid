# MACBOOK-INDEPENDENT-A: 원격 GPU 작업의 맥북 독립성 검증

- 세션 01a18739 / A
- **결론:** 승인된 단일 단계 작업(S1)은 job 하나 안에서 다음을 모두 끝낸다. 맥북이 꺼지거나 연결이 끊겨도 진행되고, 산출물은 S3에 남는다. 아래 스모크로 실제 확인했다.
  - 학습 → recovery 체크포인트 → 마일스톤 → 고정 평가 → DONE 표식·저장 → 종료
- 여러 job을 이어 붙이는 장기 실행에는 별도 제어기가 필요하다. 맥미니로 이관을 진행 중이다(MACMINI-DEPLOY-001).

## 1. 맥북에 의존하던 단계 (변경 전)

| 단계 | 기존 방식 | 맥북 의존 |
|---|---|---|
| job 생성 | `scripts/launch_sm_pretrain.py` (로컬 AWS CLI) | 기동할 때만 |
| 세그먼트 연결·마일스톤 평가 job 생성·S3 recovery 정리 | `scripts/drive_sft_campaign.py` (로컬 상시 루프) | **상시 의존** |
| 평가 | 별도 `sm_eval_weights.sh` job (driver가 생성) | 상시 의존 |
| SSM 터널·로컬 업로드 | 사용하지 않음 | 없음 |

## 2. 변경 (커밋 `9b16936`, `e0779dd`, `eba1d40`)

**`scripts/sm_sft_campaign.sh`**
- `EVAL_STEPS`와 `EVAL_BEST=1`을 주면, 학습이 끝난 뒤 같은 인스턴스에서 고정 평가를 실행한다(RAG310 evidence/free, KLUE dev 질문 민감도).
- 결과는 `<run>/eval/<label>/`에 둔다.
- 마지막에 `DONE.json`을 쓴다. 내용은 status, completed_steps, best_step, 평가 라벨 목록이다.
- 그 뒤 job이 끝나고 인스턴스가 종료된다.
- 학습 상한 `SEGMENT_SECONDS`를 넘으면 그 시점에 recovery 체크포인트를 남기고 평가로 넘어간다.

**`scripts/sm_observe_s3.sh`**
- CPU job 안에서 다른 job의 S3 출력을 20초 간격으로 기록하는 독립 관측기다.
- 읽기 전용이다.

**`scripts/drive_sft_campaign.py`** (장기 연결용, 맥미니 이관 대상)
- 재시작해도 같은 run의 실행 중 job을 찾아 넘겨받는다. 중복으로 기동하지 않는다.
- 누적 비용 상한, 기동 횟수 상한, 감시만 하는 모드를 넣었다.

## 3. 스모크 (실제 실행)

**구성**
- job `trid-pt-mi-smoke-91m-1030121501` (ml.g6e.xlarge): 91M, 200단계, 마일스톤 200, job 안 평가 2회(step-200, best-valid)
- 관측기 `trid-pt-mi-smoke-observer-1030121522` (ml.m5.large, CPU): 맥북과 별개로 클라우드 안에서 S3를 관측

**맥북 쪽 상태**
- 두 job을 기동한 직후(12:15:23) 확인했을 때 trid 제어 프로세스(driver, launcher)는 없었다.
- 실행 중이던 것은 사용자 환경의 AWS MCP 프록시뿐이었다. trid와 무관하다.
- 이후 맥북에서는 `describe-training-job` 읽기 조회만 했다. 상태 확인용이며 job을 조작하지 않았다.

**관측기 로그** (`s3://…/corpus/runs/mi-smoke-observer/mi-smoke-observer/observer-log.json`, 결과 `done_marker_seen`, UTC)

| 시각 | 관측된 객체 |
|---|---|
| 03:19:36 | `sft/training.json` |
| 03:20:16 | `sft/milestones/step-00000200/model.safetensors`, `sft/checkpoints/step-00000200/state.json` |
| 03:20:36 | `sft/checkpoints/step-00000200/optimizer.pt` (730 MB) |
| 03:20:57 – 03:21:17 | `eval/step-00000200/rag310-evidence.json`, `rag310-free.json`, `question-sensitivity.json` |
| 03:21:57 – 03:22:17 | `eval/best-valid/…` 3개 |
| 03:22:17 | `DONE.json` = `{"status":"complete","completed_steps":200,"best_step":200,"evaluated":["best-valid","step-00000200"]}` |

**job 종료**
- 학습 job: Completed 12:22:25 KST, Billable 400초
- 관측기: Completed 12:22:33 KST, Billable 390초
- 두 job 모두 끝나면서 인스턴스가 회수되었다.
- 비용: 587원 + 25원 (BUDGET-RUN-A에 기록)

## 4. 한계와 남은 일

- **이번 검증이 보이는 것:** "job 내부 완결성"이다. 맥북을 실제로 끈 상태에서 돌려 본 것은 아니다. 대신 기동 후 맥북 제어 프로세스가 0개였고, 클라우드 안의 관측기가 산출물 생성을 기록했다. 장기 실행에서 맥북이 실제로 꺼졌을 때의 확인은 맥미니 제어기 이관 뒤에 다시 한다.
- **job 하나의 시간 한계:** SageMaker MaxRuntime은 최대 28일이다. 다만 비용 정책상 승인 단계별로만 기동한다.
- **여러 job 연결:** 120만 단계처럼 job을 여러 개 이어야 하면 제어기가 필요하다. 이를 위한 별도 AWS 상시 서버는 만들지 않는다. 맥미니에 같은 driver를 배포한다(MACMINI-DEPLOY-001). `nohup`으로 맥북에서 돌리는 것은 맥북 독립이 아니다.
- **현재 S1 두 job:** 위와 같은 방식(job 안 평가, `DONE.json`)으로 기동했다. driver 없이 끝난다.

SUPERVISION_REPORT_READY MACBOOK-INDEPENDENT-A
