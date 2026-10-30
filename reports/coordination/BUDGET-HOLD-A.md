# BUDGET-HOLD-A: A 소유 AWS 자원 정지 현황

- 세션 01a18739-bcc6-77fc-b2a0-14d216f1becb / A
- 기준 시각: 2026-10-30 11:58 KST
- 수신: BUDGET-HOLD-001, BUDGET-300K-001

## 1. 자동 기동 장치 정지 (11:54 KST)

로컬 백그라운드 프로세스 3개를 종료했다. `pgrep` 결과 "no local drivers"를 확인했다.

| 프로세스 | 역할 |
|---|---|
| `scripts/drive_sft_campaign.py` (smoke campaign) | 세그먼트 연결·마일스톤 평가 자동 기동 |
| `scripts/drive_sft_campaign.py` (spot smoke campaign) | 같음 (spot) |
| 강제 중단 시험 루프 | spot 세그먼트 2 중단 시험 |

- 이후 A 세션에서 새 SageMaker 작업을 만들지 않았다.
- A-006 120만 단계 본 캠페인은 한 번도 기동하지 않았다. 스모크만 실행했다.

## 2. 정지 요청한 job (11:54:17~18 KST)

| job | 인스턴스 | 상태 | 손실 / 재개 위치 |
|---|---|---|---|
| `trid-pt-a006-spot-smoke-91m-1030114317` | ml.g6e.xlarge spot | Starting(spot 용량 대기) → Stopped | 학습 시작 전이라 손실 없음. S3 `sft-campaign-spot-smoke/…/checkpoints/step-00000200,300` 보존 |
| `trid-pt-a006-bench-sft-speed-1030113422` | ml.g6e.2xlarge spot | 약 20분 Starting(spot 용량 대기) → Stopped | 측정 미수행. 저장할 산출물 없음 |
| `trid-pt-a006-smoke-91m-ref-eval-1030114917` | ml.g6e.16xlarge | Training(평가 중) → Stopped | 스모크 기준 run의 step-600 평가가 없어졌다. 가중치는 S3에 남아 있다(`…/milestones/step-00000600`) |

## 3. 완료까지 두는 job 2개 (새 학습 단계가 아님)

| job | 인스턴스 | 11:54 진행 | 그대로 둔 이유 | 예상 추가 과금 |
|---|---|---|---|---|
| `trid-pt-a004-sft3600-390m-mha-1030110936` (A-004 E1) | ml.g6e.2xlarge, $3.44565/h | SFT 2,900/3,600단계 | 이 job은 중간 체크포인트가 없는 이전 코드다. 지금 멈추면 진행분(약 2,800초)을 저장할 수 없다. 남은 약 10분 SFT와 평가 뒤 S3에 저장되고 종료된다 | 약 30~40분, $1.7~2.3 |
| `trid-pt-a004-span-390m-mha-1030111012` (A-004 E2) | ml.g6e.8xlarge, $6.9595/h | span `sum` 완료, `mean` 진행 중 | 평가 전용이다. `mean` 결과가 끝날 때 저장된다 | 약 10~20분, $1.2~2.3 |

- 두 job 모두 끝나면 인스턴스가 자동으로 종료된다. 이후 A 소유 실행 중 연산은 0이다.
- 종료 시각과 실제 과금 시간은 A-004.md에 기록한다.

## 4. A가 이미 쓴 SageMaker 비용 (Billable 초 × 서울 리전 공식 시간당 단가, VAT·환율 제외)

| 작업 | USD |
|---|---|
| A-002 파일럿 | 4.26 |
| A-003 1B 비교 | 30.13 |
| A-004 (11:58까지 종료분) | 2.93 |
| A-006 스모크·평가·spot | 5.75 |
| **합계** | **43.08** |

- 위 3절의 진행 중 job 2개 과금은 아직 들어가지 않았다(예상 추가 $3~5).
- 공식 단가 출처: AWS Pricing API (`AmazonSageMaker`, ap-northeast-2, Training)

| 인스턴스 | 시간당 단가 |
|---|---|
| ml.g6e.xlarge | $3.2032 |
| ml.g6e.2xlarge | $3.44565 |
| ml.g6e.4xlarge | $4.6169 |
| ml.g6e.8xlarge | $6.9595 |
| ml.g6e.16xlarge | $11.6447 |

## 5. 남은 과금 자원: S3 저장소 (A 소유 prefix, 버킷 `trid-training-148761639313`)

| prefix | 크기 | 내용 |
|---|---|---|
| `corpus/runs/pretrain/` | 171.4 GB | A-003 390M 104.6, A-003 91M 24.5, A-002 파일럿 등 42.4 |
| `corpus/runs/sft-eval/` | 16.5 GB | A-003·A-004 SFT 가중치와 평가 |
| `corpus/runs/sft-campaign-smoke/` | 7.3 GB | A-006 재개 스모크 |
| `corpus/runs/sft-campaign-spot-smoke/` | 3.3 GB | spot 스모크 |
| `corpus/tokens/mix-v3-basetok-tokens/` | 1.0 GB | 사전학습 토큰 |
| `corpus/eval-inputs/a003/` | 0.06 GB | — |
| `corpus/code/pretrain/` | 0.003 GB | — |
| **합계** | **약 199.5 GB** | — |

- **월 비용:** 공식 단가 $0.025/GB-월(S3 Standard, 첫 50TB) 기준 약 $5.0/월.
- 버킷 전체는 387 GB이고, 여기에는 다른 세션의 객체가 포함된다.
- **아무것도 삭제하지 않았다.**
- 줄일 수 있는 후보(삭제는 승인 후에만):
  - A-003 사전학습 중간 recovery 체크포인트: SageMaker 동기화가 로컬 삭제를 반영하지 않아 누적됐다. 390M은 약 16개 × 4.7GB.
  - 최종 step 15,259와 best 가중치를 빼고 지우면 약 100 GB가 줄어 월 약 $2.5 절감.

## 6. 기타 자원
- **EC2:** A 소유 인스턴스 없음.
- **할당량 증가 요청 2건:** EC2 G 온디맨드 8, 스팟 16. 대기 중이며 비용은 없다.
- **endpoint·자동 스케줄러:** 없음.

## 7. 재개 위치 (재기동은 승인 후)
- **A-006 120만 단계:** 기동한 적 없다. 재개 경로는 스모크로 확인했다(A-006-spot.md에 정리 예정).
- **A-004:** 위 두 job이 끝나면 결과가 확정된다.

SUPERVISION_REPORT_READY BUDGET-HOLD-A
