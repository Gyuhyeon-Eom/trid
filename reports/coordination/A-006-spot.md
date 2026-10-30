# A-006-spot: Managed Spot 실측과 재개 경로

- **세션:** 01a18739 / A
- **상태:** 측정 완료. 신규 spot job 없음.
- 이 보고는 기존 실측만 정리한다. 추가 비용은 0원이다.

## 1. 실측 (SageMaker DescribeTrainingJob)

| job | 인스턴스 | 결과 | TrainingTime | Billable | 절감 |
|---|---|---|---|---|---|
| `trid-pt-a006-spot-smoke-91m-1030113612` (구간 1) | ml.g6e.xlarge spot | Completed | 305초 | 195초 | **36%** |
| `trid-pt-a006-spot-smoke-91m-1030114317` (구간 2, step 300에서 재개 예정) | ml.g6e.xlarge spot | Stopped | — | — | 용량 미확보 |
| `trid-pt-a006-bench-sft-speed-1030113422` (속도 측정) | ml.g6e.2xlarge spot | Stopped | — | — | 용량 미확보 |

- **용량 미확보 2건의 기록:** "Insufficient capacity error from EC2 while launching instances, retrying!" 상태가 이어졌다. A가 정지했으며 과금은 없다(Billable 없음).
- **36%의 의미:** 짧은 job 1건에서 나온 할인율이다. 긴 학습에서 어느 정도 절감될지는 확인하지 않았다.

## 2. 재개 경로 (검증됨)
- **SFT 재개:**
  - recovery checkpoint에 모델, optimizer, RNG, 데이터 위치를 담는다(`CHECKPOINT_EVERY`, 최근 2개 유지).
  - SageMaker checkpoint 동기화(`/opt/ml/checkpoints` ↔ S3)로 다음 job이 이어받는다.
- **정확 재개 검증:**
  - CPU 테스트 `tests/test_pretraining.py::test_dialogue_training_split_into_sessions_matches_one_run`
  - SageMaker에서 300+300단계로 나눠 돌린 run과 600단계를 한 번에 돌린 run의 검증 손실 차이 ≤5e-4 (A-006 재개 스모크)
- **fused AdamW 재개 결함:** CUDA에서 fused AdamW가 재개되지 않던 결함을 고쳤다(`67717e1`).
- **spot 중단 대응:** spot 중단이 나면 같은 checkpoint S3 URI로 다음 구간을 띄운다(`scripts/drive_sft_campaign.py`).
  - 연속 2회 실패하면 그 arm을 정지한다.
  - 비용 상한과 기동 횟수 상한이 있다.

## 3. 운영 판단
- **spot은 기본 경로가 될 수 없다.** g6e 타입별 spot 할당량은 1이고, 측정한 3건 중 2건이 용량을 받지 못했다.
- **정책 유지:** MaxWait는 1시간이다. 온디맨드로 자동 대체하지 않는다.
- **현재 상태:**
  - S1은 온디맨드로 승인돼 실행 중이다.
  - 제품 주경로가 공개 가중치 기반 특화로 바뀌었다(TRID-BASE-CUTOVER-001). 따라서 긴 scratch 학습용 spot 캠페인은 보류한다.
  - 특화 학습은 로컬 실행을 먼저 시도한다. GPU가 필요하면 사전 검산과 예약을 받은 뒤 짧은 job으로 쓴다. spot은 용량 대기 비용이 없는 실험에만 고려한다.

SUPERVISION_REPORT_READY A-006-spot
