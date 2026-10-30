# BUDGET-PLAN-B: B 유료 작업 계획과 예약 요청

작성: B, 2026-10-30 KST. 근거: `reports/coordination/BUDGET-300K-001.md` (B 내부 예약 한도 1만원).

## 1. 현재 상태: hold

- B 소유의 실행 중 유료 컴퓨트는 **0개**다(BUDGET-HOLD-B, 11:53 KST 확인).
- 새 유료 작업도 0개다. 재기동하지 않았고, 자동 후속 작업도 없다.
- 남아 있는 B 작업은 모두 로컬이다: 공개 업무자료 조사, 이미 확보한 원문 정리, 정책브리핑과 PRISM 품질 판정.

## 2. 이번 예약 요청: 0원

B는 지금 유료 작업이 필요하지 않다. 1만원 한도에서 **예약을 요청하지 않는다**.

남는 한도를 다른 세션에 넘기는 것도 B가 정할 일이 아니다. 배정은 총괄이 한다.

**나중에 유료 정제가 필요해지면, 시작 전에 아래 형식으로 따로 보고하고 승인을 받는다.**

예시는 이전 FineWeb 정제와 같은 형태다:

| 항목 | 값 |
|---|---|
| 작업 / 목적 | CPU 정제 1회 (예: 원문 재추출, 대용량 PII 재처리) |
| 인스턴스 / 수량 | `ml.c6i.4xlarge` × 1, 온디맨드. Spot·자동 fallback 없음 |
| 공식 단가 | SageMaker Training, 서울 리전 $0.922/시간 |
| 최대 실행 시간 | `MaxRuntimeInSeconds` 7,200 (2시간). 기동·종료 과금 시간 포함 |
| 재시도 / 후속 | 재시도 0회, 자동 후속 작업 없음 |
| 상한 | $1.844 × 1,500원(보수적 환율 가정) × 1.1(VAT) = **약 3,043원** |
| 누적 예약 | 이 상한을 한 번 쓰면 B 한도 1만원 중 3,043원 |

**단가와 가정:**
- 단가는 AWS Pricing API에서 가져왔다(`AmazonSageMaker`, `APN2-Train:ml.c6i.4xlarge`, effectiveDate 2026-10-01). 같은 조회에서 `ml.c6i.2xlarge`는 $0.461/시간이었다.
- 환율 1,500원/USD는 상한 산정용 가정이다. 실제 환율은 확인하지 않았다.
- ML 스토리지(EBS)와 S3 요청 요금은 위 상한에 넣지 않았다. 이전 작업 규모에서는 1원 단위 이하다 [추정].

## 3. 이미 발생한 B 비용 (이번 30만원과 별도로 고지)

SageMaker 학습 작업 형태의 CPU 작업, `BillableTimeInSeconds` 기준:

| 작업 | 인스턴스 | 과금 초 | 비용 (공식 단가) |
|---|---|---|---|
| `fineweb2ko-pilot-1030024855` | ml.c6i.2xlarge | 951 | $0.122 |
| `fineweb2ko-cleanv1-pilot-1030031219` | ml.c6i.4xlarge | 2,728 | $0.699 |
| `fineweb2ko-cleanv2-pilot-1030042823` | ml.c6i.4xlarge | 2,672 | $0.684 |
| `fineweb2ko-cleanv3-pilot-1030052414` | ml.c6i.4xlarge | 2,867 | $0.734 |
| `fineweb2ko-cleanv4-pilot-1030061743` | ml.c6i.4xlarge | 2,998 | $0.768 |
| **합계** | | 12,216 s (3.39 h) | **$3.01 ≈ 약 4,970원** (1,500원 × VAT 1.1 가정) |

- 모든 작업은 Spot 아님, `MaxRuntimeInSeconds` 14,400, 재시도 없음, `Completed` 상태다.
- 중단된 뒤 남은 연산은 없다.

**S3 저장 (지속 비용):**
- B 접두어 약 30.6 GB, 약 $0.8/월 ≈ 1,300원/월 [추정: 단가 $0.025/GB-월]. 세부는 BUDGET-HOLD-B 2절에 있다.
- 줄일 수 있는 항목: FineWeb clean-v1~v3 중간본(쓰지 않음)이다. 삭제는 승인 없이 하지 않는다.

SUPERVISION_REPORT_READY BUDGET-PLAN-B
