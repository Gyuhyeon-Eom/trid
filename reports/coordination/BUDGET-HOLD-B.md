# BUDGET-HOLD-B: B 소유 유료 자원 상태

작성: B, 2026-10-30 11:57 KST. 근거: 사용자 비용 제약 BUDGET-HOLD-001.

## 1. 확인 결과: 실행 중인 B 소유 유료 컴퓨트 없음

확인 시각: 2026-10-30T02:53:50Z (11:53 KST), 리전 ap-northeast-2.

| 자원 | 상태 | 소유 |
|---|---|---|
| SageMaker processing job | 진행 중 0 | 해당 없음 |
| SageMaker training job (진행 중 6) | `trid-pt-a006-*` 3, `trid-pt-a004-*` 2, `trid-c010-taskgen-*` 1 | **A, C 소유. B는 손대지 않았다** |
| SageMaker endpoint, notebook | 0 | 해당 없음 |
| EC2 `i-01e894092261d7f9a`, `i-0a4b24516d78b628d`, `i-0bdbf9d10f096dcf2` | 모두 stopped | C 소유 (이름 기준). B는 손대지 않았다 |

**B가 띄운 CPU 작업**
- `fineweb2ko-*` 학습 작업 형태의 CPU 작업 5건은 모두 `Completed` 상태다. 마지막 작업은 `fineweb2ko-cleanv4-pilot-1030061743`이며 07:08 KST에 끝났다.
- 산출물은 S3 `corpus/clean/fineweb2-ko/clean-v1~v4/`에 이미 저장되어 있고 manifest로 검증했다(B-006·B-007).
- 그 이후 B는 유료 작업을 새로 시작하지 않았다.

**자동 후속 작업**
- B 소유의 자동 드라이버나 예약 작업은 없다.
- 로컬에 `scripts/share_demo_control.py supervise --hours 24.0`(pid 63877, 10-29 14:34 시작)와 이를 붙잡는 `caffeinate`가 돌고 있다. B가 시작한 프로세스가 아니며, 이 스크립트에는 AWS 호출(ec2·sagemaker·boto3)이 없다. 손대지 않았다.

**결론:** B 쪽에서 정지할 대상이 없었다. 그래서 일시정지나 종료 조치는 하지 않았다. 손실된 작업도, 재개해야 할 위치도 없다.

## 2. 남은 과금 자원: S3 저장소

버킷 `trid-training-148761639313` 안의 B 접두어 (2026-10-30 11:55 KST 측정):

| 접두어 | 객체 | 크기 |
|---|---|---|
| `corpus/clean/fineweb2-ko` (clean-v1~v4) | 111 | 21.67 GB |
| `corpus/raw/fineweb2-ko` | 1 | 4.84 GB |
| `corpus/clean/korea-kr-briefing` (clean-v1~v4) | 49 | 2.27 GB |
| `corpus/raw/prism` | 286 | 0.88 GB |
| `corpus/clean/assembly-minutes` (clean-v1~v5) | 122 | 0.34 GB |
| `corpus/reserved` (ALIO 예약 148계열) | 1,901 | 0.31 GB |
| `corpus/raw/assembly-minutes` | 2,140 | 0.20 GB |
| `corpus/state`, `corpus/code`, `corpus/clean/prism`, `corpus/raw/public-data-standards` | 766 | 0.06 GB |
| **합계** | | **약 30.6 GB** |

**월 비용 추정** [추정]: S3 Standard 서울 리전 단가 약 $0.025/GB-월 기준으로 **약 $0.8/월**이다. 단가는 이번에 확인하지 않았다.

**줄일 수 있는 부분:** 가장 큰 항목은 FineWeb 정제 중간본이다(clean-v1~v3). 이 버전들은 사용하지 않으므로 지우면 저장량이 줄어든다. 다만 삭제는 하지 않았다. 버전 불변 원칙과 "삭제 금지" 지시를 따랐다. 지울지는 사용자·총괄이 결정한다.

## 3. 보류 중에 하는 일과 하지 않는 일

**하는 일: 로컬 분석과 이미 확보한 자료 정리**
- 정책브리핑 clean-v4: 로컬 입력으로 만들었다. 결과는 S3에 저장했고 15/15 샤드를 검증했다.
  - 이 저장 단계는 보류 지시 전에 시작했고, 컴퓨트 비용이 없는 로컬 작업이었다. 완료까지 진행했다.
- 공공데이터 공통표준 CSV 3개(3.2 MB)도 보류 지시를 받기 전에 S3 원문으로 저장했다.

**하지 않는 일**
- SageMaker, EC2, Spot 작업은 새로 만들지 않는다.
- 새 예산을 사용자가 확정하기 전에는 AWS 컴퓨트를 다시 띄우지 않는다.
- 이후 S3 쓰기는 꼭 필요한 저장에만 한정한다.

SUPERVISION_REPORT_READY BUDGET-HOLD-B
