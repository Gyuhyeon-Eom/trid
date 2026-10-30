# C-009 중간 보고(capacity): 두 모델 GPU 상주 자원으로 실제 전환

작성: 2026-10-30 09:30 KST. 새 SFT 없음. A/B 자원 변경 없음. 모델은 그대로다(생성 Qwen3-32B Q8_0, 검증 gpt-oss-120b MXFP4, 같은 sha256). 바뀐 것은 실행 자원뿐이다.

## 1. 할당량·가용성 조회 결과 (2026-10-29 23:3x UTC 실측)

| 경로 | 결과 |
|---|---|
| EC2 On-Demand G/VT, ap-northeast-2 | 8 vCPU. 48 상향 요청은 `CASE_OPENED`(case 179331452300736, AWS 수동 검토) |
| EC2 On-Demand G/VT, 다른 11개 리전 | 모두 0(us-east-1/2, us-west-2, ap-northeast-1/3, ap-southeast-1/2, ap-south-1, eu-west-1, eu-central-1, eu-north-1, ca-central-1) |
| EC2 On-Demand P | 서울 포함 12개 리전 모두 0 |
| EC2 Spot G | 서울 8 vCPU. 나머지 조회 리전은 0 |
| 인스턴스 제공 | g6e.12xlarge·p4d·p5는 서울 등 대부분 리전에 제공되나, 위 할당량 때문에 기동할 수 없다 |
| **SageMaker 작업 할당량, 서울** | **`ml.g6e.12xlarge for training job usage` = 1**(spot도 1). `ml.g5.12xlarge` endpoint 1 |
| Bedrock(도쿄·미국) | `qwen.qwen3-32b-v1:0`, `openai.gpt-oss-120b-1:0` on-demand 제공. 리비전·양자화를 고정할 수 없어 파일럿과 같은 모델임을 보증하지 못하므로 채택하지 않음 |

## 2. 채택한 자원: SageMaker 작업 `ml.g6e.12xlarge`(서울)

- **사양:** 4×L40S(46 GB씩, 드라이버 595.91.07), 48 vCPU.
- **현재 작업:** `trid-c009-taskgen-202610292345`.
- **성격:** SageMaker의 "training job" API를 컨테이너 실행 수단으로만 쓴다. 가중치 갱신이나 학습은 없고, 추론 서버 두 개와 C-009 단계 실행만 한다.
- **실행 역할:** 새로 만든 `trid-sagemaker-c009-taskgen`. 권한은 `c009/*`, `teacher-models/c009/*`의 S3 읽기·쓰기와 로그·ECR pull뿐이다. 다른 세션의 `trid-sagemaker-lora`는 쓰지 않았다.
- **컨테이너:** AWS DLC `pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-sagemaker`.
- **llama.cpp:** 같은 커밋 `8ca4060b7e30…`의 소스(GitHub 아카이브 sha256 `99899baa…19c6`, S3 `c009/sm/`)를 작업 안에서 CUDA 12.9, sm_89로 빌드했다(약 3분). 빌드 결과는 S3에 캐시한다.
  - EC2 빌드는 CUDA 13 런타임 링크였다. 커밋은 같지만 CUDA 판이 달라 출력이 완전히 같지는 않다(7절 참조).
- **GPU 배치:** GPU 0–1에 Qwen3-32B(25.6 + 24.8 GB, `-c 65536 -np 16`), GPU 2–3에 gpt-oss-120b(33.3 + 30.5 GB, `-c 65536 -np 16`). 두 모델 모두 GPU에 상주하며 CPU 오프로드는 없다. slot당 문맥은 4,096이다.
- **러너:** `sm_runner.sh`(sha `74c4968a…`). S3 `c009/sm/queue/`에서 단계 명령을 받아 draft(Qwen)와 verify(gpt-oss)를 각각 하나씩 동시에 처리한다. 작업 디렉터리는 60초마다 S3에 동기화한다.
- **종료 조건:**
  - 대기열이 20분 비면 스스로 종료한다.
  - `queue/stop` 표지가 있으면 종료한다.
  - 최대 24시간.
- **로그:** `c009/sm/logs/<job>/`(runner, gen, verify, gpu, build).

## 3. 실측 성능 비교 (같은 모델 파일, 같은 pilot2 검증 입력 86건)

| 항목 | EC2 g6e.2xlarge(CPU 오프로드 14층) | SageMaker ml.g6e.12xlarge(GPU 상주) |
|---|---|---|
| gpt-oss prefill | 18.7–28.9 tok/s | 1,549–2,310 tok/s |
| gpt-oss 생성 | slot당 4.55–4.60 tok/s × 8 slot | slot당 17.7–18.3 tok/s × 16 slot |
| pilot2 검증 처리 | 약 60분에 34건(중단) | 86건 약 10분 |
| Qwen3-32B 생성(초안) | slot당 13.4–14.1 tok/s × 6 | 같은 수준 × 16 slot(본 생산에서 측정 예정) |

- **EC2 파일럿 검증 34건:** 유효한 측정으로 보존했다(`c009/work/pilot2/verified.jsonl`, `verify.stopped-early`).
  - 같은 34건에 대한 SageMaker 판정과 상태 일치는 29/34다.
  - 차이 5건은 빌드(CUDA 13 대 12.9), 슬롯 수, 오프로드 차이에서 온 비결정성으로 본다. 이 판단은 추론이며 원인별로 분리하지는 않았다.
- **EC2 `i-0bdbf9d10f096dcf2`:** 산출물과 로그를 S3에 동기화하고 GPU 프로세스·로그인이 없음을 확인한 뒤 stop했다. 본 생산에는 이 설정을 쓰지 않는다.

## 4. 품질 측정(개발용) — 관문 아님

- pilot2(C-008 실패 문서 60 + 새 문서 60)를 SageMaker에서 검증했다.
  - 채택 45, 의미 탈락 40, 격리 1.
  - 172회 호출 중 1회가 길이 한도에 걸렸고, 해당 항목은 격리했다.
- 채택 45건을 AI 검토한 결과 정답은 27/45(60%)다. 관문(유형별 18/20)에 크게 못 미친다.
  - 주된 원인: 유형 부적합(조건·예외가 아닌 질문), 설명의 근거가 결론과 동일, 수치 편집이 제목·요약줄의 원래 값과 모순, 다중근거가 두 사례를 나열.
  - 검증기는 이 결함을 묻지 않았다(유형 적합성 항목 없음).
- 이 표본과 C-008 120 검토 문항은 모델·프롬프트 개발용이다. 확대 관문에는 새로 고정한 미사용 120문항을 쓴다.

## 5. 다음

1. 생성 프롬프트를 고친다: 유형 정의와 부적합 예시, JSON 안에 자기 점검 필드 선행, 수치 편집 시 제목·요약줄 확인.
2. 검증 판정을 고친다: 유형 적합성(`type_ok`), 수치는 원문과 편집문 대조, 설명은 결론과 다른 근거 문장.
3. 같은 개발 문서로 재측정한 뒤 고정한다.
4. 새 미사용 문서 120건(유형별 20)으로 관문을 확인하고, 통과한 경로로만 확대한다.

SUPERVISION_REPORT_READY C-009-capacity
