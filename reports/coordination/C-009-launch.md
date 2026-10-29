# C-009 시작 보고: 32B급 생성기·다른 계열 독립 검증기 교체 기동

작성: 2026-10-30 08:45 KST. 새 SFT 없음, A/B 자원·A 비교 조건 변경 없음. 외부 모델 가중치는 학습 자료 작성 도구이며 TRID 제품 가중치에 섞지 않는다.

## 1. 모델 선택과 사용 조건 근거

| 역할 | 모델 | 원본 리비전 | 양자화 파일(sha256) | 라이선스 |
|---|---|---|---|---|
| 생성 | Qwen3-32B (dense 32.8B) | `Qwen/Qwen3-32B@9216db57…`, GGUF `Qwen/Qwen3-32B-GGUF@938a7432…` | `Qwen3-32B-Q8_0.gguf` 34,817,718,912 B, `2c50eb8a…2169` | Apache-2.0 |
| 독립 검증 | gpt-oss-120b (MoE 117B, 활성 5.1B) | `openai/gpt-oss-120b@b5c939de…`, GGUF `ggml-org/gpt-oss-120b-GGUF@7200b74b…` | `gpt-oss-120b-MXFP4.gguf` 63,387,346,208 B, `582bd40f…622d` | Apache-2.0 + USAGE_POLICY |

- 서로 다른 계열(Alibaba Qwen / OpenAI gpt-oss)로 생성과 검증을 분리했다.
- 두 저장소 모두 공개 가중치이며, 토큰이나 동의 절차 없이 내려받을 수 있음을 확인했다.
- 사용 조건:
  - Apache-2.0은 출력물을 합성 학습 데이터로 쓰는 것을 제한하지 않는다.
  - gpt-oss USAGE_POLICY의 요구는 "applicable law 준수"와 안전한 사용이며, 합성 데이터 생성을 금지하는 조항은 없다.
  - 원문 LICENSE·USAGE_POLICY는 `C-009/licenses/`에 보존했다.
- 제외한 후보:
  - EXAONE 계열: 비상업 라이선스이며 출력물을 다른 모델 개발에 쓰는 데 제약이 있다.
  - Llama 3.x: 출력물로 학습한 모델의 이름 규정이 TRID 명명과 충돌한다.
  - aya-expanse-32b: CC-BY-NC.
- 기존 Kanana 2.1B·Qwen3 1.7B 경로로는 추가 생성을 하지 않는다.

## 2. GPU 작업

- **작업 이름:** `trid-c-taskgen-32b-01`(`i-0bdbf9d10f096dcf2`).
- **사양:** g6e.2xlarge, L40S 46 GB, RAM 61 GB, 8 vCPU, ap-northeast-2a, AMI `ami-0024061cbf153effc`.
- **기동:** 2026-10-29 22:55:33 UTC.
- **정지 설정:** shutdown behavior `stop`.
- **태그:** Owner `trid-session-C-01a17c28`, Task, StopCondition.
- **사양 결정 근거:**
  - 계정의 On-Demand G 인스턴스 할당량이 8 vCPU이고, P 인스턴스 할당량은 0이다. 따라서 g6e.12xlarge(4×L40S, 48 vCPU)는 지금 띄울 수 없다.
  - 할당량 상향(→48)을 요청했다. 상태는 `CASE_OPENED`(case 179331452300736)로 AWS 수동 검토 중이다.
  - 그때까지는 단일 GPU에서 생성 단계와 검증 단계를 번갈아 서빙한다.
- **타이머 점검:** 기동 직후 systemd 타이머는 OS 기본 항목뿐이었다. kms 타이머는 없고, cron·systemd에 shutdown/poweroff 항목도 없다. 다른 프로젝트의 110분 타이머는 이 인스턴스에 없으며 건드리지 않았다.
- **서빙:** `serve32.sh`(sha `7880b91a…`)의 `gen` / `verify` 단계, llama.cpp `8ca4060`의 CUDA 빌드.
  - 생성: `-c 24576 -np 6`(slot당 4,096).
  - 검증: `-c 32768 -np 8 --n-cpu-moe 14`. 46 GB에 다 올라가지 않아 14개 층의 전문가 가중치를 CPU에 둔다.

## 3. 실제 추론 증거

- 생성기 Qwen3-32B: 파일럿 1차·2차 각 120후보 초안, 총 293회 호출. 모두 `finish_reason=stop`이고 잘림은 0이다.
  - 최대 입력 2,174토큰, 최대 출력 585토큰.
  - 속도: prefill 약 2,000 tok/s, slot당 생성 13.4–14.1 tok/s, 6 slot 합계 약 80 tok/s. 최대 VRAM 39.3 GB.
- 검증기 gpt-oss-120b:
  - 상태: health ok, VRAM 39.7 GB.
  - 첫 호출 측정: prefill 18.7–28.9 tok/s, slot당 생성 4.55–4.60 tok/s(CPU 전문가 오프로드 때문).
  - 파일럿 검증은 진행 중이다.
- **병목:** 이 속도로는 파일럿 검증(86건)에 약 1시간이 걸리고, 본 생산 규모의 검증은 비현실적이다.
  - 할당량이 승인되면 g6e.12xlarge(192 GB)에서 두 모델을 GPU에 모두 올린다.
  - 승인이 늦어지면 단일 GPU에 들어가는 다른 계열의 32B급 검증기로 바꾸는 방안을 검토한다. 바꾸는 경우 같은 관문 표본을 새로 검토한다.

## 4. 내구 저장

- 모델: `s3://trid-training-148761639313/teacher-models/c009/`(두 파일, 다운로드 직후 sha256 검증).
- 코드·작업: `c009/code/`, `c009/work/{pilot,pilot2,…}/`(60초 체크포인트, S3가 원본), `c009/stage32.sh`, `c009/serve32.sh`.
- 서빙 로그: `c009/serve/<시각>-gen|verify/`(2분 간격).
- NVMe에만 있는 산출물은 없다.

## 5. 종료 조건

- C-009 생산 단계가 끝나고 S3 동기화를 확인하면 stop한다(terminate 아님).
- 할당량이 승인되어 g6e.12xlarge로 옮기면 이 인스턴스를 stop한다.
- 유휴 상태로 다음 작업 배정을 기다리지 않는다.
- 학습(SFT)에는 쓰지 않는다.

SUPERVISION_REPORT_READY C-009-launch
