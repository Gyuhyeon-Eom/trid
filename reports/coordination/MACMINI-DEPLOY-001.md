# MACMINI-DEPLOY-001: 맥미니 제어기 이관

- 세션 01a18739 / A
- 기준 시각: 2026-10-30 12:30 KST
- **상태:** 배포는 끝났다. **AWS 인증이 없어 launchd 등록과 실동작 스모크는 아직 하지 못했다(진짜 블로커).**

## 1. 원격 조사 (읽기 전용, 비밀 내용은 출력하지 않음)

| 항목 | 결과 |
|---|---|
| 호스트 | `eomgyuhyeon-ui-Macmini.local`, macOS 26.3.1, arm64 |
| 도구 | `/usr/bin/python3` (3.9.6), `/usr/bin/git`, `/opt/homebrew/bin/uv` 0.9.28, `/opt/homebrew/bin/python3.14`, brew |
| AWS | **CLI 없음, `~/.aws` 없음, `AWS_*` 환경변수 0개.** 다시 쓸 수 있는 기존 AWS 자격 증명이 없다 |
| 기존 trid 관련 | `~/trid-controller` 없음(새로 만듦) |
| 기존 서비스 | `~/apps/`(tridatum, tridatum-rag, tridatum-dashboard 등) |
| 기존 LaunchAgents | `co.tridatum.*` 7개, cloudflared 2개, `ai.openclaw.gateway` |
| 디스크 | 460Gi 중 66Gi 남음 |

- 기존 파일과 서비스는 **어떤 것도 수정·삭제·재시작하지 않았다.**

## 2. 배포한 것 (모두 새 경로 `~/trid-controller/`)

**`~/trid-controller/repo/`**
- 맥북 저장소 커밋 `e0779dd` 시점의 제어 관련 코드를 `git archive`로 옮겼다.
  - `trid/`, `configs/`
  - `scripts/drive_sft_campaign.py`, `launch_sm_pretrain.py`
  - `sm_*.sh` 진입점, 평가 스크립트
- 이후 `deploy/macmini-controller/`와 갱신된 driver를 추가했다.
- 원격에 로컬 git을 만들어 커밋했다(`deploy e0779dd` → `deploy controller files`). launcher가 기록하는 코드 해시를 남기기 위해서다.
- `.venv`: uv와 Python 3.14 기반, `awscli` 1.46.2(`aws` 실행 파일이 venv 안에 있음). 전역 설치는 하지 않았다.

**`~/trid-controller/state/campaign.json`**
- 내용: `deploy/macmini-controller/s1-monitor-campaign.json`
- **감시만 하는 모드(`monitor_only`)**다. 승인된 S1 두 job을 넘겨받아 상태와 누적 비용만 기록한다.
- job을 기동하거나 재시도하지 않는다.
- 누적 비용 상한은 $54.5다(9만 원 ÷ 1,650).

**`~/trid-controller/logs/`**: 비어 있다.

**launchd 템플릿**: `repo/deploy/macmini-controller/co.tridatum.trid.sft-controller.plist`
- **아직 `~/Library/LaunchAgents`에 넣지 않았고 로드하지도 않았다.** 인증 없이 로드하면 실패와 재시작만 반복하기 때문이다.
- 설정 내용:
  - 영속 상태: `~/trid-controller/state/state.json`. S3에는 `corpus/runs/s1/controller-status.json`으로 복사한다.
  - 로그: `~/trid-controller/logs/controller.{log,err}`
  - 재시작: `KeepAlive.SuccessfulExit=false`. 비정상 종료일 때만 300초 간격으로 다시 띄운다. 캠페인이 끝나 정상 종료하면 멈춘 상태로 남는다.

**인증 점검 스크립트**: `repo/deploy/macmini-controller/preflight.sh`
- ARN과 권한 PASS/FAIL만 출력한다. 키는 출력하지 않는다.
- 현재 결과: `The config profile (trid-controller) could not be found`
  - FAIL list training jobs
  - FAIL S3 corpus/
  - FAIL S1 status
  - PASS campaign file

## 3. 단일 writer 보장과 중복 방지

- 맥북에는 driver가 없다. 11:54에 smoke·spot driver 3개를 종료한 뒤 다시 띄우지 않았다.
- S1 두 job은 driver 없이 job 하나 안에서 끝나도록 기동했다(MACBOOK-INDEPENDENT-A).
- 맥미니 driver가 재시작하면 같은 run 이름으로 실행 중인 job을 먼저 조회해 넘겨받는다(`running_job`). 감시만 하는 모드에서는 job을 새로 만드는 코드 경로가 아예 실행되지 않는다.
- 다음 승인 단계(여러 job 연결)는 이 제어기에서만 기동한다. 맥북에서 같은 캠페인을 동시에 돌리지 않는다.

## 4. 블로커: 사용자 조치가 필요한 인증 (키를 채팅이나 코드에 넣지 마세요)

이 계정에서 키를 대신 만들면 비밀 정보를 옮겨야 하므로 A는 만들지 않았다.

1. **AWS 콘솔(IAM) → 사용자 → `trid-controller` 생성**
   - 콘솔 로그인 없이 만든다.
   - 인라인 정책으로 `deploy/macmini-controller/iam-policy-trid-controller.json`을 붙인다. 맥미니의 `~/trid-controller/repo/deploy/macmini-controller/`에도 같은 파일이 있다.
   - 정책 범위:
     - `trid-pt-*` training job의 생성·조회·중지·태그, job 목록 조회
     - 기존 역할 `trid-sagemaker-lora`만 PassRole
     - S3 `corpus/*` 읽기·쓰기·삭제와 목록
     - SageMaker 로그 읽기
2. 같은 사용자 → 보안 자격 증명 → **액세스 키 생성(CLI 용도)**
3. **맥미니 터미널에서 사용자가 직접 입력:**
   `~/trid-controller/repo/.venv/bin/aws configure --profile trid-controller`
   (region `ap-northeast-2`, output `json`)
4. 확인: `bash ~/trid-controller/repo/deploy/macmini-controller/preflight.sh` → 모두 PASS
5. A에게 알려주면 A가 다음을 진행한다.
   - plist를 `~/Library/LaunchAgents/`에 복사하고 `launchctl bootstrap gui/$(id -u) …`로 등록
   - SSH 접속을 끊은 뒤 다시 접속해서 확인:
     - `launchctl print`로 상태
     - `state.json` 갱신
     - S3 `controller-status.json`
     - 실제 SageMaker job 조회 결과
   - 확인 결과를 이 문서에 추가한다.

- 장기 키 대신 IAM Identity Center(SSO)를 쓰고 싶다면 맥미니에서 `aws configure sso`로 설정할 수 있다.
  - 단 SSO 토큰이 만료되면 사용자가 다시 로그인해야 한다. 무인 상시 제어에는 불리하다.

## 5. 현재 영향
- 승인된 S1 학습은 막히지 않았다. 두 job은 job 안 평가로 끝나고, 결과는 S3에 저장된다.
- 맥미니 제어기는 인증이 생긴 뒤 감시만 하는 모드로 붙는다.
- 별도 AWS 상시 제어 서버는 만들지 않았다.

SUPERVISION_REPORT_READY MACMINI-DEPLOY-001
