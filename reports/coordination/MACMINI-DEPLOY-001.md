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

A는 장기 키를 자동으로 만들지 않았다. 맥북의 기존 자격 증명도 복사하지 않았다.

### 4.1 감시 전용 최소 권한 (지금 붙일 것)

정책 파일: `deploy/macmini-controller/iam-policy-trid-controller.json` (맥미니에도 같은 파일 있음)

**권한을 정한 근거: 실제 호출 기록.** 감시 모드 driver를 `TRID_AWS_TRACE`로 한 번 실행해 실제로 나가는 AWS 호출을 모두 기록했다.
- `sagemaker describe-training-job` ×2 (S1 job 두 개)
- `s3 cp` (`corpus/runs/s1/<run>/sft/training.json`, `metrics.json` 읽기)
- `s3api list-objects-v2` (`corpus/runs/s1/<run>/sft/milestones/` 목록)
- `s3 cp` (`corpus/runs/s1/controller-status.json` 쓰기)
- job이 끝난 뒤 재시작했을 때만 `list-training-jobs --name-contains`

**허용하는 것은 이 다섯 가지뿐이다.**

| 동작 | 대상 |
|---|---|
| `sagemaker:DescribeTrainingJob` | `training-job/trid-pt-s1-*` |
| `sagemaker:ListTrainingJobs` | `*` (이 API는 리소스 단위로 제한할 수 없다) |
| `s3:GetObject` | `corpus/runs/s1/*` |
| `s3:ListBucket` | prefix 조건 `corpus/runs/s1/*` |
| `s3:PutObject` | `corpus/runs/s1/controller-status.json` 한 객체만 |

**없는 권한:** 삭제, 다른 prefix 읽기·쓰기, job 생성·중지, PassRole, 로그 읽기.

**driver 변경:** 감시 모드에서는 다른 job의 인스턴스 종류 조회와 S3 recovery 정리(삭제)를 하지 않게 고쳤다(커밋 `d98abf5`).

**검증 (IAM `simulate-custom-policy`):**
- allowed:
  - S1 job Describe
  - ListTrainingJobs
  - `s1/…/training.json` Get
  - prefix `corpus/runs/s1/…/milestones/` List
  - `controller-status.json` Put
- implicitDeny:
  - 다른 세션 job(`trid-c010-…`) Describe
  - CreateTrainingJob
  - `pretrain/` Get
  - prefix `corpus/runs/pretrain/` List
  - `s1/…/model.safetensors` Put, Delete
  - PassRole

**preflight:** `preflight.sh`가 위 호출을 그대로 점검한다. 맥북의 root 자격으로 로직만 확인했을 때 모두 PASS였다. 맥미니에서는 프로필이 없어서 FAIL이다(정상).

### 4.2 이후 job을 띄우는 단계용 정책 (지금 붙이지 않음)

파일: `iam-policy-trid-controller-publish-FUTURE.json`
- 다음 승인 단계에서만 따로 붙인다.
- 내용:
  - `trid-pt-*` job의 생성·중지·태그
  - 역할 `trid-sagemaker-lora`만 PassRole
  - `corpus/code/pretrain/*` 쓰기
  - 승인된 캠페인 prefix만 읽기·쓰기
  - 같은 캠페인의 `sft/checkpoints/*`만 삭제(recovery 정리용)

### 4.3 인증 방식 (2026-10-30 갱신: 장기 키 안내 중단)

- **이전 안내 철회:** 이 문서의 이전 판은 IAM 사용자를 만들고 장기 액세스 키를 발급하는 절차를 안내했다. 총괄 지시로 이 경로를 **중단**한다.
- **현재 진행:** 부모가 맥미니에서 AWS CLI v2로 업데이트하고 `aws login`을 실행하는 방안에 대해 사용자 승인을 요청하고 있다.
  - 맥미니 CLI는 현재 1.46.2다.
  - 인증 세션은 최대 12시간이다. 영구 무인 운영은 아니다.
  - A는 승인 전에 로그인을 실행하지 않는다. 맥북 자격 증명도 복사하지 않는다.
- **로그인 후 할 일:**
  - 세션의 권한도 감시 동작에 필요한 만큼으로 좁힌다. 기준은 4.1의 최소 권한 정책과 그 시뮬레이션 결과다.
  - `AWS_PROFILE`을 로그인 프로필로 지정하고 `preflight.sh`를 돌려 6개 항목이 모두 PASS인지 확인한다.
  - 그다음 A가 plist를 등록한다. 이어서 SSH를 끊었다가 다시 접속해 다음을 확인한다:
    - `launchctl print`
    - `state.json` 갱신
    - S3 `controller-status.json`
    - S1 job 상태 대조
  - 세션이 만료되면 감시는 AccessDenied나 만료 오류로 멈춘다. 이 경우 job 상태는 바뀌지 않는다.

- **CLI v2 전환 준비 완료 (커밋 `636cd61`, 맥미니 `d258f23`):**
  - 부모가 맥미니에 `/opt/homebrew/bin/aws`(aws-cli 2.37.13)를 설치했다.
  - driver는 `TRID_AWS_CLI`, plist는 PATH 맨 앞 `/opt/homebrew/bin`과 `TRID_AWS_CLI=/opt/homebrew/bin/aws`, preflight는 v2 확인 후 같은 경로를 쓴다. 따라서 `.venv`의 CLI v1(1.46.2)을 쓰지 않는다.
- **인증 만료·부재 처리:**
  - driver는 자격 증명 만료나 부재 오류(ExpiredToken, InvalidClientTokenId, 프로필 없음 등)를 받으면 상태를 `auth.status=reauth_required`, "사용자 재인증 필요"로 기록한다.
  - 그 뒤 interval마다 대기한다. 종료해서 launchd가 반복 재시작하게 만들지 않는다. job도 만들지 않는다.
  - 정상 호출이 다시 성공하면 `auth` 항목을 지운다.
  - 맥북 실측(`--once`)으로 확인했다:
    - 없는 프로필 → reauth_required
    - 가짜 키 → reauth_required
    - 정상 자격 → 해제
  - 로그에는 시각과 상태만 남는다. 토큰은 출력하지 않는다.
- **launchd:** 아직 미등록이다. 부모가 `trid-controller` 프로필로 로그인하고 STS를 검증한 뒤 등록한다.
## 5. 현재 영향
- 승인된 S1 학습은 막히지 않았다. 두 job은 job 안 평가로 끝나고, 결과는 S3에 저장된다.
- 맥미니 제어기는 인증이 생긴 뒤 감시만 하는 모드로 붙는다.
- 별도 AWS 상시 제어 서버는 만들지 않았다.

SUPERVISION_REPORT_READY MACMINI-DEPLOY-001
