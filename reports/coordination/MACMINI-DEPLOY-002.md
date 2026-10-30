# MACMINI-DEPLOY-002: 맥미니 독립 감시 가동 확인

- **세션:** 01a18739 / A
- **결과:** **감시 전용(monitor_only) 제어기가 맥미니 launchd에서 SSH 없이 동작함을 확인했다.**
- **범위 밖:** job 발행, IAM 변경, 삭제, 학습 발행은 하지 않았다.

## 1. 인증 상태 (부모와 사용자 조치)
- **부모 조치:** 부모가 맥미니에 AWS CLI v2(2.37.13, `/opt/homebrew/bin/aws`)를 설치했다.
- **사용자 조치:** 사용자가 Safari에서 승인해 `aws login --profile trid-controller --region ap-northeast-2`에 성공했다.
- **identity는 root다.** 장기 키는 없고 임시 login 세션이다(최대 12시간).
- **A의 범위:** A는 설치, 로그인, 키 생성을 하지 않았다. 자격 증명을 출력하거나 복사하지 않았다.

## 2. A가 바꾼 것 (맥북 커밋 → 맥미니 repo 동기화)

| 커밋 (맥북 / 맥미니) | 내용 |
|---|---|
| `636cd61` / `d258f23` | CLI v2 명시 경로. driver `TRID_AWS_CLI`, plist PATH 맨 앞 `/opt/homebrew/bin`과 `TRID_AWS_CLI=/opt/homebrew/bin/aws`, preflight의 v2 검사. `.venv` CLI v1은 login 프로필을 이해하지 못하므로 쓰지 않는다. |
| 같은 커밋 | 인증 만료·부재 처리. `auth.status=reauth_required`와 "사용자 재인증 필요"를 기록하고 대기한다. 반복 재시작이나 job 생성은 하지 않는다. |
| `cd27844` / `4edf476` | root 가드. 매 주기 `sts get-caller-identity`로 신원을 확인해 `identity.kind`를 기록한다. root면 경고("루트 임시 로그인 세션: 감시 전용으로만 사용, job 발행 금지")를 남긴다. monitor_only가 아닌 캠페인은 root에서 **즉시 종료**한다(맥북 실측: "Refusing to drive a launching campaign with root credentials", rc=1). preflight에도 root 경고를 넣었다. |

**맥북 검증:**
- 없는 프로필 → reauth_required
- 가짜 키 → reauth_required
- 정상 자격 → 해제
- root + 발행 캠페인 → 거부

## 3. 맥미니 실측
1. **preflight** (`~/trid-controller/repo/deploy/macmini-controller/preflight.sh`):
   - `PASS CLI v2`
   - `identity: …:root` → `WARN root temporary login session`
   - ListTrainingJobs, S1 Describe, `corpus/runs/s1/` List, `controller-status.json` Put·Get, campaign 파일 모두 PASS
   - exit 0
2. **수동 1회 실행:** `drive_sft_campaign.py --once`. 13:09:53에 두 job InProgress를 읽고 identity root를 기록했다.
3. **launchd 등록:** `~/Library/LaunchAgents/co.tridatum.trid.sft-controller.plist`, `launchctl bootstrap gui/501` → 0.
   - 상태: `state = running`, pid 79435, runs 1, never exited.
   - 기존 `co.tridatum.*` 에이전트는 건드리지 않았다.
4. **SSH 종료 후 독립 동작:**
   - 등록 SSH 세션을 닫았다. 약 5분 뒤 새 SSH로 접속해 확인했다.
   - 13:15:05 주기가 실행됐다(로그 2줄, err 0바이트, 재시작 없음).
   - `state.json` 갱신 결과:
     - 91M: **Completed**, 12,000단계
     - 390M: InProgress, 3,700단계
     - identity: root
     - auth: 없음(정상)
   - 같은 내용이 S3 `corpus/runs/s1/controller-status.json`에 13:15:05로 올라갔다.
   - SageMaker 직접 조회와 일치했다(91M Completed, Billable 2,676초).

## 4. 운영 조건과 한계
- **monitor_only다.** 이 제어기는 job을 만들지 않는다(S1 재시도 0 정책).
  - 다른 세션 job은 조회하지 않는다(busy-type 스캔 생략).
  - S3 recovery 정리(삭제)도 하지 않는다.
- **root 세션은 상시 발행 권한으로 보지 않는다.** 무인 자동 발행이 필요해지면 따로 승인을 받아야 한다. 그때 최소 권한 역할(`iam-policy-trid-controller-publish-FUTURE.json`)로 전환하는 것이 전제다.
- **세션 만료:** login 세션은 최대 12시간이다. 만료되면 제어기는 `reauth_required` 상태로 대기한다.
  - 이때 S3 status는 갱신되지 않는다(마지막 정상 시각이 남는다). 로컬 `state.json`과 로그에 재인증 필요가 표시된다.
  - 사용자가 맥미니에서 `aws login --profile trid-controller`를 다시 실행하면 다음 주기에 자동으로 재개된다.
- **S1 390M 예상:** 13:09 기준 약 1.7시간 남았고 job 안 평가가 이어진다. 현재 세션 안에 끝날 것으로 [INFERENCE] 본다.
- **맥북:** 이 감시에 관여하지 않는다(단일 writer: 맥미니). 맥북은 보고서 작성과 읽기 조회만 한다.
- **중지 방법:** `launchctl bootout gui/501/co.tridatum.trid.sft-controller`. 캠페인이 끝나면(두 arm 완료, 평가 완료) driver가 exit 0으로 멈추고, KeepAlive(SuccessfulExit=false)라 재시작하지 않는다.

## 5. 감시 완료 기록 (15:03)
- **완료 기록:** 15:01:23 주기에 맥미니 감시기가 두 S1 job Completed를 기록했다. 맥북의 SageMaker 직접 조회와 일치한다(91M Billable 2,676초, 390M 9,266초).
- **결함:** 감시 모드에서는 job 안 평가가 끝나도 controller에 "평가 대기"가 남았다. 그래서 완료된 캠페인이 종료되지 않고 감시를 계속했다(작업 생성 없음, status 쓰기만).
- **수정 (커밋 `353a241`, 맥미니 `2f69fe5`):** monitor 모드에서는 controller 평가 대기를 비운다. 맥북에서 같은 캠페인으로 실행해 즉시 종료(rc 0)되는 것을 확인했다.
- **맥미니 반영:** `launchctl kickstart -k`로 반영했다. 한 주기(15:03:46)를 돌고 끝났다.
  - 결과: `state = not running`, `runs = 2`, `last exit code = 0`
  - KeepAlive(SuccessfulExit=false)라서 재시작하지 않는다.
- **현재 상태:** 맥미니 제어기는 **idle(정상 종료)**이다. 등록은 남아 있다. 다시 쓰려면 새 캠페인 파일과 별도 승인이 필요하다. root 발행 가드는 유지한다.

SUPERVISION_REPORT_READY MACMINI-DEPLOY-002
