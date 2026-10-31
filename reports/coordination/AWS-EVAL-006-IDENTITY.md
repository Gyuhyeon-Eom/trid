# AWS-EVAL-006-IDENTITY (A): 전용 권한 초기 설정 완료 — 사용자 로그인 설정 대기

**상태: 생성과 읽기 검증 완료.**
- 사용자 승인("전용 권한 초기 설정 승인")에 따라 root로 1회 실행했다.
  ```
  python3 scripts/aws_eval/aws_eval_006.py bootstrap-identity --user-approved
  ```
  - 시각: 2026-10-31 13:12~13:15 KST.
  - 첫 시도는 아래 1절의 정책 크기 한도로 중간에 멈췄다. 같은 명령(멱등)으로 다시 실행해 끝냈다.
- **root로 만든 것:** IAM과 SG만 만들었다.
- **유료 자원 0:** EC2 실행·대기 0, Scheduler 일정 0. 기존 정책과 자원은 바꾸지 않았다.
- 비밀번호, 토큰, 비공개 문항은 이 문서에 없다.

## 1. 생성된 것과 읽기 검증 (같은 시각 API 조회)

| 자원 | 검증 결과 |
|---|---|
| IAM 사용자 `trid-aws-eval-006-operator` (A) | access key **0**, 콘솔 비밀번호 없음(아직), MFA 0, inline 0, 그룹 0. 부착: `SignInLocalDevelopmentAccess`, `policy/trid/trid-aws-eval-006-operator`(v1, 내용이 repo `operator-policy.json`과 같음) |
| IAM 사용자 `trid-aws-eval-006-c-runner` (C) | access key **0**, 콘솔 비밀번호 없음(아직), MFA 0, inline 0, 그룹 0. 부착: `SignInLocalDevelopmentAccess`, `policy/trid/trid-aws-eval-006-c-runner`(v1, `c-runner-policy.json`과 같음) |
| 역할 `trid-aws-eval-006-instance` + 같은 이름의 인스턴스 프로필 | trust와 inline이 repo 파일과 같다. 부착: `AmazonSSMManagedInstanceCore`. 프로필의 역할도 이 역할 하나다 |
| 역할 `trid-aws-eval-006-scheduler` | trust가 repo와 같다: `aws:SourceArn` = `arn:aws:scheduler:ap-northeast-2:148761639313:schedule-group/default`, `aws:SourceAccount` = 148761639313. inline은 태그가 있는 인스턴스만 종료할 수 있다. 부착 정책 0 |
| SG `sg-0e7e5b23c7c152ebc` `trid-aws-eval-006-no-ingress` | 기본 VPC `vpc-0bef0e7b66432d802`, **인바운드 규칙 0**, 아웃바운드 전체 허용(HF·GitHub·S3·SSM 접속용), 태그 `trid:eval=aws-eval-006` |

**정정: 사용자 정책을 inline에서 고객 관리형으로 바꿨다**
- IAM inline 사용자 정책은 합계 2,048 B로 제한된다. operator 정책은 공백을 빼도 4,313 B라 첫 실행이 `LimitExceeded`로 멈췄다(그 시점 operator 사용자에는 정책이 붙지 않았다).
- 그래서 두 사용자 모두 같은 이름의 고객 관리형 정책(경로 `/trid/`, 한도 6,144 B)으로 바꿨다. 정책 문서 내용은 바꾸지 않았다.
- 승인 범위(사용자 2, 역할 2와 프로필, SG 1)에 포함된 권한 부착 방식만 다르다.

## 2. 사용자가 직접 할 일 (콘솔, root로 로그인한 브라우저에서)

**1단계: 두 사용자에게 콘솔 비밀번호와 MFA를 설정한다.**
- operator: https://us-east-1.console.aws.amazon.com/iam/home#/users/details/trid-aws-eval-006-operator?section=security_credentials
- c-runner: https://us-east-1.console.aws.amazon.com/iam/home#/users/details/trid-aws-eval-006-c-runner?section=security_credentials
- 각 페이지의 "보안 자격 증명"에서 **콘솔 액세스 활성화**(사용자 지정 비밀번호)를 켜고, **MFA 디바이스 할당**을 한다.
- **"액세스 키 만들기"는 누르지 않는다.** 장기 키는 쓰지 않는다.

**IAM 사용자 로그인 정보** (계정 별칭이 없어서 계정 ID를 쓴다)
- 로그인 URL: https://148761639313.signin.aws.amazon.com/console
- 계정 ID: `148761639313`
- 사용자 이름: `trid-aws-eval-006-operator` / `trid-aws-eval-006-c-runner`

## 3. CLI 로그인 (맥북 터미널, 사용자가 두 번)

- 맥북 `~/.aws/config`에 프로필 두 개를 만들어 두었다. 리전(ap-northeast-2)만 들어 있고 자격 증명은 없다.
  - `trid-aws-eval-006-operator`
  - `trid-aws-eval-006-c-runner`
- `aws login`이 로그인한 사용자로 이 프로필에 `login_session`을 기록한다. 세션은 최대 12시간이다.
- **root 세션과 섞이지 않게 하는 방법**
  - 브라우저가 root로 로그인돼 있으면 그 세션이 그대로 쓰일 수 있다.
  - 그래서 `--remote`로 URL만 받아 **비공개(시크릿) 창**에서 연다. 창마다 하나의 사용자로 로그인한다.

```
aws login --profile trid-aws-eval-006-operator --remote
#  → 출력된 URL을 새 시크릿 창에서 연다 → "IAM 사용자" 선택 → 계정 ID 148761639313,
#    사용자 trid-aws-eval-006-operator, 비밀번호, MFA → 화면에 나온 코드를 터미널에 붙여 넣는다 → 시크릿 창을 닫는다
aws sts get-caller-identity --profile trid-aws-eval-006-operator --query Arn --output text
#  → arn:aws:iam::148761639313:user/trid-aws-eval-006-operator 이어야 한다(root가 나오면 잘못 로그인한 것)

aws login --profile trid-aws-eval-006-c-runner --remote      # 새 시크릿 창에서 같은 방법으로
aws sts get-caller-identity --profile trid-aws-eval-006-c-runner --query Arn --output text
#  → arn:aws:iam::148761639313:user/trid-aws-eval-006-c-runner
```

## 4. 이후 행동

- **사용자:** 2절(비밀번호와 MFA)과 3절(로그인 두 번)을 끝내고 알려 준다. 비밀번호와 코드는 아무 문서에도 적지 않는다.
- **A:** `get-caller-identity`가 두 사용자 ARN으로 나오면, 추가 질문 없이 operator 신원(`AWS_PROFILE=trid-aws-eval-006-operator`)으로 순서대로 진행한다.
  1. `preflight`와 `launch --mode smoke --dry-run`
  2. `watchdog-test`: t3.micro 2대로 OS 타이머와 Scheduler 독립 종료를 실측한다. Scheduler 역할의 실제 AssumeRole도 여기서 처음 확인된다.
  3. `launch --mode smoke`: C가 `c_deliver_set.py --smoke`로 공개 2턴을 암호화 전달한다.
  4. 결과를 `AWS-EVAL-006-INFRA.md`에 추가하고 C에 인계한다.
  - root로는 EC2나 Scheduler를 기동하지 않는다(스크립트가 거부한다).
- **C:** 생성 완료와 로그인 대기를 `AWS-EVAL-006-A-TO-C.md`로 알렸다. C의 CLI 신원은 `trid-aws-eval-006-c-runner` 프로필이다.

SUPERVISION_REPORT_READY AWS-EVAL-006-IDENTITY
