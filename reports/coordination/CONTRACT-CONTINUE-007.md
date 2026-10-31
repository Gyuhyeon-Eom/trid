# CONTRACT-CONTINUE-007 (C): AWS-EVAL-006 전송·수집 계약의 독립 검증

작성: C(01a17c28), 2026-10-31.
- **이 작업에서 쓴 것:** CPU만. AWS 호출 0, GPU 0, 모델 추론 0.
- **이 작업에서 하지 않은 것:**
  - 비공개 80 내용 변경
  - 005 미채점 출력 열람
  - 매핑, 패키지 내용 열람
  - 새 임계값
- **현재 실제 blocker:** 사용자 비밀번호, MFA, 전용 CLI 로그인. 그리고 아래 A 계약 불일치 1번.

## 1. 이미 검증된 것 (반복하지 않음)
- C checkpoint 인증서의 CMS 왕복
- 이전 수집기의 개별 거부 fixture(중복, 질문 변경, 백엔드 차이, 잘못된 키, 누락 → incomplete)
- id+turns 파생본 생성(gold 0)과 모의 인스턴스 키 전달
- 근거: `AWS-EVAL-006-PROGRESS.md` 준비 절. 이번에는 A의 **실제 구현**을 끼운 흐름만 검증했다.

## 2. 이번 검증: 실제 흐름 끝까지 (로컬 파일, 공개 고정 응답)

**구성**
1. bootstrap과 같은 `openssl req rsa:3072` 명령으로 인스턴스 키와 인증서를 만들었다.
2. `c_deliver_set`과 같은 방식으로 공개 20의 id+turns 파생본을 처리했다: CMS 암호화, base64, `CHUNK=24000` 분할.
3. 같은 형식의 셸 명령 문자열(`printf '%s' '<조각>' >> …/set.cms.b64`, `.ready`)을 bash로 실제 실행했다.
4. 인스턴스 쪽 base64 복호화, `cms -decrypt`, sha 확인을 했다.
5. **수정하지 않은 `chat_blind_runner.main()`**을 돌렸다(`TRID_RUNNER_REMOTE=1`).
   - llama-server 생성 대신 005·006 공개 원시 응답을 고정으로 공급했다.
   - 메모리 guard는 1로 고정했다.
   - 패키지는 임시 더미 디렉터리를 썼다. 실제 P/Q 패키지는 열지 않았다.
   - 이렇게 bootstrap처럼 P와 Q의 실제 CMS checkpoint를 **한 디렉터리**에 만들었다.
6. 수집기를 실행했다.

**관측 결과**

| 단계 | 결과 |
|---|---|
| 전달 | 암호문 4,437 B, 명령 2개. SSM 매개변수에 평문·평문 base64 없음. 인스턴스 쪽 sha 일치 |
| checkpoint | P-001…P-020, Q-001…Q-020, 40개 파일 |
| 로컬 차이 | macOS `base64`는 `-d FILE` 형식을 받지 않아(exit 64) 로컬에서만 `-d < FILE`로 바꿨다. 인스턴스(Linux coreutils)는 해당 없다 |

## 3. 새로 찾은 불일치·결함

| # | 소유 | 내용 | 실제 소비자 영향 | 처리 |
|---|---|---|---|---|
| 1 | **C 수집기** | P/Q가 한 `checkpoints/` prefix를 쓰는데 이전 수집기는 다른 alias 파일을 만나면 전체를 거부했다 | 실제 run에서 prefix를 받아 `--alias P`로 돌리면 **"Q-001.cms: alias 'Q'"로 수집 불가.** 재현됨 | **C 수정** |
| 2 | **A runner** | checkpoint에 run_id, 입력 sha, 실효 설정 고정 여부가 없다(`_constant`를 버림) | 다른 실행의 보완 절반이 섞여도 **"complete"로 통과.** run1 6~20 + run2 1~5로 재현됨 | A에 전달. C 수집기는 세 필드를 요구하도록 수정 |
| 3 | **A orchestrator** | `PRIVATE_INPUT_SHA = af46eec1…`(gold 포함 원본) | 파생본(`4134c4bb…`) 전달이 "sha differs"로 거부된다. **full이 계약상 차단** | A에 전달(차단) |
| 4 | **A bootstrap** | checkpoint를 최종 이름으로 바로 쓰고, 실패나 시한 종료 때 마지막 동기화가 없다 | 마지막 30초 안의 checkpoint 누락, 잘린 암호문 → 수집 거부 가능. **코드 판독이며 실행 재현은 아니다** | A에 전달 |

- **계약 불일치 아님으로 확인:**
  - smoke payload와 수집기 파생본이 바이트 단위로 같다.
  - 비공개 80에서 수집기가 계산한 파생본 sha가 기록값 `4134c4bb…`와 같다.
  - 코드 묶음에 비공개 파일이 없다.
- **A 전달 문서:** `AWS-EVAL-006-C-TO-A.md`. 정확한 파일·행, 재현 내용, 요청 변경이 들어 있다. A 파일은 C가 고치지 않았다.

## 4. C 수정 (최소)
- **`scripts/aws_eval_collect.py`**
  - 파일 이름 `^[PQ]-NNN.cms$`만 받는다.
  - 이름과 내용의 alias가 다르면 거부한다. 다른 alias 파일은 건너뛴다.
  - `--run-id`(필수, 승인된 이어 실행일 때만 여러 개)를 받는다.
  - checkpoint `input_sha256`이 `--set`의 id+turns 파생본 sha와 같아야 한다.
  - `effective_settings_constant`가 true여야 한다.
  - 출력 status에 `input_sha256`과 run별 개수를 남긴다.
- **회귀 테스트 `tests/test_aws_eval_collect.py`** (unittest 5개, 매번 임시 키, 결정적)
  - 공유 prefix에서 요청 alias만 수집
  - 다른 run 거부
  - 다른 전달 세트 거부
  - 실효 설정 변화 거부
  - 이름과 내용 alias 불일치 거부
- **실행 증거**
  - 수정 후: `.venv/bin/python -m unittest tests.test_aws_eval_collect` → **Ran 5, OK**
  - 수정 전 수집기로 같은 테스트(`--run-id`만 빼고): **failures 4, errors 1**
    - 다른 run, 다른 세트, 설정 변화를 모두 exit 0으로 받아들였다.
    - 공유 prefix에서는 수집에 실패했다.
  - 수정한 수집기를 A의 현재 runner checkpoint에 적용: "P-001.cms: run id None not among --run-id"(exit 2). A가 2번을 고치기 전에는 섞인 실행을 받아들이지 않는다.

## 5. 남은 일
1. A: 1번(차단), 2번, 4번 수정
2. 사용자: 로그인
3. A: watchdog, smoke 기동
4. C: smoke 전달(`c_deliver_set.py --smoke`)과 smoke checkpoint 수집
   - 수집 명령: `aws_eval_collect.py --set <smoke 세트 파일> --run-id <smoke run>`
5. 실행 전 precommit 부록을 커밋한 뒤 full을 진행한다.

SUPERVISION_REPORT_READY CONTRACT-CONTINUE-007
