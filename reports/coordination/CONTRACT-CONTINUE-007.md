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
1. A가 고칠 것 (번호는 3절 표 기준. 표 1번은 C 수집기 결함이고 이미 C가 고쳤다):
   - 표 3번: orchestrator 기대 입력 sha를 파생본 `4134c4bb…`로 바꾼다. **full 차단 사유다.** `AWS-EVAL-006-C-TO-A.md`에서는 1번이다.
   - 표 2번: checkpoint의 run_id, input_sha256, effective_settings_constant. C-TO-A 2번
   - 표 4번: checkpoint 원자 쓰기와 종료 때 동기화. C-TO-A 3번
2. 사용자: 로그인
3. A: watchdog, smoke 기동
4. C: smoke 전달(`c_deliver_set.py --smoke`)과 smoke checkpoint 수집
   - 수집 명령: `aws_eval_collect.py --set <smoke 세트 파일> --run-id <smoke run>`
5. 실행 전 precommit 부록을 커밋한 뒤 full을 진행한다.

## 6. 부록: A 통합 수정 뒤 재검증 도구 (감독 결과 수령 반영)
- **번호 정정:** 5절의 A 작업 번호를 3절 표 기준으로 고쳤다. 표 1번(C 수집기, 완료)과 표 3번(orchestrator 입력 sha, full 차단)을 구분한다.
- **도구:** `scripts/aws_eval_e2e_check.py` (C 소유, CPU, AWS 0, 모델 0)
  - 공개 20으로 실제 부품을 끝까지 돌린다: 전달 키 → 암호화 조각 명령 실행 → 복호화 → **수정 없는 A runner `main()`**의 P/Q checkpoint를 공유 디렉터리에 → 신규 수집기로 alias별 복원 → `chat_blind_pack`
  - 모델 생성만 005·006 공개 원시 고정 응답으로 바꾼다. 출력에도 "not a GPU or AWS success"라고 적힌다.
  - 기본은 일회용 키쌍이라 C 개인키 없이도 누구나 실행할 수 있다.
- **판정 항목 (하나라도 실패하면 exit 1)**
  - 전달 매개변수에 평문 없음, 복호화 sha 일치
  - checkpoint 임시 파일 잔여 없음
  - P와 Q가 각 63턴 complete로 복원되고 pack 통과
  - **다른 run 섞임은 그 run id 때문에 거부**
  - **실효 설정 변화는 "not constant" 때문에 거부**
  - 다른 이유로 거부되면 실패로 친다.
- **총괄 실행 명령:** `.venv/bin/python scripts/aws_eval_e2e_check.py` (A가 run id 환경 변수를 다른 이름으로 쓰면 `--run-id-env <이름>`)
- **현재 결과 (A 수정 전 runner):** exit 1
  - delivery true, 임시 파일 잔여 없음 true
  - collect P/Q false("run id None"), pack false, 섞임 거부 false, 설정 변화 거부 false
  - 섞임과 설정 변화는 처음에 run id 누락 때문에 "거부"로 집계돼 **거짓 통과**가 났다. 거부 사유까지 확인하도록 고쳤다.
- **도구 자체 검증:**
  - C-TO-A 2번 요청 그대로 세 필드(`run_id`=`TRID_RUN_ID`, `input_sha256`=`dialogues_sha256`, `effective_settings_constant`)를 넣은 **임시 사본 runner**로 돌렸다. A 파일은 수정하지 않았다.
  - 결과: 7개 항목 모두 true. P 63턴 complete, 섞임은 "run id 'e2e-run-2'"로 거부, 설정 변화는 P-011에서 "not constant"로 거부
- **이 도구가 확인하지 않는 것:**
  - C-TO-A 3번의 bootstrap `finish()` 종료 동기화는 셸과 AWS 동작이다. 수정 뒤 코드 판독으로 확인한다.
  - 원자 쓰기는 runner 쪽 구현이면 임시 파일 잔여 검사로 일부만 드러난다.
- **A 수정 인계를 받으면 할 일:**
  1. 이 명령을 그대로 다시 돌린다.
  2. A 통합 보고에 독립 결과를 반영한다.
  3. 실제 AWS smoke에서 run_id, 세트 sha, 실효 설정 고정이 **실제 run 값**인지 확인한다. 누락되거나 바뀐 값은 거부를 유지한다.

SUPERVISION_REPORT_READY CONTRACT-CONTINUE-007
