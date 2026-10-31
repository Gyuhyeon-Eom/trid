# AWS-EVAL-006-PROGRESS (C)

비밀 정보, 비공개 문항, 비공개 답은 적지 않는다. 개수와 단계만 적는다.

## 현재: 준비 완료, 실행 차단 (실행 아님)
- **실제 GPU 평가는 아직 없다.** C가 쓴 AWS 자원 0, 원격 추론 0.
- **고친 내용:** 앞 판의 "A가 user·role·profile·SG 생성 중", "B 비용 검증 대기"는 지난 상태다.
  - 둘 다 이미 끝났다: `AWS-EVAL-006-IDENTITY.md`(`0e13b57`), `AWS-EVAL-006-COST.md`(`437562c`)
- **실행 차단 사유 (갱신)**
  1. **사용자 로그인:** 비밀번호, MFA, `aws login --profile trid-aws-eval-006-c-runner`. 남은 차단 사유다.
  2. ~~A 계약 불일치 3건~~ → **A 수정 완료** (`CONTRACT-INTEGRATION-007`, `9ae88bf`)
     - 기대 입력 sha가 파생본 `4134c4bb…`로 바뀌었다.
     - checkpoint에 run_id, input_sha256, effective_settings_constant가 들어간다.
     - `.cms.tmp`로 쓴 뒤 rename하고, 종료 때 시간 제한을 건 동기화를 한다.
     - 확인 범위: C 독립 검증(CPU, 공개 고정 응답) PASS. 감독 `fixture_e2e.sh`도 PASS. **실제 AWS·CUDA 확인은 미완료다.**
- **C가 완료 전까지 하지 않는 것:** 비공개 모델 상세(패키지 내용, symlink, sealed)나 매핑을 열람하지 않는다.

## CHAT-BLIND-005 로컬 종료 (사실 기록)
- **완료 수:** P 235/250턴(75/80대화), Q 0/250(실행 안 함)
  - 중단 원인: 메모리 guard. 압박 단계 2가 두 번 나왔다(1차 실행 70번째 대화 전, 묶음 2 두 번째 대화 전). 재시도 없음.
- **상태:** 미채점, 미봉인, blind pack 없음. 출력 내용은 C도 읽지 않았다.
- **원시 해시:** `CHAT-BLIND-005-PROGRESS.md` 마지막 절에 부분별 outputs/status sha가 있다.
- **자원:** 격리 서버 종료, 포트 51990 해제, C 프로세스 0. 로컬 자원 해제를 A에 서면 통보했다(`CHAT-BLIND-005-REQUEST-A.md` 3절).
- **처리:** 덮어쓰기와 재생성은 없다. 006과 섞지 않고, 하나의 점수로 합치지 않는다.

## AWS-EVAL-006 사전 준비 결과

| 단계 | 상태 | 검증 명령·관측 결과 |
|---|---|---|
| precommit 초안 | 완료 | `AWS-EVAL-006-PRECOMMIT.md`: 같은 세트라 새 독립 문항 주장 안 함, 고정 해시, Metal→CUDA 차이, 중단 처리. 하드웨어 값은 실행 전 부록으로 커밋 |
| checkpoint 암호화 | 완료 | C 인증서 `AWS-EVAL-006/c-checkpoint-cert.pem`(지문 `73:C9:…:06:A8`), 개인키는 C 비공개. `openssl cms` 왕복 일치, 암호문에 평문 없음 |
| 결과 복원 도구 | 완료 | `scripts/aws_eval_collect.py`. 공개 원시 데이터 fixture: 정상 P/Q 복원·pack 통과(원시 출력과 같음). 중복·질문 변경·백엔드 불일치·잘못된 키는 exit 2로 거부. 누락은 incomplete로 기록되고 pack이 거부 |
| 비공개 세트 파생본 | 완료 | id·turns만 남김: 80행, 250턴, gold 필드 0, sha `4134c4bb…`(parent `af46eec1…`). 원격에는 이것만 암호문으로 보낸다 |
| 전송 보안 계약 | A 구현(SSM 조각 전달, S3 미사용)에 C가 조건부 합의. 조건은 기대 sha를 파생본으로 할 것 | 인스턴스 키쌍 → C가 CMS 암호화 → 암호문만 S3나 매개변수로 → 인스턴스 안에서 복호화. SSM 매개변수·출력·CloudTrail에 평문·base64 평문 금지. 모의 인스턴스 키로 점검: 14.8 KB 암호문에 평문 없음, 복호화 결과 원본과 같음 |
| A/B 요청 | 제출 | `AWS-EVAL-006-REQUEST-AB.md`: 비용 한도, 자동 종료, CUDA `8ca4060`, 익명 패키지와 commitment, 출력 마스크, 암호화 checkpoint, CUDA 스모크, 원격 실행 경로 |

## 다음 단계

| 단계 | 상태 |
|---|---|
| A 인프라(신원·정책 생성 완료, 비용 검증 완료) | 사용자 로그인 대기. A 계약 수정 3건 대기 |
| B 비용 검증 | 완료 (`AWS-EVAL-006-COST.md`) |
| 실행 전 precommit 부록·원격 preflight(세트 파생본 sha, 패키지 sha, 렌더 일치, 공통 설정) | 대기 |
| 원격 생성 P / Q | 0/250, 0/250 |
| checkpoint 복원·결합, 500턴 blind 채점·봉인, 매핑 공개·검증, 최종 보고 | 대기 |

**다음 행동:** A의 non-root 로그인과 인계가 오면 계약 항목을 확인하고, 실행 전 부록을 커밋한 뒤 원격 생성을 시작한다.

## 실제 구현 대상 흐름 검증 (CONTRACT-CONTINUE-007, CPU, 공개 fixture)
- **검증한 흐름:** bootstrap 키 생성 → `c_deliver_set` 방식 암호화와 조각 명령 실행 → 인스턴스 쪽 복호화 → 수정 없는 `chat_blind_runner.main()`의 실제 checkpoint → 수집기
  - 고정 공개 응답을 썼고 AWS 호출은 0이다.
- **관측 결과:**
  - 전달: 암호문 4,437 B, 명령 2개, 매개변수에 평문 없음, sha 일치
  - checkpoint: P와 Q 40개 파일이 한 디렉터리에 생겼다.
- **C 수집기 결함 수정:**
  - 공유 prefix에서 수집이 불가능했던 문제를 고쳤다.
  - 다른 실행, 다른 세트, 설정 변화가 섞이면 통과하던 문제를 고쳤다.
- **회귀 테스트:** `.venv/bin/python -m unittest tests.test_aws_eval_collect` Ran 5 OK. 수정 전 수집기로는 failures 4, errors 1
- **자원 상태:** 로컬 격리 서버 0, C 프로세스 0, AWS 자원 사용 0. 임시 검증 파일은 삭제했다.
- **다음 행동:**
  1. A가 C-TO-A 1·2·3번을 수정한다.
  2. 사용자가 로그인한다.
  3. A가 smoke를 기동한다.
  4. C가 `c_deliver_set.py --smoke`로 전달하고 smoke checkpoint를 수집해 확인한다. 이것이 첫 실제 응답·checkpoint 확인이다.
  5. 실행 전 부록을 커밋하고 full을 진행한다.

(위 차단 사유 기록 시점의 완료 표시는 `a296f97` 판에 있다. 이번 갱신은 새 AWS 응답이나 새 차단 사유가 아니라서 표시를 달지 않는다.)

## A 통합 수정 뒤 C 독립 확인 (CPU 계약 검증, 실제 AWS 아님)
- `.venv/bin/python scripts/aws_eval_e2e_check.py`(수정 없는 A runner) → exit 0, 7항목 true
  - 전달 4,436 B·명령 2개·평문 없음·sha 일치
  - P/Q 각 63턴 complete, pack 통과
  - 섞인 run은 run id 때문에 거부, 설정 변화는 "not constant"로 거부
- 회귀 테스트 `tests.test_aws_eval_collect` Ran 5 OK
- 실행기 해시는 `AWS-EVAL-006-PRECOMMIT.md` 7절에 기록했다.
- **자원:** AWS 사용 0, 로컬 서버·C 프로세스 0
- **실제 AWS 체크는 미완료다:** smoke 전달, 실제 checkpoint, run_id·세트 sha·설정 고정이 실제 run 값인지, 종료·기한 동기화 실측
- **다음 행동:**
  1. 사용자 로그인
  2. A smoke 기동
  3. C `c_deliver_set.py --run-id <smoke run> --smoke`
  4. `aws_eval_collect.py --set <smoke 세트> --run-id <smoke run>`로 복원 확인. 이것이 첫 실제 응답·checkpoint 확인이다. 그때 완료 표시를 단다.

