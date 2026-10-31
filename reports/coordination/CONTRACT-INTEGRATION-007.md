# CONTRACT-INTEGRATION-007 (A): AWS-EVAL-006 계약 불일치 3건 통합 수정

**상태: 수정과 로컬 공개 fixture 검증 완료. 실제 AWS 검증은 남음.**
- C가 `AWS-EVAL-006-C-TO-A.md`에 적은 불일치 3건을 고쳤다. 수정 범위는 A 소유 파일(runner, bootstrap, orchestrator)뿐이다.
- 바꾸지 않은 것: 모델, 프롬프트(006 system), sampling, 384·4096·seed 1, 평가 기준, C 수집기, 공통 `run()` 생성 로직.
- AWS 쓰기와 유료 실행은 0이다. root 사용도 0이다.

## 1. 수정

| # | C 지적 | 수정 (파일) | 확인 |
|---|---|---|---|
| 1 [차단] | 기대 입력 sha가 gold가 든 원본 `af46eec1…`였다. 전달하는 파생본 `4134c4bb…`(id+turns, gold 0)은 항상 거부됐다 | `scripts/aws_eval/aws_eval_006.py` `PRIVATE_INPUT_SHA = '4134c4bb66146637be00ff15a4fff78c332cda5138423ea50a646fe49265836c'`. 원본은 C 밖으로 나가지 않는다는 주석을 달았다. 이 값이 user data `INPUT_SHA`, phase의 기대 sha, `c_deliver_set.py` 대조로 이어진다 | 렌더한 full user data에 `INPUT_SHA='4134c4bb…'`(9,808 B, `bash -n` 통과) |
| 2 | checkpoint status에 실행 결합 필드가 없다(run id, 입력 sha, 실효 설정 고정 여부) | `scripts/chat_blind_runner.py` `end_dialogue`의 status에 세 필드를 넣었다. 생성 동작은 바꾸지 않았다(아래) | fixture: 두 실행 섞기와 잘못된 run id 모두 수집기가 거부. 실제 run id로 정상 수집하면 complete |
| 3 | checkpoint 쓰기가 원자적이지 않다. 실패·시한 종료 때 마지막 동기화가 없다 | runner: `*.cms.tmp`로 쓴 뒤 `os.replace`. bootstrap: 시한 처리와 상한이 있는 동기화(아래) | fixture: checkpoint 폴더에 `P-001.cms Q-001.cms`만 있고 임시 파일 0. **시한·실패 경로는 코드 판독뿐, 실측 아님** |

**2번 status 필드 상세**
- `run_id`
  - 값은 `TRID_RUN_ID` 환경 변수다. bootstrap이 실제 기동 ID `RUN_ID`(`aws-eval-006-<mode>-<UTC 시각>`, `launch`가 만듦)를 넘긴다.
  - **기본값이나 상수는 없다.** checkpoint를 만드는데 이 값이 없으면 runner가 시작을 거부한다.
- `input_sha256`: runner가 실제로 받은 대화 파일의 sha256이다. `manifest['dialogues_sha256']`와 같다.
- `effective_settings_constant`: 그 대화까지 실효 설정이 바뀌지 않았는지다. 버리던 세 번째 인자를 기록한다.
- 추가 결합 `code_sha256`: bootstrap `CODE_SHA`, 즉 그 실행의 코드 묶음 sha다. C 수집기는 이 값을 요구하지 않고 무시한다.

**3번 bootstrap 시간 상한** (안쪽부터 차례로 걸린다)

| 순서 | 시각 | 동작 |
|---|---|---|
| 1 | 상한 − 300 s | 자체 deadline. 돌고 있는 단계(runner, llama-server, 빌드, 다운로드)에 SIGTERM을 보내고, 스크립트에도 TERM을 보낸다. TERM trap이 exit → EXIT trap `finish()`로 이어진다 |
| 2 | 상한 | OS 타이머 `shutdown -h`. 인스턴스가 꺼지면 terminate된다 |
| 3 | 상한 + 10분 | 독립 EventBridge Scheduler terminate |

- **`finish()`가 하는 일**
  - 완료된 `*.cms`를 마지막으로 동기화한다(`timeout 90`).
  - phase와 로그를 올린다(각 `timeout 30`).
  - `shutdown -h now`
  - 합계 최대 150초로 deadline 여유 300초보다 짧다. 그래서 OS 타이머보다 먼저 끝난다.
- **모든 S3 전송에 시간 상한을 걸었다**
  - 코드 다운로드 120 s, 인증서 60 s, 주기 동기화 60 s, 끝 동기화 120 s, 결과 120 s
  - `AWS_RETRY_MODE=standard`, `AWS_MAX_ATTEMPTS=3`
  - 재시도가 무한히 이어지거나 종료가 늦어지는 일은 없다.

## 2. 재사용 가능한 공개 fixture (CPU, **실제 모델 실행 아님**)

**명령** (repo 루트, 포트 51990이 비어 있어야 한다, 약 25초)
```
bash scripts/aws_eval/fixture_e2e.sh
```

**구성 요소**
- `scripts/aws_eval/fake_llama_server.py`(+ `.sh`): 모델 없이 패키지의 공식 tokenizer와 template으로 `/props`, `/apply-template`, `/tokenize`, `/completion`에 답한다.
  - 답은 결정적인 자리표시자다("픽스처 응답 <hash> (모델 없음)").
  - 렌더 일치는 여기서는 자명하게 참이다. 실제 렌더 검증이 아니다.
- C 키는 이번 검증용으로 만든 일회용이다. C의 실제 키가 아니다.

**실제 코드로 지나가는 경로**
1. `c_deliver_set.smoke_set()` 바이트 → 인스턴스 키로 CMS 암호화 → 24,000자 조각 명령. 명령에 평문이 없음을 확인한다
2. bootstrap 방식 재조립과 복호화 → bootstrap 기대 sha와 일치하는지 확인
3. `chat_blind_runner.py` P·Q(`TRID_RUNNER_REMOTE=1`, `TRID_RUN_ID`, checkpoint)
4. C의 `scripts/aws_eval_collect.py`

**결과** (2026-10-31, 맥북, 메모리 압박 단계 1)
```
PASS delivery: decrypted set sha = bootstrap expectation
PASS runner: exactly P-001.cms and Q-001.cms, no partial files
PASS collector P complete: {"package": "P", "status": "complete", "turns": 2, "dialogues": 1, "stop_types": {"eos": 2}}
PASS collector Q complete: {"package": "Q", "status": "complete", "turns": 2, "dialogues": 1, "stop_types": {"eos": 2}}
PASS refused: wrong run id
PASS refused: checkpoint of another run mixed in
PASS refused: checkpoint of the smoke set against another set
ALL FIXTURE CHECKS PASSED (no model was run)
```
- 모든 검사는 실패하면 종료하도록 되어 있다(`check` → exit 1).

**실패 기록**
- 첫 실행은 Q runner가 대화 경계 guard(맥북 메모리 압박 단계 2, 여유 43%)로 멈춰 Q checkpoint가 없었다.
  - 원인은 A 커널이 RETRIEVAL-CONTINUE-007 검사용으로 올려 둔 색인과 임베더였다.
  - 그것을 해제해 단계 1(여유 54%)로 돌아온 뒤 다시 실행했다.
- 그 실행에서 fixture 검사가 일부 비치명이라 거짓 통과할 수 있음을 발견했다. 위처럼 모든 검사를 치명적으로 바꾼 뒤 재실행해 통과했다.

## 3. 파일 sha256 (이 commit)

| 파일 | sha256 (앞 16자리) |
|---|---|
| `scripts/base_dialogue_runner.py` | `e1a35e447565290f` (이번 변경 없음) |
| `scripts/chat_blind_runner.py` | `3cb7390dcd23400d` |
| `scripts/aws_eval/bootstrap.sh` | `26056456a20b9b3d` |
| `scripts/aws_eval/aws_eval_006.py` | `aafcc211c42ce46e` |
| `scripts/aws_eval/fixture_e2e.sh` | `36610c6e3ffbc051` |
| `scripts/aws_eval/fake_llama_server.py` | `c6f2d4126b3ef645` |

- runner sha가 005·precommit 기록과 달라졌다. C는 precommit 부록에 새 값을 적어 달라.

## 4. 남은 실제 AWS 검증 (operator·c-runner 로그인 뒤, 기존 순서)

1. `watchdog-test`: OS 타이머 단독 종료, Scheduler 단독 종료(첫 실제 AssumeRole), 일정 자동 삭제
2. smoke 실행: C가 `c_deliver_set.py --smoke`로 실제 SSM 조각 전달 → 원격 복호화 → P·Q checkpoint가 S3 `checkpoints/`에 올라감 → C가 실제 키로 `aws_eval_collect.py --run-id <실제 RUN_ID>` 수집해 complete
3. **deadline 경로 실측**: 짧은 `--max-minutes`로 진행 중 종료시켜, 끝난 checkpoint가 마지막 동기화로 올라가고 인스턴스가 종료되는지 본다
   - 수행 여부와 시간 예산은 GPU 장부 230분 안에서 정한다. smoke 상한을 줄이는 방식이다.
4. 실제 CUDA 렌더 일치, 실효 설정, VRAM, 바이너리 sha

## 5. C 재검증 요청

- C의 공개 fixture CPU E2E(`CONTRACT-CONTINUE-007`)를 새 runner와 orchestrator로 다시 돌려 달라. 확인 항목:
  - 수정된 수집기 필드 계약과 일치하는지
  - 파생본 `4134c4bb…` 전달이 받아들여지는지
- 위 `fixture_e2e.sh`도 같은 확인에 쓸 수 있다. 모델은 실행하지 않는다.

SUPERVISION_REPORT_READY CONTRACT-INTEGRATION-007
