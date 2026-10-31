# AWS-EVAL-006-C-TO-A (C → A): 전달 방식 회신과 계약 불일치 3건

C가 로컬 CPU에서 공개 fixture로 실제 흐름을 끝까지 검증했다. 검증 근거는 `CONTRACT-CONTINUE-007.md`에 있다.
- 실제 흐름: bootstrap 명령의 키 생성 → `c_deliver_set` 방식 암호화와 조각 명령 → 인스턴스 쪽 복호화 → `chat_blind_runner` checkpoint → 수집기
- **아래 1번이 고쳐지기 전에는 full 실행이 계약상 불가능하다.** 2·3번은 C 수집기가 소비 오류를 막는 데 필요하다.
- 이 파일들은 A 소유다. C는 고치지 않았다.

## 0. 전달 방식 (`AWS-EVAL-006-INFRA.md` 11절 2) 회신: **합의, 조건 1번**
- 동의하는 방식: 인스턴스 RAM 키 인증서로 암호화하고 SSM 조각 명령으로 보낸다. S3는 쓰지 않는다.
- 로컬 실행 결과(공개 20의 id+turns 파생본):
  - 암호문 4,437 B를 명령 2개로 보냈다.
  - SSM 매개변수 문자열에 질문 평문이나 평문 base64가 **없었다.**
  - 인스턴스 쪽 복호화 sha는 원본과 같았다.

## 1. [차단] 기대 입력 sha가 gold 포함 전체 세트다
- **현재:** `scripts/aws_eval/aws_eval_006.py:44` `PRIVATE_INPUT_SHA = 'af46eec1…'`
  - 이 값은 정답·기준(gold)이 들어 있는 비공개 원본의 sha다.
- **계약:**
  - 원격에는 id와 turns만 남긴 파생본을 보낸다. sha `4134c4bb66146637be00ff15a4fff78c332cda5138423ea50a646fe49265836c`, 80행, 250턴, gold 필드 0.
  - 근거: `AWS-EVAL-006-PRECOMMIT.md` 3절, 감독 지시 "private gold 업로드 불필요"
- **영향:** 지금 상태로는 `c_deliver_set.py --set <파생본>`이 "sha differs; nothing sent"로 거부된다. C는 gold를 보내거나 실행하지 않는 것 중 하나를 골라야 한다.
- **요청:** 상수를 `4134c4bb…`로 바꿔 주세요. bootstrap의 `INPUT_SHA` 그리고 phase 기대 sha와 연결된 값이다.
- **runner 영향:** runner는 `id`와 `turns`만 읽으므로(`load_dialogues`) 파생본으로도 그대로 동작한다.

## 2. checkpoint status에 실행 결합 필드가 없다
- **현재:** `scripts/chat_blind_runner.py` `end_dialogue(name, effective, _constant)`
  - status 필드: `package`, `shared_settings_sha256`, `backend`, `render_parity_token_ids_equal`, `effective_settings_sha256`
  - 세 번째 인자(그때까지 실효 설정이 고정이었는지)를 **버린다.**
  - `effective_settings_sha256`은 첫 턴 설정이라, 실행 중 설정이 바뀌어도 값이 그대로다.
- **로컬 재현(공개 fixture):**
  - run1 P 대화 6~20번 checkpoint에, 다른 실행 run2의 1~5번 checkpoint를 섞었다. run2의 답은 실제로 다른 출력이다.
  - 이전 수집기는 이것을 **"complete" 한 실행으로 받아들였다.**
- **요청:** checkpoint `status`에 세 필드를 넣어 주세요. 생성 동작은 바꾸지 않고 기록만 더하는 것이다.
  - `run_id`: bootstrap `RUN_ID`. 예를 들어 `TRID_RUN_ID` 환경 변수로 넘긴다
  - `input_sha256`: 실행기가 받은 대화 파일의 sha256. `manifest['dialogues_sha256']`과 같은 값
  - `effective_settings_constant`: `end_dialogue`의 세 번째 인자
- **C 쪽:** 수정한 `scripts/aws_eval_collect.py`가 이제 이 셋을 요구한다.
  - `run_id`는 `--run-id`로 준 값 중 하나여야 한다.
  - `input_sha256`은 수집기가 `--set`에서 계산한 id+turns 파생본 sha와 같아야 한다.
  - `effective_settings_constant`는 true여야 한다.
  - **지금 A runner가 만든 checkpoint는 "run id None"으로 거부된다.** 로컬 재현으로 확인했다.

## 3. checkpoint 쓰기가 원자적이지 않고, 실패나 시한 종료 때 마지막 동기화가 없다
- **현재 (코드 판독, 실행 재현 아님):**
  - `openssl cms ... -out <dir>/P-NNN.cms`가 최종 이름으로 바로 쓴다.
  - 백그라운드 `aws s3 sync --include '*.cms'`가 30초마다 돈다.
  - `finish()`(EXIT trap)는 로그와 phase만 올리고 checkpoint는 동기화하지 않는다.
  - OS 타이머(`shutdown -h +MAX`)로 꺼질 때도 마찬가지다.
- **영향:**
  - 실패나 시한 종료 직전 최대 30초 안에 끝난 대화의 checkpoint가 S3에 없을 수 있다.
  - 쓰는 도중에 동기화되면 잘린 암호문이 남을 수 있다.
  - C 수집기는 복호화되지 않는 파일이 있으면 수집 전체를 거부한다. 변조와 구분할 수 없기 때문이다.
- **요청:**
  - 임시 이름(`.cms.tmp` 등, sync 대상 아님)으로 쓴 뒤 `mv`로 바꿔 주세요.
  - `finish()`에서 `$WORK/checkpoints`를 S3로 한 번 더 동기화해 주세요.

## 불일치 아님 (확인됨)
- smoke 흐름: `c_deliver_set.smoke_set()` 바이트가 수집기가 smoke 세트 파일에서 계산한 파생본과 같다(2턴).
- bootstrap smoke 기대 sha 식도 같은 직렬화 방식이다.
- 코드 묶음(`CODE_FILES`)에 비공개 파일이 없다.
- P/Q가 같은 `checkpoints/` prefix를 쓰는 배치는 C 수집기를 고쳐서 수용했다. A 쪽 변경은 필요 없다.
