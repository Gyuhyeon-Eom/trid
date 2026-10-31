# CHAT-BLIND-005-HANDOFF (A → C): 비공개 80 진단용 블라인드 P/Q 패키지와 실행기

**상태: 준비 완료.**
- C가 직접 순차 실행할 수 있다.
- A는 서버를 띄우지 않았고, 비공개 80을 실행하거나 열지 않았다.
- C에게 질문이나 답을 요구하지 않았다.

**목적과 범위 (감독 결정)**
- **진단 목적만이다.** 감독은 "공개 관문 통과 전에는 비공개 실행을 보류한다"는 규칙만 이번 진단에 한해 바꿨다(사용자 고지 완료).
- **바뀌지 않은 것**
  - 합격 기준
  - 운영 배포 관문
  - 006의 FAIL(공개 55/63, 14/20, 심각 0). 취소하지 않는다.
- **이번 진단으로 보는 것:** 공개 20에 과적합했는지.
- **결과로 하지 않는 것:** 자동 튜닝, 추가 prompt 수정, 기본 모델 교체, 배포.

## 1. 비교 대상 (A만 아는 매핑으로 P/Q에 무작위 배정)

| | 내용 |
|---|---|
| 후보 2개 | 현재 006 후보 system 계약 + **Gemma 4 E4B QAT** / 현재 실서비스 기본 **원본 Qwen3-1.7B** |
| 배정 | `secrets.randbelow`로 P/Q를 정했다. A만 안다 |
| **비교의 성격** | 두 패키지는 system 계약(006), 출력 한도 384, 문맥 4096, seed 1, 공식 template, `enable_thinking:false`(명시)가 같다. **sampling은 각 모델에서 이미 확인한 설정을 그대로 쓴다.** 그래서 이 비교는 **모델 계열과 그 설정을 함께** 비교한다. 모델만의 차이로 해석하지 않는다 |

## 2. C에게 주는 것 (비식별)

**패키지 경로**
- `runs/chat-blind-005/packages/P`
- `runs/chat-blind-005/packages/Q`

**공통 설정:** `reports/coordination/CHAT-BLIND-005/shared-settings.json` (sha256 `debbd459…93ec6`)
- system 계약 파일 `CHAT-CAPABILITY-006/chat-system-candidate.txt`와 그 sha
- `enable_thinking:false`
- max_tokens 384, context 4096, seed 1
- 격리 포트 51990, raw 모드, llama.cpp `8ca4060`
- 렌더링 일치 필수: 서버 프롬프트 토큰 ID가 공식 tokenizer 렌더와 다르면 실행을 중단한다

**실행기:** `scripts/chat_blind_runner.py`. 공통 runner `scripts/base_dialogue_runner.py`의 `run()`을 그대로 쓴다.

**실행 명령 (C 소유, 한 번에 하나씩 순차로)**
```
.venv/bin/python scripts/chat_blind_runner.py --package P --dialogues runs/c-private/chat-eval-004-private-80.jsonl --out <C 출력 디렉터리>/P
.venv/bin/python scripts/chat_blind_runner.py --package Q --dialogues runs/c-private/chat-eval-004-private-80.jsonl --out <C 출력 디렉터리>/Q
```
- 실행 전 preflight(`scripts/chat_eval_preflight.py`, 비공개 세트 sha `af46eec1…`)는 C가 한다.

**출력 구성**

| 파일 | 내용 | C가 채점 전에 열어도 되나 |
|---|---|---|
| `<out>/outputs.jsonl` | dialogue, turn, question, output, stop_type, truncated_at_limit | **채점용. 연다** |
| `<out>/status.json` | 상태, 턴 수, stop 집계, 세트 sha, 공통 설정 sha, 렌더링 일치 여부, 실효 설정 고정 여부, 메모리·실서비스 상태, outputs와 sealed 파일들의 sha256 | 연다 |
| `<out>/sealed/` | `manifest.json`(template, 서버·실효 설정, sampling, 모델 sha), `turns.jsonl`(프롬프트, 토큰 ID, timings), `llama-server.log` | **채점 봉인 전에는 열지 않는다** |
| `runs/chat-blind-005/packages/<P\|Q>/` | 별칭 `model.gguf`, `tokenizer`와 `package.json`(sampling, 기대 sha) | **열지 않는다. 실행기만 읽는다** |

- 실행기 stdout은 상태, 턴 수, stop 집계, 렌더링·설정 일치 여부, 메모리, 실서비스 상태만 보인다. 모델 식별 정보는 내지 않는다.

**안전**
- 대화를 시작할 때마다 메모리 압박 단계와 실서비스 상태를 확인한다. 1이 아니거나 running이 아니면 그 자리에서 멈추고 부분 결과를 남긴다.
- 출력 디렉터리가 비어 있지 않으면 거부한다.
- 실행이 끝나면 격리 서버를 반드시 종료한다.
- 실서비스(:51873, :51912, :51917, :51918)는 건드리지 않는다.
- AWS 0.

## 3. 봉인 (A)

**매핑 파일:** `runs/chat-blind-005/sealed-mapping.json` (gitignore, A 보관)
- **sha256 `bc58255820a43ad5cf5c862e9830a97b6eb0941930cd8b9033d81b28dcbe2feb`**
- 내용: 무작위 nonce(256비트), P/Q별 모델 출처, 파일, sha256, tokenizer, sampling과 그 출처
- nonce가 있어서 두 가지 경우를 대입해 해시를 맞히는 방법은 통하지 않는다.

**패키지 설정 sha256**
- P `package.json` `14c8dce1…8e32`
- Q `package.json` `c381396d…8867`

**공개 시점:** C가 채점 봉인 증거(판정 파일 sha가 들어간 commit)를 올린 **뒤에만** A가 매핑 파일 원문을 공개한다. 이 sha와 같은지는 누구나 대조할 수 있다.

## 4. 사전 점검 (A, 평가 문항 아님)

- **두 패키지 실행 확인:** 평가 세트가 아닌 임시 3줄 대화(인사, 새 대화, 회상 질문)로 돌렸다.
  - 둘 다 complete, eos 2/2
  - 렌더링 토큰 ID 일치 true, 실효 설정 고정 true
  - 메모리 압박 단계 1, 실서비스 running
  - 출력은 지웠다.
- **공통 runner 리팩터 확인:** `run()`과 `load_dialogues()`를 분리했다. 기존 `base_dialogue_runner.py` CLI도 같은 임시 파일로 2턴 실행해 정상 동작을 확인했다.

## 5. 블라인드 한계 (완전 차단 아님)

- **파일 정보로 구분 가능:** `model.gguf`·`tokenizer`는 symlink 별칭이라 `readlink`로 원본 경로가 보인다. `package.json`의 sampling 값도 모델 계열을 드러낸다. **실행기만 읽는다는 약속**에 기대는 구조다.
- **출력으로 추측 가능**
  - 말투, 서식(이모지, 목록), 답 길이
  - 384 잘림 빈도
  - 실행 시간과 속도 차이(모델 크기가 다르다)
  - architecture 추측은 완전히 막을 수 없다.
- **실행 시점:** 두 패키지를 같은 머신에서 순차로 실행하므로 실행 시각과 메모리 기록이 다르다.

## 6. 하지 않은 것

- 서버 실행(점검 외), 비공개 80 열람·실행
- 기존 모델·서비스 변경
- 학습, AWS, 추가 후보, prompt 수정

SUPERVISION_REPORT_READY CHAT-BLIND-005-HANDOFF
