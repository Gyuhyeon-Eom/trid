# CHAT-BLIND-005-PRECOMMIT: 비공개 80 독립 진단의 사전 기록 (결과 전)

기록: C(01a17c28), 2026-10-31.
- **생성 전이고 결과 전이다. 추론 0.**
- 기계 판독본: `CHAT-BLIND-005-PRECOMMIT.json` (sha256 `71f324a0…`, 커밋 메시지에도 적음)

## 1. 규칙 예외의 범위
- **해제한 것:** "공개 관문 통과 전 비공개 80 보류"(`CHAT-EVAL-004-GATE.md` 2절) 하나뿐이다. 감독이 **진단 목적으로만** 명시적으로 해제했다. 사용자 고지는 감독이 완료했다.
- **해제하지 않은 것:** 운영 합격, 배포, 서비스 평가, 기준·임계값 변경.
- **두 모델 중 하나가 비공개 관문을 통과해도:**
  - 공개 관문은 FAIL로 남는다.
  - 서비스 경로(RAG, router, `RULE_CLAIM`, presence 1.5, 첫 턴 문장)는 평가되지 않는다.
  - **자동 배포는 하지 않는다.**
- **공개 20에 맞춘 추가 튜닝은 중단했다.**

## 2. 고정 항목 (해시)

| 항목 | 값 |
|---|---|
| 질문·rubric | 비공개 80 `runs/c-private/chat-eval-004-private-80.jsonl` sha256 `af46eec1…`. 80 시나리오, 250 사용자 턴. 항목 기준(expect, track, severe, boundary, external)은 이 파일 안에 있고 바꾸지 않는다 |
| manifest / schema | `e4e77000…` / `36843d67…` |
| gate | gate.json `4e1d1457…`, 관문 문서 `68dcabd3…`, 집계기 `7a8043c7…` |
| 기준 (모델별 따로) | 완전 통과 턴 **≥ 225/250**, 모든 턴 통과 시나리오 **≥ 64/80**, **심각 실패 0**. partial은 분자에서 뺀다 |
| runner | `scripts/base_dialogue_runner.py` `1193146b…` (기존 RAW 평가 runner 재사용) |
| server | llama.cpp `8ca4060` Metal `llama-server` `bf5eeac5…` |
| preflight | `scripts/chat_eval_preflight.py` `63a3cf2b…`. 모델별 run-config로 exit 0이어야 실행한다 |
| blind 도구 | `scripts/chat_blind_pack.py` `186bd60e…` |
| system (양쪽 같음) | `CHAT-CAPABILITY-006/chat-system-candidate.txt` `43a16276…` |
| 공통 생성 조건 | max_tokens 384, context 4096, seed 1, `enable_thinking: false`, raw 모드 |
| 렌더 검사 | 서버 프롬프트 토큰 ID가 공식 HF 렌더링과 다르면 runner가 중단한다 |

## 3. 비교 대상

| 모델 | 파일 sha256 | 공식 template 출처 | 고정 sampling (출처) |
|---|---|---|---|
| 현재 기본 원본 **Qwen3-1.7B-Q8_0** | `061b54da…` | `qwen3-1.7b-hf-rev70d244cc` tokenizer_config template, non-thinking | temperature 0.7, top_p 0.8, top_k 20, min_p 0 (Qwen3-1.7B@70d244cc README의 non-thinking 권장값) |
| 후보 **Gemma 4 E4B QAT q4_0** | `676c3507…` | `hf-qat-unquantized-476025a0` chat_template.jinja, thinking off | temperature 1.0, top_k 64, top_p 0.95, min_p 0 (generation_config.json 476025a0) |

- **비교 범위:** 같은 system, 같은 한도, 각 모델 공식 template과 고정 sampling으로 raw 경로를 비교한다.
- **서비스와 다른 점:** 서비스는 Qwen에 presence_penalty 1.5를 쓰고, RAG·라우팅도 있다.
- **결론의 범위:** 결과는 "raw 모델 + 006 system" 비교다. **사이트 개선으로 일반화하지 않는다.**

## 4. 절차
1. **A:** P/Q 별칭 패키지를 만들고 매핑 commitment를 커밋한다(5절).
2. **C preflight:** P와 Q의 run-config로 각각 실행해 exit 0이어야 한다.
3. **C 생성:**
   - P → Q 순서로 한 번에 한 모델만 돌린다. 로컬 Metal, 격리 포트를 쓴다.
   - runner 메모리 guard가 압박을 감지하면 중단하고 미완료로 보고한다. 재실행은 감독 승인 뒤에만 한다.
   - 실서비스는 바꾸지 않는다. AWS·외부 API는 0이다.
   - 출력은 `runs/c-private/chat-blind-005/`(gitignore)에 둔다.
4. **C blind pack:** `chat_blind_pack.py pack`으로 시나리오마다 두 대화록을 무작위 1/2 순서로 섞는다.
   - 모델, 별칭, 파일, 시간 정보는 패킷에서 뺀다.
   - 순서 키는 따로 저장하고 열지 않는다.
5. **C 판정:** 500 사용자 턴을 패킷만 보고 판정한다. 봉인 기준, 판정 예시, "모름" 원칙을 그대로 쓴다.
6. **봉인:** 판정 파일 sha256을 `CHAT-BLIND-005-PROGRESS.md`에 커밋한다.
7. **공개와 대조:**
   - 키로 unblind한다.
   - A가 매핑과 salt를 공개하고, C가 commitment를 검증한다.
   - 모델별로 봉인 집계기를 돌린다.
8. **최종 보고:** `CHAT-BLIND-005.md` READY.

## 5. A 요청: P/Q 별칭 패키지와 매핑 commitment
- **위치:** `runs/blind-005/` (gitignore)
  - 하위 `P/`, `Q/`
  - 각각: `model.gguf`(복사나 하드링크), `hf/`(공식 tokenizer·template 파일), `run.json`
- **`run.json` 필드:**
  - `alias`, `expected_sha256`, `sampling`, `sampling_source`, `template_kwargs` (`{"enable_thinking": false}`)
  - 그리고 preflight 필드: `model_sha256`, `template`, `max_tokens` 384, `mode` raw, `scenario_set_sha256` `af46eec1…`
- **commitment:**
  - 매핑 JSON 예: `{"P": {"model_id", "sha256"}, "Q": {…}, "salt": "<32바이트 hex>"}`
  - 그 sha256만 `CHAT-BLIND-005-COMMIT.md`에 **C의 생성 전에** 커밋한다.
  - 매핑 원문과 salt는 C가 판정 봉인 해시를 커밋한 뒤에 공개한다.
- **A가 하지 않는 것:** 비공개 파일을 열거나 실행하지 않는다.

## 6. blind의 한계 (미리 밝힘)
- **C가 생성을 맡으므로 별칭을 추정할 수 있다.**
  - runner manifest에 모델 sha256이 기록되는데, C는 두 sha를 이미 안다.
  - template 토큰과 sampling 값도 모델마다 다르다.
- **판정 단계의 blind도 절차상의 blind다.** 패킷에서 식별 필드는 뺐지만, 답의 문체(길이, 마크다운, 이모지)로 모델이 드러날 수 있다.
- **판정자는 C 단일 AI다.** 순서 키도 C가 만든다. 독립 판정자의 blind가 아니다.

## 7. 이견과 정정
- `CHAT-EVAL-004/errata.jsonl` 규칙을 그대로 쓴다.
- 결과를 본 뒤 기준과 임계값은 바꾸지 않는다.
- 다시 확인할 수 있는 C 기대 값 오류만 정정으로 인정한다. 그때는 원래 판정과 정정 판정을 둘 다 보고하고, 관문 판정은 원래 판정으로 한다.

- **진행 상황:** `CHAT-BLIND-005-PROGRESS.md`에 개수와 단계만 적는다.
- **비공개 원문과 답:** C만 보관한다. 개발팀에 공유하지 않는다.

## 8. 생성 전 부록 (A 인계 `b6c35eb` 반영, 아직 결과 없음)
- **매핑 commitment (A):** `bc58255820a43ad5cf5c862e9830a97b6eb0941930cd8b9033d81b28dcbe2feb`
  - 매핑 파일: `runs/chat-blind-005/sealed-mapping.json`, nonce 256비트, A 보관
- **패키지:**
  - 경로: `runs/chat-blind-005/packages/{P,Q}`
  - package.json sha: P `14c8dce1…8e32`, Q `c381396d…8867` (A 보고값)
  - **C는 채점을 봉인하기 전까지 열지 않는다.** package.json, symlink(readlink 포함), tokenizer, `sealed/`(manifest, turns, timings, log)가 모두 해당한다.
- **실행기:** `scripts/chat_blind_runner.py` `917eeb5a…`
  - 이 실행기가 쓰는 `scripts/base_dialogue_runner.py` `1193146b…`는 precommit 값과 같다. A의 리팩터는 precommit 전에 이미 반영돼 있었다.
- **공통 설정:** `CHAT-BLIND-005/shared-settings.json` `debbd459…`
  - system sha `a8c0d301…`은 006 파일의 앞뒤 공백을 제거한 본문 기준이다. 파일 자체의 sha는 `43a16276…`이다.
- **sampling 배정:** 두 모델의 sampling은 3절 값 그대로다. 어느 별칭이 어느 값인지는 C가 모른다.
- **blind 도구 변경:**
  - `chat_blind_pack.py`가 실행기 출력(`outputs.jsonl` + `status.json`)을 읽게 했다. `sealed/`는 읽지 않는다.
  - 새 sha256 `05b0c752cd5cbfda0902f781db986e7cb5d88a12be206f54afcbb92618599a99`
  - 검증: 기존 형식과 새 형식 모두 pack이 되고, 미완료 상태는 거부한다.
- **preflight:**
  - run-config: `CHAT-BLIND-005/c-run-config-{P,Q}.json`. 식별 값은 봉인된 패키지 참조로만 기록했다.
  - `chat_eval_preflight.py` 실행 결과 P·Q 모두 **pass(exit 0)**. 결과 파일은 `c-preflight-{P,Q}.json`이고 개수만 담고 있다.
- **채점 근거:** 채점 전에는 `outputs.jsonl`과 `status.json`만 쓴다. 채점은 응답 순서를 섞은 패킷으로 한다.
- **실패 처리:** 실행이 실패하면 출력을 그대로 두고 원인만 알린다. 자동 재실행이나 튜닝은 하지 않는다.
- **봉인 commit 직후:** 매핑 공개 요청을 A에게 직접 전달한다.
