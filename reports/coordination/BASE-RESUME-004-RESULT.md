# BASE-RESUME-004-RESULT (A): Gemma 4 E4B QAT 후보 확보와 개발 실행 원시 결과

**상태: 실행 완료, 판정 대기.**
- 후보 1개를 확보하고 검증했다.
- 개발 smoke 48턴과 CHAT-EVAL-004 공개 20(63턴)을 같은 설정으로 실행했다.
- **A는 판정하지 않았다.** C가 판정한다. C 사전 관문은 공개 완전 통과 57/63 이상, 모든 턴 통과 시나리오 16/20 이상, 심각 실패 0이다. 관문을 통과해도 운영 합격이 아니다.
- 이번에 하지 않은 것:
  - 비공개 80은 열지도 실행하지도 않았다. 감독이 따로 확정한 뒤에 한다.
  - 배포, 사이트 변경, 학습, AWS, 유료 사용은 0이다.
  - 실서비스는 그대로다.
  - README는 바꾸지 않았다. 실제 판정 결과가 아직 없다.
- 실패한 첫 실행(아래 3절)은 숨기지 않고 원시 그대로 남겼다. **환경 오류 기록이며, 모델 성능 퇴행으로 세지 않는다.**
- **C 판정 전 금지:** 출력 한도 증대, 설정 튜닝, 추가 후보, 비공개 실행, 배포. 실서비스는 현 상태를 유지한다. 통과·실패 관문 결과를 본 뒤에 다음을 결정한다.

## 1. 후보와 출처

| 항목 | 값 |
|---|---|
| 선택 | **공식 Google QAT GGUF 1개.** 감독이 기존 승인 범위(후보 1개, 8 GB, AWS 0) 안에서 정했다. B가 처음 추천한 ggml-org Q8_0(8.03 GB)은 이번 단계 한도 8 GB를 넘어 받지 않았다 |
| 저장소 / revision | `google/gemma-4-E4B-it-qat-q4_0-gguf` @ `4b4a2c1d584be7264f87aac328a1bc739ce81b6c` (gated false) |
| 파일 | `gemma-4-E4B_q4_0-it.gguf`, 5,154,941,280 B, sha256 `676c35070db6dbe52f93e9c864ee0fba4eddea94b9c875d9cb10daff453fbaee` |
| 검증 | 받기 전 HF `x-linked-size`·`x-linked-etag`가 B 값과 일치했다. 받은 뒤 크기와 sha256도 일치했다. `.partial`로 받아 검증한 뒤 `mv -n`으로 옮겼다(덮어쓰기 없음) |
| 저장 경로 | `runs/base-resume-004/gemma-4-E4B_q4_0-it.gguf` (gitignore) |
| checkpoint | **일반 `-it`와는 다른 QAT checkpoint다.** 카드의 `base_model`은 `google/gemma-4-E4B-it-qat-q4_0-unquantized`이고, 그 카드의 base는 `google/gemma-4-E4B-it`이다(B 보완 `e5c7a10`). GGUF 카드는 base의 revision을 적지 않는다 |
| 모드 | **텍스트 전용.** `gemma-4-E4B-it-mmproj.gguf`(비전 projector)는 받지 않았다 |
| 크기 표기 | E4B는 **effective 4.5B, 임베딩 포함 총 8B**다(B가 인용한 카드 표). "총 4B"로 표기하지 않는다 |
| 라이선스 | GGUF 카드 front matter는 `license: apache-2.0`, `license_link: https://ai.google.dev/gemma/docs/gemma_4_license`이고, 본문에 "License: Apache 2.0"이 있다(A 직접 확인). **Gemma 3의 Gemma Terms와는 다른 문서다.** 한국어 품질은 카드로 확인되지 않는다 |
| 공식 template·sampling 파일 | `google/gemma-4-E4B-it-qat-q4_0-unquantized` @ `476025a01dbf99361c062bbeca3d6a76bb4c4566`(현재 main)에서 받았다: `chat_template.jinja`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `config.json` |
| template 검증 | `chat_template.jinja` sha256 `0a2c8073…c5b5`로 B 값과 같다. B에 따르면 GGUF 업로드 시점의 base `ddb04d73`과 template 파일이 같다 |
| 다운로드 합계 | GGUF 5,154,941,280 B + 공식 소형 파일 32,197,264 B = **약 5.19 GB** (한도 8 GB 안) |

## 2. 실행 조건 (dev smoke와 공개 20이 같음)

| 항목 | 값 |
|---|---|
| 런타임 | 같은 맥북, llama.cpp `8ca4060` Metal `llama-server`, `-c 4096 -np 1 -t 4 -tb 4 --jinja`, 격리 포트 :51990 |
| 모델 실행 | 한 번에 1개, 순차 |
| template | GGUF 내장 template(sha256 `603a42db…`). 공식 파일과는 끝 줄바꿈 1자만 다르다 |
| 렌더링 일치 | system + 3턴 표본에서 **서버 프롬프트 토큰 ID가 공식 HF tokenizer 렌더링과 같다**(`token_ids_equal: true`) |
| 문자열 비교 | `text_equal`은 false다. 공식 문자열은 앞에 `<bos>` 글자가 있고, 서버는 BOS를 토큰으로 붙이기 때문이다. 토큰 ID는 같다 |
| 템플릿 인자 | `enable_thinking: false`를 명시했다. 이것은 공식 template의 기본값이다 |
| system | 서비스 `CHAT_SYSTEM`. 코드로 `shared_demo.CHAT_SYSTEM`과 같음을 확인했다 |
| sampling | 공식 `generation_config.json`: temperature 1.0, top_k 64, top_p 0.95 |
| min_p | 0. HF 설정에는 min_p가 없어서, llama.cpp 기본값 0.05를 끄려고 0으로 넣었다 |
| 실효 설정 | 서버가 보고한 값이 같다: repeat 1.0, presence 0, dry 0, xtc 0, typical 1, top_n_sigma −1. **두 실행의 모든 턴에서 실효 설정이 바뀌지 않았다**(`effective_settings_constant: true`, 턴마다 비교) |
| 출력 한도 / seed | 384 토큰 / 1 |
| history | 대화마다 자기 답을 쌓는다. `__CLEAR__`·`<NEW_CHAT>`이면 새 대화 |
| 하지 않은 것 | 재시도, 필터, 문자 차단, 질문별 처리 |

## 3. 첫 실행은 환경 오류로 무효 (원시 보존, 성능 퇴행으로 세지 않음)

- **원시:** `BASE-RESUME-004-RESULT/run1-thinking-injected-invalid/`
- **무슨 일이 있었나:** 첫 실행은 `enable_thinking`을 넘기지 않았다. 그러자 llama-server `--jinja`가 system 블록에 **`<|think|>`를 넣었다. thinking이 켜진 것이다.**
  - 공식 기본값(thinking off)과 다른 조건이었다.
  - 렌더링 검사가 이것을 잡았다. 토큰 수가 96 대 94로 달랐고, 공식 렌더에는 `<|think|>`가 없다.
- **출력에 남은 흔적:** 48턴 중 6턴에 `<|channel>thought …<channel|>` 내부 추론이 그대로 나왔다(B-S1 t3, B-S3 t1·t2·t4, P1 t2, P2 t1).
- **이 실행은 후보 조건으로 판정하지 않는다.** 환경 오류(런타임이 thinking을 켜서 공식 template과 불일치)로만 기록하고, 모델 성능 퇴행으로 세지 않는다.
- **조치:** runner를 고쳤다. 이제 공식 렌더와 토큰 ID가 다르면 대화를 시작하지 않는다.
  - 다시 실행할 때 바꾼 것은 `enable_thinking: false` 명시 하나다. 공식 기본값을 되돌린 것이고, 튜닝이 아니다.
  - 나머지 조건은 그대로다.

## 4. 개발 smoke 48턴 (원시: `BASE-RESUME-004-RESULT/dev-smoke-48/`)

| 항목 | 값 |
|---|---|
| 상태 | complete. 대화 13개, 턴 48개 |
| 렌더링 일치 / 실효 설정 고정 | true / true |
| 종료 | eos 46, 384 잘림 2 |
| 생성 속도 | 중앙값 34.7 tok/s |
| 턴당 시간 | 중앙값 4.5 s, p90 9.8 s |
| 출력 토큰 | 중앙값 146 |
| 메모리·실서비스 | 압박 단계 내내 1, 여유 45~47%. 실서비스 running |
| 내부 추론·`<\|channel>` 출력 | 0 |
| 한자·가나가 나온 턴 | 3. P1 t1 고유명사 `浅草寺`, P2 t2 요청한 번역, P3 t2 `여명(黎明)` 한자 병기 |

**A의 표면 관찰 (판정 아님, C 확인용)**
- B-S8의 새 대화 뒤 질문("제 이름이 뭐라고 했죠?", "이사하는 날")에는 "말씀해주신 적이 없다"고 답했다. 누출이나 날조는 보이지 않았다.
- B-S7 t0의 "최대 20일 정도를 기준으로 시작"은 근거 없는 규정 수치일 수 있다(G4 확인 필요).
- B-S7 t1은 연가의 근거로 "공무원보수규정"을 들었다.
- B-S4 t3 "처음 고친 버전"에서 나열한 세 표현 중 하나는 t1(짧게 한 판)의 문장이다.

**공개 20 실행의 성격:** 합격 판정이 아니라 **추가 개발 진단**이다.
- 48턴에서 이미 규정 수치 날조 가능성(B-S7 t0)이 보였다. 그러니 이 단계에서 통과 근거를 주장하지 않는다.
- 표면 확인 결과: 내부 추론 출력 0, 요청 없는 언어 전환 관찰 0, 새 대화 기억 누출 관찰 0. 이것도 A의 통과 판정이 아니다.

## 5. CHAT-EVAL-004 공개 20 (원시: `BASE-RESUME-004-RESULT/public-20-run/`)

| 항목 | 값 |
|---|---|
| 세트 | `public-20.jsonl` sha256 `908f7415…7518`. 비공개 80은 열지 않았다 |
| preflight | `scripts/chat_eval_preflight.py`에 run-config(`public-20-run-config.json`)를 넣어 **pass**. 결과: `public-20-preflight.json` |
| 실행 | 09:13:17~09:18:12 KST. complete. 시나리오 20개, 턴 63개 |
| 무결성 | 시나리오 id와 턴 순서가 세트와 같고, 63턴이 모두 서로 다르다 |
| 설정 일치 | dev smoke와 요청 설정, template, system, 실효 설정이 모두 같다(코드 비교) |
| 렌더링 일치 / 실효 설정 고정 | true / true |
| 종료 | eos 54, **384 잘림 9**. CE4-CA-04 t0~t2, CE4-EL-01 t0~t3, CE4-TR-04 t2, CE4-SW-04 t1. 잘림이 판정에 영향을 줄 수 있다 |
| 속도·시간 | 34.8 tok/s, 턴당 중앙값 3.2 s, p90 11.5 s |
| 메모리 | 압박 단계 1, 여유 44~47% |
| 내부 추론 출력 / 한자·가나 턴 | 0 / 0 |

## 6. 다음 관문

- **C 판정 대상:** dev smoke 48과 공개 20을 개발 지표로 판정한다.
  - 공개 20 사전 진입 관문은 C가 정한 세 조건(57/63, 16/20, 심각 실패 0)이다.
  - 공개 20 결과는 비공개 80과 섞지 않는다.
- **비공개 80:** 감독이 따로 확정한 뒤에만 실행한다. A는 그 파일을 열지 않는다.
- **판정 전에 하지 않는 것:** 기본 모델 교체, 배포, 설정 변경, 추가 후보.
- **남겨 둔 자산:**
  - 후보 가중치와 공식 소형 파일(`runs/base-resume-004/`)
  - 원시 결과 3개(첫 실행 무효본 포함), run-config, preflight 결과
  - runner 변경: 공식 렌더와 토큰 ID가 다르면 중단, BOS 중복 방지, `<NEW_CHAT>`과 jsonl 지원

SUPERVISION_REPORT_READY BASE-RESUME-004-RESULT
