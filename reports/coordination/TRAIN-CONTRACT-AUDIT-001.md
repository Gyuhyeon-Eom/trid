# TRAIN-CONTRACT-AUDIT-001: 수리본(model-repair-002) 34행이 실제로 무엇을 학습했나

- **범위**
  - model-repair-002의 학습 입력: `data/case-training-view-v1r1` train 34행 × 2 epoch. manifest sha `5a8368e5…`, train sha `e4eb5f67…`.
  - 학습기 `scripts/train_open_weight_lora.py`.
  - 서빙된 GGUF 2개: P = 수리본 `21715b4f…`, Q = 원본 `29508ae0…`.
- **하지 않은 것**
  - 학습, forward 실행, GPU 사용: 0.
  - eval 원문·답은 보지 않았다. 블라인드 수치는 감독이 준 값만 인용했다.
  - 기존 파일은 바꾸지 않았다. v2r1은 투입하지 않았다.
- **방법**
  - 학습 때와 같은 환경을 썼다(uv, torch 2.9.1, transformers 4.57.6, peft 0.17.1, offline).
  - 학습기의 `load_rows`·`encode`를 그대로 import했다. label 텐서는 `batch_loss`와 같은 방식으로 만들고 한 칸 shift했다. 모델 가중치는 읽지 않았다.
  - 결과 `TRAIN-CONTRACT-AUDIT-001/audit.json`에 행별 34개 기록이 있다.
  - 스크립트 `scripts/audit_case_training_contract.py`, code sha `522a5098…`.
- **학습기 버전:** 마지막 커밋이 `425c0ac`(10-30 18:09)이고 학습 종료는 21:06이다. 그 뒤 커밋은 없다.
  - 학습 당시 미커밋 수정이 있었는지는 확인할 수 없다. [추정] 같은 코드다.
  - `trid/case_prompts.py`와 `prompts.py`는 학습 전(13:34) 이후 바뀌지 않았다.

## 1. 행별 답 말미의 supervised 경계 (MODEL-REPAIR-001과 다른 점: 이번 34행 그 자체를 봄)

| 확인 | 결과 (34행) |
|---|---|
| 마지막으로 supervised된 target (shift 후)이 EOS `<\|im_end\|>`(151645)인가 | **34/34** |
| 답 안의 EOS 개수 | 34행 모두 정확히 1개(말미) |
| `<\|endoftext\|>`(151643, pad)이 supervised된 위치 | 0 |
| target 문자열 안에 특수 토큰 글자(`<\|im_end\|>`, `<\|endoftext\|>`, `<think>`)가 있는가 | 0 |
| target 끝 공백·줄바꿈 | 0 |
| 프롬프트와 답을 따로 토큰화한 결과 = 이어 붙여 토큰화한 결과(경계에서 병합 없음) | 34/34 |
| max_tokens 6144 때문에 버려진 행 / 잘린 행 | 0 / 0 (최대 전체 길이 2,841) |
| 답 토큰(EOS 포함) | 최소 78, 중앙값 157, **최대 252**. 200 이상 8행, 240 이상 3행. 256 초과 0 |
| 데이터에 저장된 `answer_tokens` = 학습기 집계 | 34/34 |
| 마지막 4개 supervised 토큰 예 | `D`,`2`,`].`,`<\|im_end\|>` / `해야`,` 합니다`,`.`,`<\|im_end\|>` |

- **배치 형상 스모크 (CPU):** 가장 짧은 행(375토큰)과 가장 긴 행(2,841토큰)을 `batch_loss` 방식으로 한 배치에 pad했다.
  - 두 행 모두 마지막 supervised target이 EOS다.
  - pad 위치가 supervised된 경우는 0이다.
  - 실제 학습은 micro batch 1이었다. 그래서 pad 자체가 없었고, packing도 없다(한 시퀀스에 한 행).
- **결론:** 34행 모두에서 답의 끝 EOS가 실제로 loss에 들어갔다. 마스크, 잘림, 템플릿 경계 어디에서도 반례를 찾지 못했다.

## 2. 학습 EOS = 서빙 정지 토큰, 템플릿 일치

- **HF tokenizer**
  - `eos_token = <|im_end|>`(151645), `pad = <|endoftext|>`(151643).
  - `generation_config.eos_token_id = [151645, 151643]`.
  - 학습 원본 dir과 merged dir의 tokenizer, template, generation_config 파일 sha가 모두 같다.
- **GGUF P·Q**
  - `tokenizer.ggml.eos_token_id = 151645 (<|im_end|>)`, `add_bos_token = false`.
  - chat template sha `a55ee1b1…`는 HF `chat_template.jinja`와 같다(P·Q 모두).
- **llama.cpp 8ca4060** (eval 생성 커밋, `src/llama-vocab.cpp`): 이름으로 `<|im_end|>`와 `<|endoftext|>`를 EOG(생성 종료)로 등록한다. 학습이 가르친 끝 토큰은 서빙에서 정지 토큰이다.
- **서빙 프롬프트**
  - `/apply-template`(enable_thinking false) 결과가 학습과 같은 no-think 접두어로 끝나는지 서빙 코드가 검사한다.
  - minja 렌더 결과와 HF 렌더 결과를 글자 단위로 비교하지는 않았다. 템플릿이 같다는 것만 확인했다.

## 3. 학습 답 길이와 서빙 한도

| 경로 | 생성 한도 | 학습 답과의 관계 |
|---|---|---|
| 학습 계약 `max_answer_tokens` | 256 (EOS 포함) | 34행 최대 252 |
| 서비스 case 모드 `answer_document_case` | 256 | 학습 답은 모두 이 안에서 끝난다 |
| 값 계약 평가 `evaluate_external_chat.MAX_TOKENS` | 64 | 학습 답의 최소가 78이다. case 형식 답은 64 안에 끝날 수 없다 |
| 블라인드 패키지 생성 | 512 | 학습 답보다 넉넉하다 |

- **사실:** 학습 답 중 64토큰 이하는 0행이다. 이 모델에 값 계약(64)으로 물으면 case 형식 답이 잘릴 수 있다.
- **판단 보류:** 값 계약 98→14가 이 때문이라고 단정하지 않는다. 형식 이동인지, 잘림인지, 내용 손상인지는 출력을 봐야 나뉘는데 eval 출력은 보지 않았다.

## 4. case/value 형식 혼합

- 34행 모두 adapter `case`이고 system은 `CASE_SYSTEM`이다. value 형식(값만 쓰는 SYSTEM, `prompt_answer`) 행은 0이다. **혼합은 없다.**
- 다르게 말하면, 값 계약을 유지시키는 재연습(replay) 행이 하나도 없다.
- **형식 분포 (34행):**
  - table_lookup 15, statute_application 8, code_unit_lookup 4, form_field_lookup 4, official_answer_plain_explanation 3.
  - "결론:"으로 시작하는 target 0행, "확인 불가" 9행.
  - 마지막 줄이 "기준:"·"시점:"·"출처:"·"단위:"·"주의:"·"조건:" 같은 안내로 시작하는 행은 14/34다(행별 `last_line`).
  - target 안에서 같은 문장이 반복된 행은 0이다. 즉 반복을 직접 가르친 데이터는 없다.

## 5. 해석: EOS 원인으로 단정하지 않음

- **확인된 것:** 데이터와 학습기 경로에는 EOS 학습을 막는 반례가 없다.
  - 34/34행이 끝 EOS를 supervised했다.
  - 잘림, 병합, 템플릿 불일치, 혼합이 모두 0이다.
  - 그 EOS는 서빙 정지 토큰이다.
- 그러므로 블라인드 결과(EOS 3 vs 32, 같은 문장 반복 9 vs 0)를 **"EOS가 학습되지 않아서"라고 단정하지 않는다.**
- **남는 가설 (검증 안 함, 서로 구분됨):**

  | 가설 | 근거 |
  |---|---|
  | H1 데이터·단계가 작다 | 고유 34행 × 2 epoch = 68 예제, optimizer 9단계(batch 8), 최종 train loss 1.047로 수렴 전이다. 대조: MODEL-REPAIR-001에서는 6행을 40단계 과적합해 loss 0.0003에 이르렀고 EOS에서 멈췄다. 경로는 정상인데 노출이 적어 종료 행동이 약하게 학습됐을 수 있다 |
  | H2 형식 분포가 eval과 다르다 | "결론:" 형식 0행, 해석례 쉬운 설명 3행, 표·서식·코드 조회가 23행이다. 결론 수(22 vs 31)와 형식 차이가 관련 있을 수 있다 |
  | H3 값 계약은 replay가 없다 | value 행 0. 64토큰 한도에서는 학습 답이 잘린다 |

- **판정 방법:** H1~H3을 가르려면 수리본으로 34행 train의 EOS 위치 loss·확률을 재거나, 행 수와 단계만 바꾼 ablation이 필요하다. 둘 다 forward나 학습이 필요하므로 이번에는 하지 않았다.

## 한계

- transformers 4.57.6이 이 tokenizer에 "incorrect regex pattern(fix_mistral_regex)" 경고를 냈다.
  - 학습도 같은 버전과 로더를 썼으므로, 이 감사는 학습이 본 토큰화를 그대로 재현한다.
  - llama.cpp의 프롬프트 토큰화와는 비교하지 않았다. `llama-tokenize`가 빌드되어 있지 않고, 모델 로드도 피했다.
- 템플릿은 문자열 sha만 일치를 확인했다.
- eval 출력과 원문을 보지 않았으므로, eval 수치와 이 감사 사이의 인과는 연결하지 않았다.

SUPERVISION_REPORT_READY TRAIN-CONTRACT-AUDIT-001
