# 공개 가중치 기준선 (2026-10-29)

trid와 같은 평가 자료로 튜닝하지 않은 공개 instruct 모델 두 개를 비교했다. 목적은 "외부 사전학습 가중치를 쓰지 않는다" 원칙을 완화할 때 출발점이 어디인지 재는 것이다. 어떤 모델도 제품 경로에 넣지 않았다.

| 모델 | 출처·revision | 라이선스 | 실행 |
|---|---|---|---|
| Qwen3-1.7B | `Qwen/Qwen3-1.7B-GGUF` Q8_0 (공식 GGUF) | Apache 2.0 | llama.cpp CPU 4스레드, n_ctx 4096 |
| Kanana-1.5-2.1B-instruct | `kakaocorp/kanana-1.5-2.1b-instruct-2505` @ `7df4bc35` → `convert_hf_to_gguf.py --outtype q8_0` | Apache 2.0 | 동일 |

공통: 모델 고유 chat template, greedy, thinking 끔, `scripts/evaluate_external_chat.py`. 모델 파일 SHA256은 각 JSON의 `model_sha256`.

## 결과

### 지정 근거 질의응답 (RAG310, 문서 근거를 프롬프트로 제공)

`grammar_instructed` 조건(“값 [Dn] 형식, 없으면 확인 불가”). 310 = 답 있음 228 + 답 없음·충돌 82.

| | strict (기존 채점) | content_strict | mentions_gold |
|---|---:|---:|---:|
| trid 91M SFT v1, 자유 생성 | 1 | 1 | 6 |
| trid 91M SFT v1, 근거 제한 디코딩 | 3 | 3 | 10 |
| Qwen3-0.6B (기존 출력 재채점) | 2 | 61 | 105 |
| Qwen3-1.7B | 2 | **106** | 171 |
| Kanana-1.5-2.1B | 1 | 26 | **199** |

- `strict`: 기존 `evaluate_rag_eval.score`. `값 [Dn]` 순서만 인정한다.
- `content_strict`: 첫 줄의 `[Dn] 값` 순서도 인정하고 값·출처를 모두 맞혀야 한다. 추가 설명이 붙은 값은 인정하지 않는다.
- `mentions_gold`: 정답 값(답 없음 문항은 `확인 불가`)이 출력 어딘가에 들어 있는지. 독해 여부의 느슨한 지표이며 정답률이 아니다.

기존 README의 “Qwen 정답 0~2개”는 출력 순서(`[D3] 7만 원`) 때문에 생긴 수치였다. 같은 0.6B 출력이 순서만 허용하면 61/310이다.

### 실제 기관 보도자료 probe (40문항, `data/institution-qa-frozen-01`)

| | 답 있음 30: 정답 | 답 없음 10: 보류 |
|---|---:|---:|
| trid reader (cpt-v1/head-v2/data-v4), exact | 14 | 6 |
| trid reader (cpt-v1/head-v2/data-v2), exact | 7 | 9 |
| Qwen3-1.7B, content_strict | 14 (+정답 언급 8) | 0 |
| Kanana-1.5-2.1B, content_strict | 4 (+정답 언급 16) | 0 |

Qwen의 미인정 답 대부분은 단위 누락(`225` vs `225대`)이나 범위 표기다. 두 모델 모두 근거가 없을 때 정해진 `확인 불가`로 멈추지 못했다. 이 probe는 반복 사용한 진단 자료다.

### 공개 문서 대화 (public-dialogue-01과 같은 프롬프트, 256토큰)

필수 사실 기준(`reports/public-dialogue-01-protocol.json`)을 그대로 적용했다. 항목별 판정·경계 사례는 `public-dialogue-assessment.json`, 집계표는 `summary.md`(`scripts/summarize_open_weight_baseline.py`)에 있다. 어시스턴트 검토이며 독립 사람 검수가 아니다.

| | held-out 8 | 회귀 3 (인사·기능 질문·지원 한도) | 128토큰 내 종료 | 반사실 (35%·2천만 원) |
|---|---:|---:|---|---|
| trid public-dialogue-01 | 0 | 2 | — | 30%로 답함 (실패) |
| Qwen3-1.7B | 4 (경계 포함 5) | 2 (기능 질문에 예시 Q&A 생성) | 실패 | 통과 |
| Kanana-1.5-2.1B | 7 (엄격 기준 6) | 3 | 통과 | 통과 |

held-out 실패 예: Qwen은 근거 없는 “예산 기준” 설명을 붙였고, 재난안전특교세라는 지원 종류를 빠뜨렸으며, 용도 질문에 “경기도”라고 답했다. Kanana는 근거에 없는 “오염수 처리 시설 설치”를 덧붙였다. 일상 대화 3문항은 Kanana가 모두 자연스럽게 답했고, Qwen은 “어떤 일을 부탁하면 되니?”를 문장 해석으로 받았다.

## 해석

- 튜닝 전 2B급 공개 모델이 trid의 생성 독해·대화 수준을 크게 넘는다. 근거 값을 문서에서 찾고, 바뀐 수치를 그대로 읽는다.
- 남은 격차는 **출력 형식 준수**와 **답이 없을 때 보류**다. trid 학습 자료(`korean-rag-curriculum-v4` 등)로 LoRA SFT를 하고, 기존 근거 제한 디코딩(`trid/constrained.py`)을 얹어 해결할 영역이다. 아직 실행하지 않았다.
- 성향 차이: Qwen3-1.7B는 형식 지시를 더 잘 따라 짧은 값 답(RAG310 content_strict)이 높다. Kanana는 문서 대화·일상 대화가 가장 좋고 정답 언급도 가장 많지만 장황해 형식 점수가 낮다. 형식은 SFT와 제한 디코딩으로 고칠 수 있으므로, 두 후보 모두 같은 SFT 후 다시 비교해 고른다.

## 한계

- 단일 실행, greedy, Q8_0. 평가 자료 일부는 같은 개발자가 작성했다.
- 공개 모델의 사전학습 데이터는 일부만 공개되어 있다. 가중치 출처 설명 범위는 라이선스와 공개 기술 보고서까지다.
- 서비스 검색·권한·지연은 측정하지 않았다. CPU 4스레드 기준 문항당 약 1~4초(프롬프트 200~800토큰, 출력 ≤64토큰)였다.

## 재현

```sh
llama-server -m <model.gguf> --host 127.0.0.1 --port 18473 -c 4096 -np 1 -t 4 -tb 8 --jinja
.venv/bin/python scripts/evaluate_external_chat.py --model-file <model.gguf> --model-id <id> --revision <rev> \
  --family qwen3|generic --data data/korean-rag-eval-v1 --output <report.json> [--compare-report <old.json> ...]
.venv/bin/python scripts/evaluate_external_chat.py ... --data data/institution-qa-frozen-01 --output <report.json>
.venv/bin/python scripts/evaluate_external_chat.py ... --freeform-prompts reports/public-dialogue-01-after.json --output <report.json>
```

반사실 프롬프트는 `korea-kr-press-156612711` 회귀 프롬프트의 `70%`→`35%`, `4천만 원`→`2천만 원` 치환이며, 원시 프롬프트는 `counterfactual-*.json`에 저장되어 있다.
