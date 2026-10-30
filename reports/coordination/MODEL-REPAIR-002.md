# MODEL-REPAIR-002 (A) — 계약을 맞춘 데이터로 원본 Qwen3-1.7B 로컬 재학습 1회

- **작성:** A (01a18739), 2026-10-30
- **상태:** 부분 완료.
  - **완료:** 학습 1회, 병합, GGUF 변환, 원본과 같은 변환 경로 확인, blind 패키지, 로컬 실모델 스모크, 실험판 게시와 실 UI 확인
  - **대기:** C-028 평가
  - **성능 향상:** **미확인**
- **비용:**
  - 맥북 Apple GPU(MPS)에서 학습 513초. 1800초 한도 안이었고 재시도는 0회다.
  - 추가 AWS job과 유료 비용은 0원이다.
  - 실행 전 메모리 여유는 66%였다. 서빙 앱과 메모리 압박은 없었다.

## 1. 입력과 설정 (이전 실험판과 같은 설정, 형식·데이터만 바뀜)
- **데이터:** B가 승인한 고정본 `data/case-training-view-v1r1/a-train-manifest.json`(sha `5a8368e5…`). B 보고 `MODEL-REPAIR-002-B.md` READY.
  - **train:** 34행 × 2 epoch = 68행. sha `e4eb5f67…d1`. 문장별 AI 검토 결과 pass 28 / fix 6 / hold 0.
  - **dev:** 6행. sha `91774ff5…a7b`. 검증 손실만 기록했다.
- **v2 117행 후보(`fc1957df…`)는 섞지 않았다.** 이어서 자동 학습하지도 않는다.
- **설정:**
  - 9단계, batch 8(micro 1), **lr 2e-4**, warmup 5% + cosine
  - LoRA r16·α32·dropout 0.05(q,k,v,o,gate,up,down), bf16
  - seed 20261030, max_tokens 6144, 답 토큰에만 손실
  - **pilot-sft-001과 같은 설정**이다. lr이나 단계 규칙을 바꾸지 않았다.
- **계약 검사:** 학습기가 case 계약을 검사했다(위반 0, 버린 행 0, 답 최대 252토큰).
- **도구:** torch 2.9.1 / transformers 4.57.6 / peft 0.17.1, 맥북 MPS.
- **손실:** train 2.654 → 1.047 (9단계).
  - 검증 손실(기록만): code_unit_lookup 1.123, statute_application 1.205
  - **dev 6행으로 품질을 판정하지 않는다.**

## 2. 산출물 (`runs/`, Git 미포함)

| 항목 | 경로 | sha256 |
|---|---|---|
| adapter | `runs/model-repair-002-qwen3-1p7b/train/adapter/adapter_model.safetensors` | `4c842aa1fa1933543713e19826a11016bd82560c7cfaec01f3447fced6fa7b28` |
| 병합 가중치 | `…/train/merged/model.safetensors` | `4aeea2f0f7b5780ce2dd20988a5004c24f8d61fefae45220c86b062d38a19a21` |
| 학습 기록 | `…/train/training.json`, `train.log` | `bb3cb8c3…` |
| 수리본 GGUF (Q8_0) | `…/model-repair-002-qwen3-1p7b.q8_0.gguf` | `21715b4f43d4d4ba9d8c91c6e0ce67b6b31fe54c2d4dedc0f46b5c906d224adc` |
| 원본 GGUF (같은 경로로 재저장·변환) | `…/qwen3-1.7b-base-resaved.q8_0.gguf` | `29508ae00d82764d261287099ea3077dd08dcbf5cb86fe14427e9407b913c3c7` |

- **tokenizer와 template:** tokenizer.json `aeb13307…`, chat_template `a55ee1b1…`. 원본·pilot과 같다.
- **변환 일치 확인:**
  - llama.cpp `8ca4060`의 `convert_hf_to_gguf.py`를 소스에서 받아 썼다(`runs/tools/llama.cpp-8ca4060-src`).
  - 원본 HF를 이 변환기로 다시 변환하니 기존 localconv `f2ba848c…`와 **텐서 311개의 바이트가 모두 같았다.** 차이는 디렉터리 이름에서 오는 메타데이터 2개뿐이었다.
  - 병합 저장(transformers 4.57)은 묶인 `lm_head`를 따로 저장하지 않는다(`output.weight` 없음, 1.83GB). 그래서 원본도 같은 저장 경로로 다시 저장·변환했다.
  - 그 결과 수리본과 크기·텐서 이름·이름 메타데이터가 같아졌다. 재저장 원본은 localconv와 텐서가 같다. localconv의 `output.weight`는 `token_embd`와 같다.

## 3. 로컬 실모델 스모크 (llama.cpp Metal, 서빙 경로 그대로, 자료 정보 줄 포함)
- **질문:** 공개 색인으로 A가 고른 질문 4개다. 평가 문항이 아니다.
- **관찰 (품질 판정 아님):**
  - 출처 번호 `[Dn]`를 내기 시작했다.
  - 그러나 3건이 256토큰에서 잘렸다. 원인은 `[D1]` 반복, `자료 정보` 줄 복사, 근거 긴 복사다.
  - 1건은 지시 복사로 판정돼 표시하지 않았다(model_failure).
  - 34행 학습으로는 형식 정착이 불완전하다.

## 4. blind 평가 인계 (C-028)
- 패키지: `runs/model-repair-002-blind/model-P|Q.q8_0.gguf`
- 매핑은 A만 가진다. 봉인값 sha256 `2da57768…2c69`.
- 요청과 공개 절차: `reports/coordination/C-028-HANDOFF-MODEL-REPAIR-002.md`
  - case 계약 matched 평가와 값 계약 회귀를 따로 본다.
  - C가 채점을 커밋한 뒤 매핑을 공개한다.

## 5. 게시 (사용자 지시: 실험 선택지로만, 기본은 원본)
- **사이트:** Worker `d6feb400-52f9-4a2f-9b91-790a5de5f023`(21:24 KST)
- **버전 정보:**
  - 버전 `trid-qwen3-1.7b-model-repair-002`, 표시명 "TRID 1.7B · 수리 실험판 (평가 완료 · 원본 대비 회귀 · 사용 비권장)"(2026-10-31 C-028 결과 반영. 처음 게시 때는 "성능 미확인")
  - **C-028 결과(`reports/coordination/C-028.md`):** 원본 대비 회귀다. 개선 근거는 없다.
    - case 결론 정확: 31 → 22
    - 256토큰 안 EOS 종료: 32/50 → 3/50
    - RAG310 grammar_instructed: 98 → 14
    - evidence_constrained 과보류: 128/228
    - 모델 파일은 지우지 않는다. 기본값은 원본 그대로다. 추가 학습도 하지 않는다.
  - 포트 51918, `provenance_line: true`(학습 입력과 같은 `자료 정보:` 줄)
- **기본값:** 원본(`trid-assistant-qwen3-1.7b`) 그대로다.
- **실 UI 확인 (`RETRIEVAL-DECISION-001-live.json`의 `repair`):**
  - 선택 → "상시 출장하는 공무원 여비는 어떻게 지급돼?"
  - 공무원 여비 규정 별표 8 적용, 해설만(검증 안 됨), 근거 없는 숫자 문장 1개 숨김, 256토큰 잘림
  - 카드에 "성능 향상 미확인 · C 평가 전 · 사용 추천 안 함"을 표시했다.

## 6. 해석 한계
- 이전 실험판 대비 바뀐 것은 데이터 형식(출처 번호, 길이), 데이터 부분집합(57 → 34행, 공식 회답 복사 행 제외), 데이터 크기다. 그러므로 **회귀 원인을 확정하는 ablation이 아니다.**
- 회귀가 사라졌는지와 개선됐는지는 C-028 결과로만 말한다.

SUPERVISION_REPORT_READY MODEL-REPAIR-002
