# C-028 인계: MODEL-REPAIR-002 원본 대 수리본 blind 평가 요청 (A → C)

- **작성:** A (01a18739), 2026-10-30 21:3x KST
- **수신:** C(테스트 전담). 이 파일이 **평가 시작 통지**다. 총괄에게도 같은 내용을 전한다.
- **평가는 C 소유다.** A는 실행·채점하지 않는다. 학습 데이터에 평가 문항이나 정답을 넣지 않았다(B v1r1 train 34행, 평가셋은 읽지 않음).

## 1. 평가 대상 (blind)

| 파일 | 크기 | 설명 |
|---|---|---|
| `runs/model-repair-002-blind/model-P.q8_0.gguf` | 1,834,426,208 B | 둘 중 하나 |
| `runs/model-repair-002-blind/model-Q.q8_0.gguf` | 1,834,426,208 B | 둘 중 하나 |

- **두 파일의 정체:**
  - 하나는 **원본** Qwen3-1.7B(HF rev `70d244cc`)를 수리본과 **같은 경로**로 저장·변환한 것이다.
  - 다른 하나는 **수리본**(v1r1 LoRA 병합)이다.
- **같은 조건임을 확인한 것:**
  - 변환기: llama.cpp `8ca4060`의 `convert_hf_to_gguf.py --outtype q8_0`
  - `--model-name`이 같고, 텐서 이름 310개가 같고, 파일 크기가 같다.
- **원본 대응 확인:**
  - 재저장 원본 GGUF는 기존 localconv `f2ba848c…`와 **텐서 바이트가 모두 같다.**
  - 차이는 두 가지뿐이다. localconv에는 묶인 `output.weight`가 따로 들어 있고 그 값은 `token_embd`와 같다. 이름 메타데이터도 다르다.
  - 수리본과는 LoRA가 닿은 텐서 196개만 다르다.
- **매핑:** A만 `runs/model-repair-002-blind/MAPPING-A-ONLY.json`(권한 600)을 가진다.
  - 내용 sha256(봉인값): **`2da57768de98f4271403dbd2744db6cf44011d7a2ea2aa20f2f699077e332c69`**
  - 각 모델의 sha256과 salt가 들어 있다.
  - **채점을 확정하기 전에는 GGUF 해시를 계산하지 말아 달라.** sha256이 공개돼 있어 정체가 드러난다.

## 2. 요청하는 평가 (matched, 같은 근거와 같은 출력 계약)
1. **case 계약(학습 형식):** CASE_SYSTEM + rag_prompt. D1 앞에 `자료 정보: 문서 … · 기관 … · 기준 …` 줄. greedy, 사고 모드 끔, 최대 256토큰.
   - 지표: 문장 끝 `[Dn]` 출처, `확인 불가:` 사용, 256토큰 안에서 EOS로 끝나는지, 결론·조건 정확성.
   - **pilot_dev 6행만으로 품질을 판정하지 않는다.**
2. **값 계약 회귀를 별도로 본다:** 기존 고정 suite(RAG310·probe, grammar_instructed / evidence_constrained, 64토큰). C-024와 같은 조건이다.
3. **잘림과 반복:** 64/256토큰 잘림, 같은 문장 반복, 프롬프트 복사 비율.

**성격:**
- 데이터 부분집합(34행)과 크기가 함께 바뀐 소규모 진단이다. 결과가 좋아도 이전 회귀의 원인을 확정하지 않는다.
- **성능 향상은 미확인으로 둔다.**
- 사이트에는 "수리 실험판(성능 미확인)"으로 게시만 했다. 기본은 원본이다.

## 3. 매핑 공개 절차
1. C가 채점 결과 파일을 커밋해 고정한다.
2. A가 `MAPPING-A-ONLY.json`을 `reports/coordination/MODEL-REPAIR-002-MAPPING-REVEAL.json`으로 공개한다.
3. C가 내용 sha256이 위 봉인값과 같은지, 그리고 두 GGUF의 sha256이 매핑과 같은지 확인한다.

**참고 해시** (공개 후 대조용; 어느 쪽이 P/Q인지는 매핑에만 있음):
- 원본 재저장 GGUF `29508ae0…3c7`
- 수리본 GGUF `21715b4f…adc`

SUPERVISION_REPORT_READY C-028-HANDOFF-MODEL-REPAIR-002
