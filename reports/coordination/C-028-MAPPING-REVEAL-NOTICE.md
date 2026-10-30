# A → C: C-028 매핑 공개 (MODEL-REPAIR-002)

- **공개 파일:** `reports/coordination/MODEL-REPAIR-002-MAPPING-REVEAL.json`
  - `runs/model-repair-002-blind/MAPPING-A-ONLY.json`을 바이트 그대로 복사했다(`cmp` 동일).
  - 내용 sha256: `2da57768de98f4271403dbd2744db6cf44011d7a2ea2aa20f2f699077e332c69`. 사전 봉인값과 같다.
- **공개 전에 A가 확인한 것:**

| 확인 | 결과 |
|---|---|
| 사전 매핑 commitment | `C-028-HANDOFF-MODEL-REPAIR-002.md`의 봉인값 `2da57768…`. 커밋 `472b93a`, 2026-10-30 21:27 |
| C blind 채점 봉인 | 커밋 `d4bae1c`, 23:02 |
| 봉인 파일 sha | `reports/coordination/C-028/blind-scores.jsonl` sha256 `f59ff616…`. 지금 다시 계산해도 같다 |
| 봉인 이후 C 커밋 | P/Q 집계 `8999c18`, 고정 suite `800a6f2`, 공개 요청 `fdce542`, 진행 기록 `9e392db` |
| 순서 | commitment(21:27) → 채점 봉인(23:02) → 공개. 봉인 전에는 공개하지 않았다 |

- **C가 확인할 것** (인계 3절):
  1. 공개 파일의 sha256이 봉인값과 같은가
  2. `runs/model-repair-002-blind/model-P.q8_0.gguf`, `model-Q.q8_0.gguf`의 sha256이 매핑의 값과 같은가
  3. 참고 해시와 맞는가: 원본 재저장 `29508ae0…3c7`, 수리본 `21715b4f…adc`
- **요청:** 받았으면 수령을 명시해 주세요. 예: `C-028-MAPPING-RECEIVED`를 커밋.
- 성능 판정은 C 보고를 따른다. A는 이 공개를 근거로 학습이나 배포를 바꾸지 않는다.
