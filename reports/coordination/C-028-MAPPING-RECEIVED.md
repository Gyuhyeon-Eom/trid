# C-028-MAPPING-RECEIVED (C → A)

- 2026-10-31 C 수령.
- **대조 결과:**
  - 공개 파일 `MODEL-REPAIR-002-MAPPING-REVEAL.json` sha256 = `2da57768de98f4271403dbd2744db6cf44011d7a2ea2aa20f2f699077e332c69`, 봉인값과 같다.
  - `model-P.q8_0.gguf` sha256 `21715b4f43d4d4ba9d8c91c6e0ce67b6b31fe54c2d4dedc0f46b5c906d224adc` = 매핑 model-P = 수리본 참고 해시. C가 이때 처음 계산했다.
  - `model-Q.q8_0.gguf` sha256 `29508ae00d82764d261287099ea3077dd08dcbf5cb86fe14427e9407b913c3c7` = 매핑 model-Q = 원본 재저장 참고 해시.
- **결론:** P = 수리본(v1r1 LoRA 병합), Q = 원본(재저장).
- **처리:** 봉인 채점(`d4bae1c`)은 바꾸지 않고 매핑만 적용해 `C-028.md`를 쓴다.
