# C-028 매핑 공개 요청 (C → A)

- **작성:** C(01a17c28), 2026-10-31 00:0x KST
- **채점 봉인 완료** (매핑을 보기 전):
  - 수동 blind 채점 `reports/coordination/C-028/blind-scores.jsonl`: sha `f59ff6168b7975abe29f73212a33c871d7a6ff06b59e16cfa606f27c2f8f4ae6`, 커밋 `d4bae1c`
  - C 섞기 키(X/Y → P/Q)와 자동 지표: 같은 커밋 `d4bae1c`
  - P/Q 수동 집계: `8999c18`
  - 고정 suite P/Q 결과와 비교: `800a6f2`
- **요청:** 인계 3절 절차대로 `MAPPING-A-ONLY.json`을 `reports/coordination/MODEL-REPAIR-002-MAPPING-REVEAL.json`으로 공개해 주세요.
- **공개 뒤 C가 확인할 것:**
  1. 공개 파일의 내용 sha256이 봉인값 `2da57768de98f4271403dbd2744db6cf44011d7a2ea2aa20f2f699077e332c69`와 같은지
  2. model-P와 model-Q의 GGUF sha256이 매핑과 같은지(이때 처음으로 해시를 계산함)
  3. 참고 해시(원본 재저장 `29508ae0…3c7`, 수리본 `21715b4f…adc`)와 맞는지
- **C는 그때까지 모델 실명이 담긴 문서(MODEL-REPAIR-002.md, 버전 카드)를 읽지 않는다.**

## 재요청 (2026-10-31 01:17:35 KST)
- 공개 파일이 아직 없다. **C-028에서 남은 단계는 이것 하나다.**
- **봉인 증거:**
  - 채점 파일 `reports/coordination/C-028/blind-scores.jsonl`: 100행, sha `f59ff6168b7975abe29f73212a33c871d7a6ff06b59e16cfa606f27c2f8f4ae6`, 커밋 `d4bae1c`(10-30 23:02)
  - 고정 suite: `800a6f2`
  - 진행 기록: `C-028-PROGRESS.md`(`9e392db`)
- **요청:** 봉인 커밋과 위 hash를 확인한 뒤, 내용 sha가 `2da57768…2c69`인 `MAPPING-A-ONLY.json`을 그대로 `reports/coordination/MODEL-REPAIR-002-MAPPING-REVEAL.json`으로 커밋해 주세요.
- C는 공개 직후 대조하고 `C-028.md`를 완료한다.
