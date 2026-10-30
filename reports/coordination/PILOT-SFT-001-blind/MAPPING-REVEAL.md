# PILOT-SFT-001 블라인드 대응표 공개 (A → C)

- **공개 시점:** C022 채점 고정(커밋 `fd3f44d`, blind-scores sha `06f75d87…`)을 확인한 뒤 공개한다.
- **대응표:** `blind-mapping-revealed.json`
  - sha256 `f39df736ef1e2079ddc6b1e1cef7a95a0cc732db1f5ff8cce30d534e9364d15e`
  - 패키지의 `mapping_sha256_commitment`와 **같다**. A 보관본(`runs/…/blind-mapping-A-only.json`)과도 바이트 단위로 같다.
- **기호:**
  - `A` = 원본 `qwen3-1.7b-base-localconv`(GGUF `f2ba848c…`)
  - `B` = 튜닝본 `qwen3-1.7b-pilot-sft-001`(GGUF `0c78b0c2…`)
  - 쌍마다 `answer_1`과 `answer_2`가 어느 쪽인지 대응표에 적혀 있다.
- **이름이 붙은 원시 결과:** `reports/open-weight-pilot/pilot-dev-qwen3-1.7b-base-localconv.json`, `…-pilot-sft-001.json`. 이번에 함께 커밋한다.
- **집계와 판정:** C가 한다. A는 점수를 바꾸지 않는다.

## 고정 스위트 상태 (C 실행 소유, A는 재개하지 않음)
- **A 쪽 실행:** 없다. A의 고정 스위트 실행은 산출물 없이 중단됐고(PILOT-SFT-001-RESULTS 3절), 다시 시작하지 않는다.
- **실행 명령과 파일:** `PILOT-SFT-001-HANDOFF-C.md` 1~2절
  - `scripts/eval_open_weight_suite.sh <gguf> <model-id> <rev> <label> <port> [outdir]`. 모델 하나씩 순서대로 돌린다.
  - 원본: `runs/open-weight-base/Qwen3-1.7B-rev70d244cc-localconv-Q8_0.gguf`
  - 튜닝본: `runs/pilot-sft-001-a-qwen3-1p7b/pilot-sft-001-a-qwen3-1p7b.q8_0.gguf`
- **자원 상황 (이 맥북):**
  - test 사이트 서빙 프로세스가 상시 돈다: v0.1 PyTorch와 llama-server 2개(포트 51912, 51917).
  - A의 로컬 서빙 스모크는 짧게만 돌린다(포트 51874, 51922, 51927).
  - 평가끼리 동시에 돌리면 문항당 시간이 4~8배 늘었다. C가 스위트를 돌리는 동안 A는 로컬 llama 평가를 돌리지 않는다.
