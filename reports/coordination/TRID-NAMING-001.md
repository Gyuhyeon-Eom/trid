# TRID-NAMING-001: 특화 pilot 표시명 "TRID 1.7B · 실험판"

- **작성:** A (01a18739)
- **사용자 승인 반영:** 2026-10-30 15:19 KST 게시. 이후 15:36 재게시에도 유지했다(Worker `f3ad7b2c…`).

## 바꾼 것
- **선택창과 응답 표시명:** `trid Qwen3-1.7B pilot-sft-001 (실험판, 평가 중)` → **`TRID 1.7B · 실험판`** (`deploy/shared-demo/versions.json`의 `label`)
- **버전 상세는 유지했다:**
  - `pilot-sft-001 · 평가 중`
  - 학습 이력: C017 57건 × 2 epoch, 15단계
  - 기반: `Qwen/Qwen3-1.7B` HF rev `70d244cc…` + LoRA 병합, Alibaba Qwen 원 모델 기반, "독자 기초모델 아님"
  - 라이선스: Apache-2.0
  - 실험 한계, 서빙 조건
- **README 현재 상태 표 (커밋 `10fdd76`):**
  - "실험 모델 이름" 행을 추가했다: TRID 1.7B = Qwen3-1.7B + LoRA. 기반 모델과 라이선스를 공개하고 독자 기초모델이 아니라고 밝혔다.
  - 테스트 서비스 행을 실제 상태로 갱신했다.

## 바꾸지 않은 것
- **미특화 원본:** `기준선: Qwen3-1.7B 원본 (미특화, trid 아님)` 그대로다. TRID로 리브랜딩하지 않았다.
- **내부 식별자:** artifact id `trid-qwen3-1.7b-pilot-sft-001`, 파일명, 해시, API 계약은 바꾸지 않았다.

## 확인 (실제 브라우저, 헤드리스 Chromium, 공개 HTTPS)
- **선택창:** `['trid v0.1 추출 독해기', '기준선: Qwen3-1.7B 원본 (미특화, trid 아님)', 'TRID 1.7B · 실험판']`
- **상세 카드:** "TRID 1.7B · 실험판 · pilot-sft-001 · 평가 중 …", 기반·라이선스·실측("아직 없음")·한계가 표시된다.
- **응답 표시명:** `trid [TRID 1.7B · 실험판] > …`

## C 통지
- 이 보고와 TRID-WORKSPACE-001 6절이 C 통지다. C022 이후 실 UI 확인을 요청한다.

SUPERVISION_REPORT_READY TRID-NAMING-001
