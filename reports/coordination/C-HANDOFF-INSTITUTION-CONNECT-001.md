# C 통지: INSTITUTION-CONNECT-001 실배포 (A → C)

- **배포:** 2026-10-30 22:48 KST에 시작했다. Worker는 `08f77f3c-3b27-4441-a3c7-78dd6876bf4e`이고, 2026-10-31 22:48 KST까지 24시간 운영한다.
- **색인:**
  - 공유 색인: `runs/workspace-index/v5` (manifest `1c9540ca…`, 변경 없음)
  - 한국부동산원 별도 색인: `runs/workspace-index/inst-reb-002` (manifest `f2239545…`, holds evaluation/training)
- **고정 평가 조건:**
  - C-027 고정 색인 v3(`8afc0384…`)는 바꾸지 않았다.
  - 서빙 코드는 바뀌었다. 같은 조건으로 재현하려면 v3와 코드 `242dbf2`를 써야 한다.
  - C-028 blind 패키지와 매핑은 건드리지 않았다.
- **이번 코드에 들어간 것** (평가에 영향을 줄 수 있음):
  - 별도 기관 색인 연결 (ALIO C0150일 때만)
  - 집행내역·정형 값 처리 (직원 평균 질문은 계산하지 않고 거절, 직위를 지정한 질문만 값 조회)
  - 별표 번호를 원문 캡션에서 판정 (6의2와 6을 구분)
  - B가 AI 시각 대조로 검토한 별표 행만 파생 근거로 사용 (인용으로는 표시하지 않음)
  - 근거 3,600자 한도
  - C-029 보호 계약 (보호 판정 범위, fail-closed, value_hold 안내, 원출력 차단)
  - 적용 대상 경고를 답변 본문에 표시
- **C-029 2절 재검증 대상:**
  - privacy·hold 경로는 실제 `/api/ask` 경로에서 case·value·reader 세 버전 모두 테스트로 확인했다. `tests/test_shared_demo.py`의 `ProtectedPassageApiTests`다.
  - 실사이트 재검증은 C가 맡는다.
- **성능 향상은 주장하지 않는다.** A가 관찰한 오답과 실패 사례는 `INSTITUTION-CONNECT-001.md` 5절과 남은 실패 절에 있다.

SUPERVISION_REPORT_READY C-HANDOFF-INSTITUTION-CONNECT-001
