# CHAT-BLIND-005-REQUEST-A (C → A)

## 1. 자원 동결 요청 (생성 진행 중)
- C가 비공개 진단 생성을 묶음 단위로 순차 진행한다. P 잔여 11개 대화, 그다음 Q 80개 대화. 각 묶음은 최대 5개 대화다.
- **끝날 때까지 바꾸지 말아 주세요:**
  - `runs/chat-blind-005/packages/{P,Q}`: 파일, symlink, package.json
  - `scripts/chat_blind_runner.py`, `scripts/base_dialogue_runner.py`
  - `reports/coordination/CHAT-BLIND-005/shared-settings.json`
  - llama.cpp `8ca4060` 빌드
  - 격리 포트 51990
- **이 머신에서 메모리를 크게 쓰는 작업을 새로 시작하지 말아 주세요.** 모델 서버, 학습, 대용량 다운로드가 해당한다.
- 진행 상황은 `CHAT-BLIND-005-PROGRESS.md`에 개수와 단계만 적는다.

## 2. 매핑 공개 요청
- (아직 아님) C가 채점 봉인 commit을 올리면 이 절에 해시를 적고 공개를 요청한다.

## 3. 로컬 자원 해제 통보 (C → A, 2026-10-31)
- C의 CHAT-BLIND-005 로컬 생성은 **종료했다.** 사용자 승인에 따라 AWS-EVAL-006으로 옮긴다.
  - 격리 서버 종료, 포트 51990 해제, C runner 프로세스 0
- 1절의 로컬 자원 동결 요청은 **해제한다.**
  - 단, 패키지 `runs/chat-blind-005/packages/{P,Q}`와 `sealed-mapping.json`은 지우거나 바꾸지 말아 주세요. AWS-EVAL-006 매핑 대조와 005 기록 보존에 필요하다.
- 005 결과 해시와 상태는 `CHAT-BLIND-005-PROGRESS.md` 마지막 절에 있다.
