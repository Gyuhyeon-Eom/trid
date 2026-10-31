# AWS-EVAL-006-REQUEST-AB (C → A 인프라 담당, B 비용 검증)

C가 실행을 시작하려면 아래 계약이 확인돼야 한다.
- **비공개 문항과 답은 A/B에게 보내지 않는다.**
- A/B도 비공개 세트, checkpoint 평문, `sealed/`를 열지 않는다.

## A (인프라)
1. **비용·자원 (B 검증 근거 첨부)**
   - 최대 비용 min(계정 잔액, 3만 원), GPU 1대, 최대 4시간
   - 자동 종료: 인스턴스 자체 종료 예약과 종료 시 terminate 동작을 둘 다 둔다.
   - 학습 0, 사이트 모델 변경 0
2. **실행 환경 기록:** 인스턴스 유형, 리전, AMI, GPU, 드라이버, CUDA
   - llama.cpp는 **같은 커밋 `8ca4060`**의 CUDA 빌드다. 빌드 플래그와 바이너리 sha를 적는다.
3. **익명 패키지 P/Q**
   - 005와 같은 package.json(P `14c8dce1…`, Q `c381396d…`)과 매핑 commitment `bc582558…`을 그대로 쓰는지 명시한다. 새로 만들면 새 commitment를 C 실행 전에 커밋한다.
   - 원격 모델 파일 sha는 실행기가 확인한다(불일치면 거부).
   - **완전 블라인드가 아니다.** 패키지 내용, 출력 문체, 속도로 계열을 추정할 수 있다. 이 한계를 계약에 명시한다.
4. **실행기 (원격판)**
   - `base_dialogue_runner.run()` 로직을 그대로 쓴다: 대화별 history, `<NEW_CHAT>`, 렌더 토큰 일치 확인, 재시도·필터 없음.
   - 공통 설정: `shared-settings.json`과 같은 값(006 system, 384, 4096, seed 1, `enable_thinking:false`)
   - **출력 마스크는 같다.** C가 채점 전에 보는 것은 출력과 상태뿐이다.
   - 대화가 끝날 때마다 **C 인증서 `AWS-EVAL-006/c-checkpoint-cert.pem`으로 CMS 암호화**한 checkpoint를 S3에 둔다. 형식은 `scripts/aws_eval_collect.py` 상단 계약을 따른다.
     - 필드: alias, dialogue, rows, status(package, shared_settings_sha256, backend, render_parity_token_ids_equal, effective_settings_sha256)
   - 원격 `sealed/`(프롬프트, 토큰, 로그)도 같은 인증서로 암호화해서만 보관한다.
   - Linux guard: 호스트·GPU 메모리를 확인한다. 부족하면 대화 경계에서 멈추고 재시도하지 않는다.
5. **S3:** 비공개 버킷, 공개 차단, 서버측 암호화를 쓴다. C 실행분 prefix는 하나만 쓴다.
6. **CUDA 스모크:** 평가 문항이 아닌 공개 2턴 대화로 P와 Q 둘 다 실행한다.
   - 기준: complete, 렌더 토큰 일치 true, 실효 설정 고정, checkpoint 1개 이상이 암호화돼 S3에 올라가고 C 쪽 복원 확인이 가능할 것
7. **원격 실행 경로 인계:** 맥미니에서 C가 실행하는 명령, 백그라운드 방식, checkpoint 위치와 내려받는 방법, 종료·삭제 절차
   - 비공개 세트는 C가 실행 호스트에 직접 전송하고, 끝나면 삭제한다. S3에는 올리지 않는다.

## B (비용 검증)
- 계정 잔액과 예상 비용을 확인한다. 인스턴스 유형 단가 × 최대 4시간 + 스토리지 + 전송.
- 한도 min(잔액, 3만 원) 안에 드는지 확인한다.
- 자동 종료 설정이 실제로 걸려 있는지 확인한다.

## 매핑 공개 (나중)
- C가 500턴 채점 봉인 commit을 올리면, 이 문서 끝에 해시를 적고 A에게 매핑 공개를 바로 요청한다.
