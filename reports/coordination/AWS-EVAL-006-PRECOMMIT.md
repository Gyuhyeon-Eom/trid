# AWS-EVAL-006-PRECOMMIT (초안: A/B 인계 전, 생성 전, 결과 전)

기록: C(01a17c28), 2026-10-31.
- **상태:** 준비 단계다. AWS 자원 0, 추론 0.
- 하드웨어 값(4절 표의 "A 인계" 칸)은 A 인계를 받은 뒤 채우고, **실행 전에** 이 문서 부록으로 커밋한다.

## 1. 새 실행으로 분리
- **AWS-EVAL-006은 CHAT-BLIND-005와 다른 실행이다.** 백엔드가 Metal에서 CUDA로, OS가 macOS에서 Linux로 바뀌고 서버 바이너리도 다르다.
- **005 로컬 결과와 섞지 않는다.** P 235/250턴, Q 0이 그것이다.
  - 하나의 점수로 합치지 않는다.
  - 005 결과는 채점하지 않았고 C도 내용을 읽지 않았다. 그대로 보존한다.
- **질문 세트는 005와 같다.** 비공개 80, sha `af46eec1…`, 250 사용자 턴이다. 그래서 **새 독립 문항이라고 주장하지 않는다.**
  - 이 세트는 C만 봤다. 공개 20 반복에 쓰인 적은 없다.
- **목적은 진단이다.** 운영 합격, 배포, 사이트 모델 변경과는 무관하다.
  - 결과가 어떻든 공개 관문 FAIL은 그대로다. 서비스 경로도 평가하지 않았다.

## 2. 고정 항목 (CHAT-BLIND-005-PRECOMMIT과 같음)

| 항목 | 값 |
|---|---|
| 비교 대상 | Qwen3-1.7B-Q8_0 `061b54da…` (현재 기본 원본) / Gemma 4 E4B QAT q4_0 `676c3507…` (후보) |
| 공통 | system 006 `43a16276…`, max_tokens 384, context 4096, seed 1, `enable_thinking:false`, 공식 렌더링 토큰 일치 필수, raw 모드 |
| sampling | 각 모델에서 이미 확인한 공식 값. 별칭 배정은 C가 모른다. 그래서 **모델 계열과 설정을 함께 비교한다** |
| 기준 | 모델별로 따로: ≥ 225/250턴, ≥ 64/80시나리오, 심각 실패 0. gate.json `4e1d1457…`, 관문 문서 `68dcabd3…`, 집계기 `7a8043c7…` |
| 판정 | 500턴 전체를 응답 순서를 섞은 패킷(`chat_blind_pack.py`)으로 C가 단일 AI 의미 판정. 봉인 커밋 → A 매핑 공개 요청 → commitment 검증 → 모델별 집계 |
| 정정 | `CHAT-EVAL-004/errata.jsonl` 규칙 그대로. 결과 뒤 기준 변경 없음 |

## 3. 비공개 보호
- **원격 checkpoint 암호화:** 대화별 checkpoint는 원격에서 **C 인증서로 CMS 암호화**한 뒤에만 S3에 올린다.
  - 인증서: `AWS-EVAL-006/c-checkpoint-cert.pem`, SHA256 지문 `73:C9:1A:F4:…:FF:06:A8`
  - 개인키는 C 비공개 보관. A/B와 S3 관리자는 평문을 볼 수 없다.
- **비공개 세트 전송** (감독 지시 반영, 앞의 "직접 전송" 문장을 대체한다)
  - **원격으로 보내는 것은 파생 파일 하나뿐이다.** id와 turns만 남긴 `private-80-turns-only.jsonl`, sha256 `4134c4bb66146637be00ff15a4fff78c332cda5138423ea50a646fe49265836c`
    - parent `af46eec1…`, 80행, 250 사용자 턴
    - **정답·기준 필드(expect, track, severe, boundary, external)는 0개다. 비공개 gold는 올리지 않는다.**
  - **전달 방식:**
    1. 인스턴스가 자기 키쌍을 만들고 공개키만 내보낸다.
    2. C가 그 공개키로 CMS 암호화한다.
    3. **암호문만** S3나 명령 매개변수로 전달한다.
    4. 인스턴스 안에서만 복호화한다.
  - **SSM SendCommand 매개변수, 출력, CloudTrail에 평문이나 base64 평문을 넣지 않는다.**
  - 실행이 끝나면 원격 평문과 인스턴스 키를 삭제한다.
  - 로컬 점검: 14.8 KB 암호문에 질문 평문이 없다. 인스턴스 쪽 복호화 결과는 원본과 같다.
- **원격 `sealed/` 파일:** 프롬프트에 질문이 들어 있으므로 같은 방식으로 암호화해 보관한다.
- **한계:** 실행 중 원격 인스턴스에는 평문이 있다. 인스턴스 root 권한자는 접근할 수 있다. 실행 뒤 인스턴스를 종료하고 볼륨을 삭제하는 것으로 노출 시간을 줄인다.

## 4. 하드웨어와 새 실행 차이

| 항목 | 005 (로컬) | 006 (AWS) |
|---|---|---|
| 하드웨어 | 맥북 Apple Silicon 32 GB, Metal | A 인계: 인스턴스 유형, GPU, 드라이버, CUDA |
| 서버 | llama.cpp `8ca4060` Metal `bf5eeac5…` | A 인계: 같은 커밋 `8ca4060` CUDA 빌드 플래그와 바이너리 sha |
| guard | macOS 메모리 압박 단계, 실서비스 running | A 인계: Linux 호스트·GPU 메모리 확인. 실서비스 확인은 해당 없음으로 기록 |
| 실행 위치 | 맥북 직접 | C가 맥미니에서 EC2 백그라운드로 실행. 맥북 연결 불필요 |
| 결과 보관 | 로컬 `runs/c-private` | 대화별 암호화 checkpoint를 S3 비공개 위치에 두고, C가 내려받아 `aws_eval_collect.py`로 복원 |

- **같은 GPU에서 순차 실행한다.** 한 인스턴스에서 P 80대화 다음 Q 80대화, 모두 500턴이다.
- **결정성:** 같은 seed라도 Metal과 CUDA의 부동소수 경로가 달라 출력이 005와 다를 수 있다. 006은 006끼리만 비교한다.

## 5. 중단 처리
- **재시도와 튜닝은 금지다.**
- 중단되면 이미 저장된 대화별 checkpoint는 보존하고, 원인과 개수만 보고한다.
- 이어 실행은 감독이 승인한 경우에만 하고, 대화 경계에서만 한다. 완료된 대화는 다시 생성하지 않는다.
- 비용 한도나 시간 한도(최대 4시간)에 닿아 자동 종료되면, 저장된 checkpoint만으로 미완료를 보고한다.

## 6. 준비된 C 도구 (공개 원시 데이터로 검증, 새 추론 0)
- **`scripts/aws_eval_collect.py`:** 대화별 암호화 checkpoint를 `outputs.jsonl`·`status.json`으로 복원한다.
  - 거부 조건: 키 불일치, 중복, 질문·턴 차이, 백엔드·설정 차이(exit 2)
  - 누락이 있으면 incomplete로 기록하고, 그러면 pack이 거부한다.
  - 정상 복원 결과는 원시 출력과 같다.
- **기존 도구 그대로:** `chat_blind_pack.py`(`282846e3…`), `chat_eval_gate.py`(`7a8043c7…`), `chat_eval_preflight.py`(`63a3cf2b…`)

## 7. 부록: 실행기·계약 해시 (A 통합 수정 `9ae88bf` 뒤, 아직 AWS 실행 전)
- **A 생산 쪽 불일치 3건 수정 완료** (CONTRACT-INTEGRATION-007)
  1. 기대 입력 sha가 파생본 `4134c4bb…`로 바뀌었다. gold가 든 원본 `af46eec1…`은 원격으로 가지 않는다.
  2. checkpoint에 `run_id`(launch RUN_ID, 없으면 runner 거부), `input_sha256`, `effective_settings_constant`가 들어간다.
  3. `.cms.tmp`로 쓴 뒤 rename한다. 종료·실패·기한 때 `finish()`에서 시간 제한을 건 checkpoint 동기화를 한다. 3번은 bootstrap 셸 동작이라 코드 판독으로 확인했고, 실제 AWS 실측은 아직이다.
- **006 실행에 쓰는 파일 sha256** (005 precommit의 runner 값과 다르다)

| 파일 | sha256 |
|---|---|
| `scripts/base_dialogue_runner.py` | `e1a35e447565290fe49368a9b4ab0341ee3c0d6afcb2b5d1d6081772b33b7fda` |
| `scripts/chat_blind_runner.py` | `3cb7390dcd23400d1d23f2f02d8dfaf286f9090bcabc6c30397bfcdfca7c1b4b` |
| `scripts/aws_eval/aws_eval_006.py` | `aafcc211c42ce46e131c3f46166a66187966e7cf21ff2547dd2d31eda2fadbc8` |
| `scripts/aws_eval/bootstrap.sh` | `26056456a20b9b3d7c4dbd68b7b992500501a8cdc6bb2e05c016182cbc5ed6b4` |
| `scripts/aws_eval/c_deliver_set.py` | `ab67ac2a219af376378706a5571a21c26f17e1d7dacd0d4b6370a0476d48f6f6` |
| `scripts/aws_eval_collect.py` (C) | `1d887a9f32b999df573f5a7ffd68326bd4aeddc25f6c3d4df298dd5ae1909ae3` |
| `scripts/chat_blind_pack.py` (C) | `282846e3eac698cf581d417d9fa43b5919804c5a2c584aed60fbd71a8218cc69` |
| `scripts/aws_eval_e2e_check.py` (C 검증 도구) | `57841098dcfb7f7c9af94e56f978679ba9716cff135a36b82583da35f73f9c5a` |

- **005 runner 대비 바뀐 점** (diff 판독)
  - 원격 환경 변수(`TRID_RUNNER_REMOTE`, 서버 경로, `-ngl 999`, 백엔드 이름)
  - Linux 메모리·GPU guard. 실서비스 확인은 해당 없음으로 처리한다.
  - 대화 끝 checkpoint hook
  - **생성 요청은 바뀌지 않았다:** messages 구성, `/completion` 인자, sampling, seed, max_tokens, 렌더 토큰 일치 확인이 같다.
- **C 독립 확인 (CPU, 공개 고정 응답, GPU·AWS 성공 아님)**
  - `.venv/bin/python scripts/aws_eval_e2e_check.py` → exit 0, 7항목 모두 true
    - P/Q 각 63턴 complete와 pack 통과
    - 다른 run은 run id 때문에 거부
    - 설정 변화는 P-011에서 "not constant"로 거부
  - `.venv/bin/python -m unittest tests.test_aws_eval_collect` → Ran 5 OK
  - 감독이 직접 돌린 `bash scripts/aws_eval/fixture_e2e.sh`도 PASS(28.31초)로 보고됐다.
- **아직 아닌 것:** 실제 CUDA 실행, 모델 성능, 종료 실측(기한·실패 때 동기화), 실제 S3 경로. 로그인 뒤 smoke에서 확인한다.
