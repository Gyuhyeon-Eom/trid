# BASELINE-COMPARE-001-ACCESS: 맥미니 접속 확인과 읽기 전용 조사

**상태:** 접속 확인 완료, 비교 대기.
- 맥미니에 접속해 읽기 전용으로 조사했다.
- 비교 실행(모델 load·생성)은 C-031 동결 해제까지 하지 않는다.
- 다운로드, 설치, 유료 사용, AWS, 학습은 0이다. 기본 모델도 바꾸지 않는다.
- **정정:** BASELINE-COMPARE-001의 "접속 경로 없음" 판정은 틀렸다.
  - 근거로 든 `.local` DNS 실패와 mDNS 무응답은 맞다.
  - 하지만 이미 받은 IP 경로를 시도하지 않고 내린 판정이었다.

## 1. 접속 시도별 결과 (2026-10-31 01:4x KST)

| 경로 | 명령 조건 | 결과 |
|---|---|---|
| DNS `eomgyuhyeon-ui-Macmini.local` | `ssh -o BatchMode=yes -o ConnectTimeout=8 -i ~/.ssh/trid-macmini-ed25519` | `ssh: Could not resolve hostname eomgyuhyeon-ui-macmini.local: nodename nor servname provided, or not known` (mDNS `dns-sd -G v4` 5초도 응답 없음) |
| IP `211.48.84.156` | 같은 키, `BatchMode=yes`, `ConnectTimeout=8`, `StrictHostKeyChecking=yes` | **성공.** 원격 `hostname` = `eomgyuhyeon-ui-Macmini.local`, macOS 26.3.1 |

- **호스트 신원:**
  - `known_hosts`에 211.48.84.156 키가 3개 있다(ed25519, rsa, ecdsa, 2~4행).
  - `StrictHostKeyChecking=yes`로 대조해 통과했다. 키가 바뀌었다면 ssh가 연결을 거부했을 것이다.
  - `ssh-keyscan`으로 키를 추가하지 않았다. 보안 설정도 바꾸지 않았다.
- **조회 방법:**
  - 원격에서 실행한 것은 `sysctl`, `vm_stat`, `uptime`, `df`, `ps`, `launchctl list`, `ollama --version`, 그리고 `curl http://127.0.0.1:11434/api/tags|ps|show`뿐이다.
  - 포트 포워딩이나 외부 노출은 하지 않았다.
  - Ollama는 `127.0.0.1:11434`에서만 대기한다(pid 10658).

## 2. 맥미니 인벤토리

| 항목 | 값 |
|---|---|
| 하드웨어 | Mac16,10, Apple M4, 메모리 16 GB, 디스크 460 GB 중 64 GB 여유 |
| 부하 | uptime 38일, load average 7.24 / 6.41 / 6.15(조회 시점) |
| 메모리 | memory_pressure 기준 시스템 여유 67%. RSS 상위: Docker backend 1.8 GB, Virtualization VM 0.8 GB, openjdk 0.7 GB, node 0.3 GB |
| 운영 서비스(launchd) | `co.tridatum.secho.tunnel`(실행 중, pid 10092), `co.tridatum.rag.refresh`(마지막 종료 코드 1). Docker와 Java도 실행 중이다 |
| Ollama | 0.40.3. 로드된 모델 없음(`/api/ps` 빈 목록) |

**Ollama 모델 목록**

| 모델 | 파라미터 | 양자화 | 크기 | digest |
|---|---|---|---|---|
| **`qwen2.5:7b`** | 7.6B | Q4_K_M | 4.7 GB | `845dbda0ea48` |
| `exaone3.5:7.8b` | 7.8B | Q4_K_M | 4.8 GB | `c7c4e3d1ca22` |
| `gemma3:4b` | 3.9B | Q4_K_M | 3.4 GB | `68ceab7e468a` |
| `bge-m3` | 567M | F16 | 1.2 GB | `790764642607` |

**`qwen2.5:7b` 상세**(`/api/show`, 로드 없음):
- 라이선스: Apache-2.0
- context: 32,768
- family: qwen2
- template: ChatML(`<|im_start|>`). 원문 sha256 앞 16자는 `eb4402837c7829a6`, 길이 1,482자다.
- Modelfile `parameters`: **없음.** 그대로 쓰면 Ollama 기본 샘플링이 적용된다. 그래서 비교할 때 Qwen2.5 공식 권장값을 요청마다 명시해야 한다.
  - 권장값: temperature 0.7, top_p 0.8, top_k 20, repeat_penalty 1.05
  - `[INFERENCE: 공개 generation_config 기준, 실행 전 재확인]`

## 3. 실행 가능성 판단과 조건

- **메모리:** `qwen2.5:7b` Q4_K_M(4.7 GB)을 한 번에 하나만 올리면, 현재 여유 67%에서 들어갈 것으로 본다 `[INFERENCE]`. 아직 load하지 않았다.
- **운영 영향:** 맥미니는 운영 서비스(secho 터널, rag, Docker)를 돌리고 있고 부하도 약 6~7이다. 그래서 비교는 다음 조건에서만 한다.
  - 순차 실행, 동시 요청 1
  - `keep_alive`를 짧게 설정
  - 운영 서비스를 내리거나 바꾸지 않는다
  - 실행 전후로 load, 메모리, 서비스 상태를 기록하고, 이상이 보이면 중단한다
- **지연 측정:** 맥미니(M4, 부하 있음)와 맥북(M5) 사이 지연은 하드웨어와 부하가 달라 직접 비교할 수 없다. 각 모델의 지연은 해당 장비 조건과 함께 따로 기록한다.

## 4. 다음 단계 (C-031 "FREEZE RELEASED" 뒤)

- `BASELINE-COMPARE-001.md` 3절 계획대로 진행한다.
  - Qwen3-1.7B: 맥북에서 격리 llama-server로 실행한다.
  - `qwen2.5:7b`: 맥미니 Ollama에 `/api/chat`을 SSH 안에서 localhost로 호출한다.
  - 같은 개발 대화, 같은 시스템 프롬프트, 같은 출력 한도(384, `num_predict`)를 쓴다. 대화마다 세션을 따로 쓴다.
  - 각 모델의 공식 template과 권장 설정을 기록한다.
- 결과는 `BASELINE-COMPARE-001.md`의 후속 보고로 연결한다.
- 세대와 크기가 함께 다르므로 파라미터 수의 인과로 해석하지 않는다. 어느 모델도 자동으로 기본값이 되지 않는다.

SUPERVISION_REPORT_READY BASELINE-COMPARE-001-ACCESS
