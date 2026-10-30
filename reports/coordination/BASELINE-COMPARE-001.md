# BASELINE-COMPARE-001: 기존 로컬 원본 모델 조사와 비교 준비

**상태: 차단됨.**
- 지정 비교 대상 `qwen2.5:7b`는 맥미니에만 있다. 그 맥미니에 지금 접속할 수 없다.
- 비교 실행 0회. 다운로드, 설치, 유료 사용, AWS, 학습 모두 0이다.
- C-031 동결 중(01:38:19 시작)이라 생성 실행도 하지 않았다.
- 기본 모델 교체는 없다.
- 이번 조사는 모두 읽기 전용이다: `ollama list`·`show`·`ps`, `sysctl`, `vm_stat`, `df`. 어떤 모델도 메모리에 올리지 않았다.

## 1. 인벤토리

### 맥북 (Mac17,2, Apple M5, 메모리 32 GB, 디스크 여유 352 GB)

| 모델 | 위치 | 크기·양자화 | 라이선스 | 상태 |
|---|---|---|---|---|
| Qwen3-1.7B 원본 Q8_0 | `runs/open-weight-base/Qwen3-1.7B-Q8_0.gguf` | 1.7B · Q8_0 | Apache-2.0 | 실서비스 기본. llama-server :51912 실행 중(RSS 약 4.1 GB) |
| `qwen3:30b-a3b` | Ollama | 30.5B MoE(활성 약 3B) · Q4_K_M · 18 GB | Apache-2.0 | 설치됨, 미로딩 |
| `gemma3:12b` | Ollama | 12.2B · Q4_K_M · 8.1 GB | Gemma Terms of Use | 설치됨, 미로딩 |
| `exaone3.5:7.8b` | Ollama | 7.8B · Q4_K_M · 4.8 GB | EXAONE AI Model License 1.1 – **NC(비상업)** | 설치됨, 미로딩 |
| `bge-m3` | Ollama | 임베딩 | — | 생성 모델 아님 |
| **`qwen2.5:7b`** | — | — | — | **맥북에 없다** |

- **메모리:** 조사 시점 vm_stat 기준으로 wired 약 6.5 GB, active 약 9.2 GB, inactive 약 9.2 GB, free 약 0.5 GB였다.
- 실서비스 llama-server 세 개가 떠 있다. 1.7B GGUF 두 개는 mmap 공유다.
- 7~12B Q4 모델 하나는 실서비스를 내리지 않고도 들어갈 여지가 있어 보인다 `[INFERENCE]`. 실제로 올려 보지 않았다.
- `qwen3:30b-a3b`(18 GB)는 실서비스와 함께 쓰면 메모리 압박이 크다고 본다 `[INFERENCE]`.

### 맥미니

- `TRID-BASE-CUTOVER-001.md`(38·91행)에 "맥미니 Ollama `qwen2.5:7b`는 품질 상한을 보는 참고군"이라는 기록이 있다.
- **지금 접근 불가.**
  - `ssh -i ~/.ssh/trid-macmini-ed25519 eomgyuhyeon@eomgyuhyeon-ui-Macmini.local`: 호스트 이름을 풀지 못했다.
  - mDNS `dns-sd`(5초): 응답이 없다.
  - 이 맥북은 지금 172.30.1.55 네트워크에 있다. 같은 증상이 `HYBRID-RAG-001`에도 기록돼 있다.
- **확인하지 못한 것:** 모델 존재 여부, 메모리, 실행 가능성.
- 같은 기록에 따르면 맥미니는 운영 서비스(rag, secho, ollama)를 돌리고 있었다. 접속되더라도 그 서비스를 내리지 않는 범위에서만 쓴다.

## 2. 블로커와 필요한 결정

1. **맥미니 접속 경로가 없다.** 다음 중 하나가 필요하다.
   - 맥미니와 같은 네트워크
   - 사용자가 주는 접속 주소 / 사용자 지정
   - 사용자가 맥미니에서 직접 `ollama list` 결과를 전달
2. **대체 후보 사용 여부.** 지시받은 대상은 `qwen2.5:7b`다. 맥북에 이미 설치된 다른 모델로 바꾸려면 사용자 승인이 필요하다.
   - `exaone3.5:7.8b`: 비상업 라이선스라 공공기관 제품 후보로 쓰기 어렵다.
   - `gemma3:12b`: Gemma 이용 약관을 따른다.
   - `qwen3:30b-a3b`: Apache-2.0이지만 메모리 부담이 크다.

## 3. 비교 절차 (실행 조건 충족 후, C-031 동결 해제 뒤)

**대상과 서빙 설정**

| 모델 | 서빙 | 템플릿 | 설정 |
|---|---|---|---|
| Qwen3-1.7B 원본 | 실서비스와 같은 GGUF를 격리 llama-server로 따로 띄움 | 공식 template(GENERAL-CHAT-003에서 렌더링 일치 확인), `enable_thinking=False` | Qwen3 비사고 권장 샘플링(temperature 0.7, top_p 0.8, top_k 20, min_p 0). presence_penalty 1.5는 서비스 값을 그대로 기록 |
| Qwen2.5-7B (Ollama `qwen2.5:7b`) | Ollama `/api/chat` | Ollama Modelfile template(ChatML) 원문과 sha를 기록 | 공식 generation_config(temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.05) `[INFERENCE: Qwen2.5-7B-Instruct 공개 설정, 실행 전 모델 디렉터리에서 재확인]` |

**대화와 조건**
- 공통 대화는 개발용이다.
  - B `GENERAL-CHAT-001-SCENARIOS.md` 8개
  - A 확인 대화 P1~P3(고유명사·번역·이름), D2·D3
  - C-031 비공개 시나리오는 쓰지 않는다.
- 공통 조건:
  - 시스템 프롬프트는 같은 문장(현재 `CHAT_SYSTEM`)을 쓴다.
  - 출력 한도는 384토큰으로 같다.
  - 대화마다 세션을 새로 시작한다. 모델끼리 history를 섞지 않는다.
  - 각 모델의 history에는 그 모델 자신의 답이 들어간다.
  - 시나리오마다 seed 3개로 반복한다.

**기록 항목**
- 턴별 원문
- 조건 준수: 매운 음식, 견과류
- 축약 지시, 번역 핵심(빠른 답장), 이전 판 재현
- 생략형 후속, 반복
- 지연: 첫 토큰 시간과 전체 시간. 같은 맥에서는 동시에 실행하지 않는다.
- stop 사유와 잘림

**판정 원칙**
- 세대(Qwen2.5 대 Qwen3)와 크기(7B 대 1.7B)가 함께 다르다. 그래서 차이를 파라미터 수의 효과로 해석하지 않는다.
- 어느 모델도 자동으로 기본값이 되지 않는다.
- 메모리가 부족하면 실서비스 모델을 내리지 않고 기다린다.

## 4. 함께 처리한 것: C-028 결과 반영 (배포는 동결 해제 뒤)

**C-028 결과**(`reports/coordination/C-028.md`): 수리본(`model-repair-002`)은 원본 대비 회귀다.

| 지표 | 원본 | 수리본 |
|---|---|---|
| case 결론 정확 | 31 | 22 |
| 256토큰 안 EOS 종료 | 32/50 | 3/50 |
| RAG310 grammar_instructed | 98 | 14 |
| evidence_constrained 과보류 | — | 128/228 |

**반영한 곳**(커밋 `6eef88a`):
- `deploy/shared-demo/versions.json`(표시명·상태·평가)
- `README.md`
- `docs/SHARING.md`
- `MODEL-REPAIR-002.md`

표시는 "평가 완료 · 원본 대비 회귀 · 사용 비권장"이다. 모델 파일은 지우지 않았다. 기본값은 원본 그대로다. 추가 학습은 없다.

**사이트 표시명 반영:** 다음 재시작 때 반영된다. C-031 동결 중이므로 "FREEZE RELEASED" 뒤에 배포한다.

SUPERVISION_REPORT_READY BASELINE-COMPARE-001
