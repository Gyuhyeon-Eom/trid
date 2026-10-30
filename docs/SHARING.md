# trid 임시 웹 공유

공유 주소: https://test.tridatum.co

이 배포는 Cloudflare Worker → 임시 Cloudflare Tunnel → 이 Mac의 CPU 모델 서버로 연결됩니다. 별도 클라우드 머신으로 모델을 옮긴 배포가 아닙니다. Mac 전원과 인터넷 연결이 필요하며 덮개를 닫거나 재부팅하면 중단될 수 있습니다. 화면은 꺼져도 되도록 `caffeinate -i`를 서버 실행 동안에만 사용합니다.

**2026-10-30 17:53 재배포 (HYBRID-RAG-001 + TRID-ANSWER-002).** Worker `e90e8468…`, 색인 `runs/workspace-index/v3`. 24시간 실행하며 자동 종료 예정은 **2026-10-31 17:53 KST**다. 중단되면 도메인은 임시 서버 종료 안내를 표시한다. 소규모 테스트용이며 운영 서비스의 가용성을 보장하지 않는다.

## 제공 버전 (`deploy/shared-demo/versions.json`)

| 버전 | 기반 모델 | 라이선스 | 상태 |
|---|---|---|---|
| `trid-assistant-qwen3-1.7b` (**기본**) | Qwen/Qwen3-1.7B-GGUF Q8_0 원본(업무 특화 학습 없음), sha `061b54da…`, llama.cpp 8ca4060 Metal | Apache-2.0 | 대화 + 근거 검색. 특화 모델 아님 |
| `trid-v0.1-reader` (롤백 기준) | 자체 91M 추출 독해기 `runs/reader-factorial-01-cpt-v1-v2-v4` (sha `90ccb60f…`) | 자체 | 실험판, 품질 기준 미달 |
| `baseline-qwen3-1.7b` | 위와 같은 원본 GGUF(서버 공유), 값 형식 출력 | Apache-2.0 | 비교 기준선, 품질 기준 미달 |
| `trid-qwen3-1.7b-pilot-sft-001` | 원본 + LoRA pilot-sft-001 (GGUF `0c78b0c2…`) | Apache-2.0 | **C-023에서 원본 대비 회귀.** 버전 확인용 |

- 각 버전의 기반 모델, 라이선스, 실측, 한계는 화면의 모델 정보(접힘)에 표시한다.
- 같은 모델 파일·실행 파일·context를 쓰는 버전은 llama-server 하나를 공유한다.

**롤백:** `versions.json`에서 `trid-assistant-qwen3-1.7b` 항목을 빼면 이전 기본(`trid-v0.1-reader`)으로 돌아간다. 그다음 `stop` → `start` → `deploy`를 실행한다.

## 일반 대화와 근거 검색 (TRID-CONVERSATION-001, 기본 버전)

- **경로 선택:** 입력마다 서비스 모델 자신이 2단어 문법(`검색` | `대화`)으로 처리 경로를 고른다. 키워드 규칙이나 고정 응답은 없다.
  - 2-word grammar는 입력만 보고 판단한다.
  - 직전 턴이 근거 검색이었으면, 기관·문서명이 없는 입력 중 직전 문서가 전체 최고 점수의 60% 이상으로 답하는 것은 이어지는 검색으로 본다(`Workspace.continues`).
- **일반 대화:**
  - 모델의 기본 chat template에 최근 6개 메시지(세션 메모리)를 넣는다.
  - 근거 검색을 하지 않으며 "일반 대화 · 검증 대상 아님"으로 표시한다.
  - 숫자+단위(원, %, 일, 조 등)가 들어간 문장은 근거가 없으므로 숨기고 숨긴 개수를 알린다. API 응답에도 남기지 않는다.
- **근거 검색:**
  - 공유 색인 `runs/workspace-index/v3`(14,672건 / 101,119구간; v2 + B API 스냅샷 신규 5건)에서 hybrid 검색(bge-m3 0.9 + BM25 0.1, 권위 가중치)으로 문서 1건을 고른다. 근거 적합성(코사인 + 핵심 명사)을 통과한 문서만 쓴다. B search-corpus-v4 검토를 질의 단계에서 적용한다(개인정보 2구간 제외, 값 보류 87구간은 숫자를 답으로 쓰지 않음). 다시 만드는 순서: `build_workspace_index.py --practice --api-snapshot …` → `build_embeddings.py`(변경분만 임베딩). 상세는 `reports/coordination/HYBRID-RAG-001.md`.
  - 원문과 글자 그대로 일치하는 인용만 확인된 근거로 표시하고, 모델 해설은 "검증 안 됨"으로 표시한다.
  - 기관이 모호하면 되묻는다.
  - 자료에 없는 기관이거나 일치하는 문서가 없으면 모델을 실행하지 않고 그렇다고 답한다. 이때 일반 대화로 넘기지 않는다.
  - 이어지는 질문은 직전 문서에서 찾고 `workspace.continued`로 표시한다.
- **대화 기억:** 대화 기억과 업로드 문서는 `/clear` 또는 15분 미사용 시 삭제된다. 검색 턴은 대화 기억에 `[근거 검색 결과] …(적용 문서 …)`로만 남는다. 근거 자체는 매번 다시 검색한다.
- **알려진 한계:** 라우터 개발 측정 27개 중 25개를 맞혔다.
  - 틀린 2개는 업무 단어가 들어간 글쓰기 요청(보고서 제목 추천, 공문 문장 다듬기)이 검색으로 간 경우다. 안전한 쪽으로 틀린 것이지만 원하는 답은 나오지 않는다.
  - 1.7B 원본은 인용 없이 "검증되지 않음" 해설만 내는 경우가 많다.

## 사용

- 기관·규정명을 넣어 물으면 업무 질문은 공유 검색 색인에서 근거를 찾습니다. 다른 후보 문서는 `/doc 번호`로 고정할 수 있습니다(고급 기능).
- `/pdf`, `/paste`로 넣은 문서가 있으면 그 문서에서만 찾습니다. 텍스트 PDF는 2MB·20페이지·5만 자, 붙여넣기는 5,000자까지입니다.
- 모델 파일은 방문자에게 내려주지 않습니다. GGUF와 llama.cpp 실행 파일은 `runs/`에 있으며 기동할 때 sha256을 검사합니다.

PDF와 질문은 Cloudflare HTTPS 중계를 거쳐 운영자의 Mac으로 전송됩니다. 비공개·민감한 자료 대신 공개 가능한 테스트 자료만 사용하세요. 앱은 문서와 질문을 파일·학습 자료로 저장하지 않습니다. 문서는 HttpOnly/Secure/SameSite 세션별 메모리에 저장되며, 지우기 또는 15분 미사용 후 정리됩니다. Cloudflare 자체 네트워크 처리·메타데이터 정책과는 별개입니다.

## 중지 / 재실행

프로젝트 폴더에서:

```sh
.venv/bin/python scripts/share_demo_control.py status
.venv/bin/python scripts/share_demo_control.py stop
```

다시 24시간 열기:

```sh
.venv/bin/python scripts/share_demo_control.py start --hours 24
.venv/bin/python scripts/share_demo_control.py deploy
```

새 실행마다 터널 주소가 바뀌므로 `deploy`까지 실행해야 도메인이 새 서버로 연결됩니다. 시작이 실패하면 `.shared-demo/server.log`, `supervisor.log`, `tunnel.log`를 확인합니다. 비밀키는 `.shared-demo/proxy-secret`(0600)에 있고 Cloudflare Worker secret으로도 저장됩니다. 이 폴더를 공유하지 마세요. Cloudflare OAuth 자격 증명은 기존 Wrangler 저장소를 사용하며 프로젝트에 복사하지 않습니다.

2026-10-27 연결 복구: 터널 프로세스가 살아 있어도 실제 연결이 0개인 상태가 발생했습니다. 이제 시작 시 터널의 `/ready`와 외부 HTTPS 모델 응답을 모두 확인하며, 실행 중에는 30초마다 검사합니다. 두 번 연속 실패하면 모델 서버를 유지한 채 터널을 재시작하고 Worker의 연결 주소를 재배포합니다. 새 터널의 DNS 음성 캐시를 피하기 위해 외부 상태 검사는 요청별 DNS-over-HTTPS를 사용하며 Mac의 DNS 설정은 변경하지 않습니다. 복구 중 잠시 중단될 수 있고, 네트워크·Cloudflare 인증 문제가 지속되면 자동 복구도 완료되지 않을 수 있습니다. `status`의 `connection_status`, `last_health_at`, `recoveries`로 확인할 수 있습니다.

서버는 127.0.0.1에만 바인딩하며 인터넷 터널 직접 접근은 프록시 비밀키 없이 거절합니다. 요청 원본과 CSRF 토큰을 검사하고, 브라우저가 보내는 프록시 신원 헤더는 Worker가 덮어씁니다. Worker는 IP별 분당 30회, 서버는 세션별 분당 20회를 제한합니다. 최대 32개 세션, HTTP 작업 8개, 모델·PDF 처리 1개를 허용합니다. PDF는 별도 프로세스에서 15초 제한 및 CPU/메모리 감시하에 읽습니다. 이 조치들은 정식 보안 심사나 운영용 격리 환경을 대신하지 않습니다.

## 검증

- `tests/test_shared_demo.py`: 방문자별 문서 격리, CSRF/Origin/프록시 차단, 만료, 잘못된 입력, 크기 제한.
- `deploy/shared-demo/worker.test.mjs`: 경로 차단, 프록시 헤더 교체, 비밀키 전달 범위, 연결 종료 화면.
- `reports/shared-demo-local-smoke.json`: 이전 short-v3-hangul 모델의 샘플 4항목, PDF 담당 부서 응답에 대한 역사적 검증.
- `reports/shared-demo-public-smoke.json`: 이전 모델의 HTTPS 공개 추론과 방문자 문서 접근 차단 검증. 현재 추출 모델의 새 측정 결과가 아닙니다.
- `reports/shared-demo-extractive-update.json`: 새 가중치 해시, 배포 버전, 272개 회귀 테스트 통과, 공개 브라우저 `/sample`·`/paste` 담당 부서 응답과 정확한 원문 하이라이트 확인. 실제 모델의 로컬 샘플 해외 식비·숙박비 질문은 보류했으며 정답으로 집계하지 않았습니다. 이번 공개 UI 검증은 PDF 업로드를 재실행하지 않았습니다.
- 공개 문서 선택 UI(정책브리핑 20건)는 2026-10-30에 제거했습니다. 같은 20건은 공유 검색 색인에 포함됩니다. 질문 중심 경로의 검증은 `reports/coordination/TRID-WORKSPACE-001.md`에 있습니다.
- 공개 HTTPS 브라우저에서 문서 선택·교체·질문 요청·`/source` 원문 표시를 확인했다. 잘못된 문서 ID는 422로 거절하며 기존 문서를 유지한다. 전파산업 지원 비율·한도 질문은 6/6구간 확인 후 모델이 보류했고, 산림투자 설명회 참가 비용 질문은 0/4구간 검색 실패였다. **배포 경로 동작 검증이지 정답률 개선이 아니다.** 새 103건 SFT 데이터의 본학습·모델 배포는 하지 않았다.
- `reports/shared-demo-release-test-001.json` (RELEASE-TEST-001):
  - 로컬 HTTP 스모크에서 확인한 것: 프록시 비밀키 없는 접근 403, 모델 경로 404, 잘못된 버전 422, CSRF 403, 외부 Origin 403.
  - 두 버전의 실제 응답 원문도 기록했다.
  - 공개 HTTPS 헤드리스 브라우저에서 `/sample` → v0.1 답(운영지원팀, 원문 강조 일치) → 버전 전환과 기준선 카드 표시 → 기준선 답(복귀 후 7일 이내 [D1], 원문 강조 일치)을 확인했다.
  - 문장으로 바꿔 말한 출력은 답으로 표시하지 않음도 확인했다.
  - **배포 경로 동작 검증이지 정확도 측정이 아니다.**

새 DNS 레코드는 일부 네트워크의 이전 NXDOMAIN 캐시 때문에 약 30분 늦게 보일 수 있습니다. 권한 DNS, Cloudflare 1.1.1.1, Google 8.8.8.8에는 새 주소가 반영됐습니다. 접속이 안 되면 잠시 후 또는 다른 네트워크에서 다시 시도하세요.

참고: [Cloudflare 임시 터널](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/), [Workers Custom Domains](https://developers.cloudflare.com/workers/configuration/routing/custom-domains/).
