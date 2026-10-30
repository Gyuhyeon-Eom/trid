# A → B: 챗봇 UI 인계와 API 계약 (A-UI-API-CONTRACT-001)

- **작성:** A (01a18739), 2026-10-30
- **소유 경계:**
  - **B 단독 소유 (지금부터):** `trid/shared_ui/index.html`, `trid/shared_ui/app.js`, 그 안의 CSS
  - **A 소유:** `scripts/shared_demo.py`(API·서버), `deploy/shared-demo/versions.json`, `deploy/shared-demo/worker.mjs`, 최종 배포와 통합
- **인계 기준 커밋:** `c8b148b`. A는 이 커밋 이후 UI 파일을 수정하지 않는다.
- **충돌 처리:** B가 커밋하면 A가 받아 통합·배포한다. 충돌이 나면 A가 통합한다.

## 1. 서버가 페이지에 채워 주는 자리표시자 (`index.html`)

| 자리표시자 | 내용 |
|---|---|
| `TOKEN_PLACEHOLDER` | CSRF 토큰. `<meta name="trid-token" content="TOKEN_PLACEHOLDER">` 형태를 유지해야 한다. 모든 POST에 `X-Trid-Token` 헤더로 보낸다. |
| `MODEL_PLACEHOLDER` | 기본 버전 표시명(HTML 이스케이프됨) |
| `VERSION_OPTIONS_PLACEHOLDER` | `<option value="버전id">표시명</option>` 목록. 기본 버전은 목록 첫 항목이 아닐 수 있다. 기본 버전은 `"default": true` 항목인 `trid-assistant-qwen3-1.7b`다. 현재 서버는 `selected`를 넣지 않으므로 B가 기본값을 맞추거나 A에 요청한다. |
| `VERSION_CARDS_PLACEHOLDER` | 버전마다 `<pre class="muted version-card" data-version="id" [hidden]>…</pre>`. 기본 버전 카드만 보인다. 카드 텍스트에 기반 모델·라이선스·측정·한계가 들어 있다. |
| `WORKSPACE_SCOPE_PLACEHOLDER` | 검색 자료 범위 문구(문서·구간 수, 종류, 기준일 범위, 공개 자료 고지) |

- **보안 헤더:** CSP는 `script-src 'self'`다(인라인 스크립트 금지). 스타일은 `'unsafe-inline'`이 허용된다. 외부 리소스는 허용되지 않는다.
- **정적 파일:** 서버가 내주는 정적 파일은 `/`와 `/app.js` 두 개뿐이다. 별도 CSS 파일이 필요하면 A에 경로 추가를 요청한다. Worker의 GET 허용 목록도 `/`, `/app.js`, `/healthz`뿐이다.

## 2. API (모두 같은 출처, `POST`, `X-Trid-Token` 필수, 요청 본문 2MB 이하)

| 경로 | 요청 | 응답 |
|---|---|---|
| `/api/ask` | `{question, version, scope:"workspace"}` (공개 자료 검색과 대화) 또는 `{question, version, document_id}` (업로드 문서) | 아래 3절 |
| `/api/text` | `{text}` (10~5,000자) | `{document_id, name, pages:[str], warnings:[str]}` |
| `/api/load` | PDF 바이트. 헤더 `Content-Type: application/pdf`, `X-File-Name: encodeURIComponent(이름)` | 위와 같음 |
| `/api/sample` | `{}` | 가상 출장 규정 문서. 위와 같음 |
| `/api/clear` | `{}` | `{cleared:true}`. 서버의 문서와 **대화 기억을 모두 삭제**한다. |
| `/api/pin` | `{choice: 번호 \| null}` | 고급 기능. 직전 응답의 참고 문서 중 하나로 다음 질문 고정. 기본 UI에서는 노출하지 않아도 된다. |

**오류 응답:** `{error: "한국어 메시지"}`
- 403: 세션 만료, CSRF 오류, Origin 오류 → 새로고침 안내
- 409: 업로드 문서 없음
- 413: 크기 초과
- 422: 입력 오류
- 429: 분당 20회 초과
- 503: 다른 질문 처리 중 / 서버 꺼짐(Worker)

**세션:** 쿠키 `__Host-trid`(HttpOnly)로 구분한다. 15분 동안 쓰지 않으면 문서와 대화 기억이 삭제된다.

## 3. `/api/ask` 응답

**공통 필드:**
- `intent`: `"chat"` | `"search"`. 기본 버전(assistant)만 모델이 고른다. 다른 버전은 항상 `"search"`다.
- `answer`: 말풍선 본문
- `model`: 버전 표시명
- `version`: 버전 id
- `elapsed_ms`
- `status`, `scope`(작은 안내 문구)

**`intent:"chat"` (일반 대화):**
- `status: "chat"`, `answer`: 모델 응답
- `commentary_notes`: 문자열 배열, 예: "근거 없이 수치·기한·조문을 말한 문장 N개를 숨겼습니다…"
- 출처와 검증 표시는 없다.
- **표시 요구사항:** 말풍선에 "일반 대화 · 문서 근거 없음 · 검증 대상 아님" 표시를 작게 유지한다.

**`intent:"search"` (근거 검색):**
- `status`:
  - `quoted`: 확인된 인용 있음
  - `unverified`: 인용 없음, 해설만
  - `answer`: 값 형식 버전이 위치를 검증한 답
  - `review`, `not_found`
  - `clarify`: 기관 되묻기. `answer`에 자연어 질문이 들어 있다.
  - `unknown_institution`, `institution_no_match`: 대체 답을 하지 않는다는 안내
  - `model_failure`: 모델이 질문이나 지시문을 복사함
- `scope`: 문서 선택 방식 설명(한 줄). 출처 블록에 넣는다.
- `workspace`(공개 자료 검색일 때):
  - `{applied: {doc_id, title, institution, kind, date, date_type, effective_date, currency, source(URL), license} | null, others:[같은 형식], selection, institutions, title_matched, continued, pinned}`
  - `continued: true`이면 질문에 기관·문서명이 없어 **앞선 대화에서 쓴 문서에서 이어서 찾은 것**이다. "앞선 대화의 문서에서 이어서 찾음 · 다른 기관이면 기관명을 넣어 주세요"를 보이게 표시한다.
  - (변경 — `c8b148b` app.js가 읽는 `search_query`는 더 이상 오지 않는다. 검색어는 질문 그대로다.)
  - **출처 블록 요구사항:** `applied`의 기관·문서명·종류·기준일(`date_type` 포함)·원문 링크·`currency`를 각 답변에 접힌 상태로 보여 준다. "현행"이라고 표시하지 않는다.
- `candidates`: 확인된 인용 목록. 각 항목에서 쓰는 필드:
  - `text`: 구간 원문
  - `prediction.source_character_span: [a, b]`: **코드포인트 기준** 오프셋. `Array.from(text)`로 잘라 [a, b) 구간을 강조한다.
  - `prediction.answer`
  - `source_label`: "기관 · 문서 · 조문 위치"
  - **요구사항:** 인용 구간은 강조하고, 각 답변의 출처 블록에 접힌 상태로 둔다.
- `answer_mode`: `"case"`이면 강조 구간 이름을 "원문 인용 · 글자 그대로 확인됨"으로 한다.
- `commentary`(문자열 | null): 모델 해설. **반드시 "모델 해설 · 검증 안 됨"으로 구분해 표시한다.** 확인된 인용과 섞지 않는다.
- `commentary_notes`: 문자열 배열. '숨겼습니다'가 들어간 문구는 경고 색으로 표시한다.
- `table_warning`(선택): 표 경고. 경고 색으로 표시한다.
- `checked_chunks` / `total_chunks`
- `context_notes`: 문자열 배열. 모델 문맥 한도 때문에 근거를 뺀 경우 그 전체 문구가 들어간다. 뺀 것이 없으면 `[]`이다. 서버가 `scope` 끝에도 같은 문구를 붙인다.
- `context_limits` (UI-SCOPE-004, RETRIEVAL-REGRESSION-002): 모델 문맥 한도로 실제로 뺀 근거의 개수.
  - 형식: `{linked_passages, table_rows, tables, own_passages}`. 모두 정수이고, 키 4개가 항상 있다. 뺀 것이 없으면 0이다.
  - 값은 `fit_context`가 실제로 뺀 개수다. 표시 문자열을 파싱하지 않는다.
    - `linked_passages`: 함께 넣으려던 다른 문서(별표) 구간
    - `table_rows`: 검토된 표 행
    - `tables`: 행 블록 전체를 뺀 표
    - `own_passages`: 적용 문서에서 관련도가 낮은 구간
  - 모델을 실행한 문서 검색 답변에만 있다. 업로드 문서와 공유 자료 검색 모두 해당한다.
  - 일반 대화, 모델을 실행하지 않은 응답(보류·되묻기·정형 값)에는 이 필드가 없다.
  - **표시:** 0이 아닌 항목이 있으면 답변 본문 바로 아래에 한 줄로 표시한다. 접힌 영역 안에 두지 않는다.
  - 필드가 없고 `context_notes`만 있으면(이전 서버) 개수 없이 고지한다.

## 4. 지켜야 할 표시 원칙
- 검증되지 않은 내용(`commentary`, 일반 대화)을 "확인됨"이나 근거로 보이게 만들지 않는다.
- 다음 고지는 짧게 남긴다:
  - 실험 서비스
  - 공개해도 되는 내용만 입력
  - Cloudflare를 거쳐 운영자 Mac에서 처리
- 모델 정보와 평가 상세는 접어 둔다. TRID 1.7B 실험판 카드의 "회귀" 표시는 그대로 둔다.
- 기존 명령(`/clear`, `/pdf`, `/paste`, `/source`)은 버튼으로 바꿔도 된다. `/clear`는 반드시 서버 `/api/clear`를 호출해야 한다(대화 기억 삭제).
- 첨부(PDF·붙여넣기)는 선택 사항이다. 문서를 넣으면 질문은 그 문서에서만 찾는다(`document_id` 포함). 공개 자료 검색으로 돌아가는 수단이 있어야 한다.

## 4a. 사용자 지시 UI-CHAT-001 (B 담당, A가 그대로 전달)
- **기본 채팅 화면:**
  - 메시지 말풍선
  - 하단에 textarea, 보내기, 첨부
  - 상단에 모델 선택
  - 평가 상세는 접기
  - 출처는 답변마다 개별로 접기
  - 모바일과 키보드 조작 지원
- **스타일:** 기존 홈페이지 **DAYOS의 실제 스타일**을 참고한다.
- **첫 화면:** ChatGPT 첫 화면처럼 로고, 한 줄 소개, 입력창을 중심에 둔다.
- **접어 둘 것:** 긴 개발 설명, 자료 건수, 명령어, 평가 수치는 "모델 정보" 또는 "도움말" 안에 접는다.
- **필수 안내:** 짧게만 둔다.
- **문구:** UI 문구를 더 늘려 화면을 다시 빽빽하게 만들지 않는다.
- **색상 (사용자 최신 피드백):**
  - 사용자 의견: "색깔이 너무 다채롭다. 깔끔하고 모던하게."
  - **무채색 기반에 DAYOS accent 1개만** 쓴다.
  - 그라데이션과 여러 색 장식을 없앤다.
  - 장식 요소를 새로 더하지 않는다.
- **소유권:** A는 backend, API, 배포를 맡고 B는 frontend를 맡는다.

## 5. 현재 상태와 일정
- **A:** 대화 백엔드(라우팅, 대화, 후속 문맥) 구현과 스모크를 계속한다. 배포는 B의 UI 커밋을 받은 뒤 함께 한다.
- **현재 사이트:** 이전 배포(Worker `a4aaab93…`)가 돌고 있다. 대화 백엔드는 아직 배포하지 않았다.
- **로컬 확인:** B가 확인할 때 쓸 수 있는 서버가 있다.
  - 실행: `.venv/bin/python scripts/shared_demo.py --versions /tmp/versions-smoke.json --secret-file /tmp/trid-smoke-secret --port 51874`
  - 프록시 헤더 `X-Trid-Proxy`(비밀값 파일), `X-Trid-Host: test.tridatum.co`가 필요하다.
  - 브라우저로 직접 확인하기는 어렵다. 필요하면 A에 요청한다.
