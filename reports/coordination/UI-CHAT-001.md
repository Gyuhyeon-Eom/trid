# UI-CHAT-001 — 일반 챗봇 화면 (B, 임시 배정)

## 상태
- **미배포.** 실사이트(서버 pid 3497, 포트 51873, cloudflared 뒤)는 바꾸지 않았다.
  - 실사이트는 요청마다 디스크의 `trid/shared_ui`를 읽는다. 그래서 작업은 main이 아니라 별도 worktree에서 했다.
  - 실URL에서 바뀐 화면을 확인하는 일은 **A의 병합·배포 뒤에 할 일**이다. 아직 확인하지 않았다.
- 커밋 `14d9603`
  - 브랜치 `ui-chat-001`, base `f5e5b69`.
  - worktree `/tmp/trid-ui-wt`.
  - 바꾼 파일은 `trid/shared_ui/index.html`, `trid/shared_ui/app.js` 두 개뿐이다.
  - `scripts/shared_demo.py`, `deploy/shared-demo/versions.json`, `worker.mjs`, API, 테스트는 건드리지 않았다.
- 테스트: worktree에서 `.venv/bin/python -m unittest discover -s tests -q` → `Ran 358 tests … OK (skipped=4)`.
  - 데이터 symlink: `data/security-suite-v1`, `data/reading-demo-v3`.

## A에게 요청
1. 브랜치 `ui-chat-001`의 `14d9603`을 병합하고 실사이트에 배포해 주세요.
2. 배포 후 실URL에서 첫 화면, 질문 1회, 모바일 화면을 확인해 주세요.
- 로컬 확인에는 A의 **미커밋** `scripts/shared_demo.py`(작업 시점 사본)를 썼다.
  - placeholder 5종과 `<meta name="trid-token">`이 그 버전에서 정상 치환되는 것을 확인했다.
  - 커밋 전에 그 사본은 되돌렸다(`git checkout scripts/shared_demo.py`).

## 디자인 근거 (DAYOS)
- "DAYOS"라는 이름 자체는 찾지 못했다.
  - repo 문서·deploy, `~/사업/homepage`, `~/사업/tridatum`, `~/사업/05 브랜드와 디자인`, https://tridatum.co 본문을 검색했고 모두 0건이다.
  - 계약서 `A-UI-API-CONTRACT-001.md` 105행에도 "DAYOS의 실제 스타일"이라는 문구만 있다.
- 그래서 **실브라우저로 본 https://tridatum.co**와 그 소스를 기준으로 삼았다.
  - 소스: `~/사업/tridatum/assets/css/design-tokens.css`, `chat.css`. 이 사이트 안의 AI 채팅 위젯 스타일이다.
  - 실측값:
    - 바탕 `#f3f3f3`, 글자 `#000`.
    - 본문 서체 `"Wanted Sans", Archivo`, 제목 Archivo 700에 음수 자간.
    - accent `#fff100`. 위젯 전송 버튼은 `#eee000`.
    - 버튼: 검정 primary에 radius 8px. 보조 버튼은 테두리 `#ccc`.
    - pill 내비게이션, 카드 radius 16–28px.
- [추정] DAYOS는 이 tridatum 디자인 체계를 가리키는 이름으로 보았다. 실제로 다른 사이트를 뜻한다면 다시 맞춰야 한다.
- 웹폰트는 쓰지 않았다.
  - CSP가 self만 허용하고, 서버는 `/`와 `/app.js`만 제공한다.
  - 그래서 Wanted Sans나 Archivo가 설치되지 않은 PC에서는 시스템 서체(Apple SD Gothic Neo, 맑은 고딕 등)로 보인다.

## 사용한 색 토큰 (2차 피드백 "너무 다채롭다" 반영)

| 토큰 | 값 | 쓰는 곳 |
|---|---|---|
| `--canvas` | `#fff` | 바탕, composer, 대화창 |
| `--surface` | `#fafafa` | 인용·출처 카드 |
| `--fill` | `#f2f2f2` | 사용자 말풍선, hover |
| `--ink` / `--soft` / `--muted` | `#111` / `#3d3d3d` / `#6b6b6b` | 본문, 보조글, 안내 |
| `--line` / `--line-strong` | `#e5e5e5` / `#d4d4d4` | 얇은 테두리 |
| `--accent` | `#fff100` | **전송 버튼, 키보드 포커스 링에만** |
| `--error` | `#b42318` | 실제 오류 메시지와 `model_failure` 배지에만 |

- 상태 배지(확인됨, 검증 안 됨, 근거 못 찾음 등)는 모두 같은 회색 테두리 pill이다. 상태는 글자로 구분한다.
- 인용 강조는 색 배경 대신 굵은 글씨와 밑줄로 했다.
- 없앤 것: 녹색·주황 배지, 노란 형광 배경, 그라데이션, 큰 그림자.
- 대비: `--muted #6b6b6b`는 흰 바탕에서 약 5.3:1로 AA를 충족한다.
- 실브라우저 색 감사:
  - 대화 1회와 PDF 인용 답변이 있는 화면에서, 보이는 모든 요소의 `color`, `background`, `border`, `outline`, `box-shadow` 중 무채색이 아닌 값을 모았다.
  - 데스크톱과 모바일 모두 결과는 `rgb(255, 241, 0)` → `#send` 하나뿐이었다(포커스 링 제외).
- 다크 모드는 넣지 않았다. tridatum.co가 밝은 테마뿐이어서 `color-scheme: light`로 고정했다.

## 화면 구성
- **첫 화면**
  - 상단: `trid 실험판`, 모델 선택, ✎(새 대화), i(정보) 버튼.
  - 가운데: "무엇을 도와드릴까요?"와 입력창.
  - 입력창 아래 한 줄: "공개 자료만 입력해 주세요 · 답변은 원문 확인이 필요합니다 · 자세히".
  - 실측 결과, 첫 화면에서 보이는 글은 이것뿐이다. 모델 option 라벨은 제외했다.
  - 개발 설명, 코퍼스 건수, 날짜 범위, 평가 수치, 명령어 목록, 여러 경고문은 첫 화면에서 없앴다.
- **대화가 시작되면** 입력창이 하단에 고정된다.
  - 사용자 말풍선은 오른쪽 정렬, 회색 바탕 `#f2f2f2`에 얇은 테두리.
  - 답변은 말풍선 없이 왼쪽 본문으로 표시한다.
- **i / "자세히" 대화상자** (`<dialog>`, Esc나 바깥 클릭으로 닫힘)에 옮긴 내용:
  - 이용 안내: 실험 서비스라는 점, Cloudflare를 거쳐 운영자의 Mac에서 처리된다는 점, 15분 후 삭제.
  - 선택한 모델 카드. 불리한 평가, 라이선스, 회귀 내용도 **삭제하지 않고 그대로 보존**했다.
  - 검색 자료 범위(`WORKSPACE_SCOPE_PLACEHOLDER`)와 답변 표시 규칙.
  - 사용법과 명령어.
- **회귀 pilot 표시**: option 라벨에 "회귀"가 있으면 모델 선택 옆에 회색 pill "실험 · 회귀"를 표시한다.
- **답변 아래에 바로 보이는 것**
  - 상태 배지. 검증 안 됨이나 실패 상태를 숨기지 않는다.
    - 일반 대화일 때는 "일반 대화 · 문서 근거 없음 · 검증 대상 아님".
  - 숨긴 문장이 있다는 안내와 표 경고.
  - 검증된 인용 구간. 문서 내용과 인용 1건만 표시한다.
- **"출처" 접기에 넣은 것**
  - 적용 문서, 원문 링크, 라이선스.
  - 인용 앞뒤 원문(강조 표시 포함).
  - "모델 해설 · 검증 안 됨 — 원문과 대조하세요".
  - 검색된 다른 문서.
  - 검색어, 잘림 안내, 확인한 구간 수, 모델 이름, 소요 시간(초).
  - 소요 시간 같은 기술 정보는 본문에 상시 노출하지 않는다.
- **첨부**: 입력창 왼쪽 ＋ 버튼으로 연다. 선택지는 PDF 열기, 문서 붙여넣기, 예시 문서.
  - 문서를 넣으면 입력창 위에 "내 문서에서만 찾는 중 · 이름 (N쪽)"과 "문서 보기", "공개 자료 검색으로"가 나타난다.
  - 사용자 문서 선택은 강제하지 않는다. 기본은 공개 자료 검색이다.

## 계약 준수 (A-UI-API-CONTRACT-001)
- 쓰는 엔드포인트: `/api/ask`(`scope: workspace` 또는 `document_id`), `/api/text`, `/api/load`, `/api/sample`, `/api/clear`.
- status 9종과 `intent: chat`을 모두 처리한다. 알 수 없는 status가 오면 원문 그대로 표시한다.
- `candidates[].prediction.source_character_span`은 코드포인트 기준으로 자른다(`Array.from`).
  - 확인: `highlighted('가나다라😀마',[1,3])` → `가<mark>나다</mark>라😀마`.
- 서버 값은 전부 `textContent`로 넣는다. 응답에서 받은 HTML은 쓰지 않는다. 링크는 http(s)일 때만 만든다.
- 인라인 JS는 없다. 외부 `/app.js` 하나만 쓰고, 새 프레임워크나 의존성도 없다.
- `/clear` 명령과 ✎ 버튼은 `/api/clear`를 호출한다. 기존 slash 명령 8종도 유지했다.

## 실브라우저 검증
- 환경
  - 로컬 서버: worktree에서 `shared_demo.py`를 포트 51990으로 실행. llama 포트 51952/51957, 실제 모델.
  - 토큰 주입 proxy: 51991.
  - 관리형 Chromium 탭 2개: 데스크톱 1280×860, 모바일 390×844 (isMobile, touch).
- **데스크톱**
  - [x] 일반 대화 응답에 chat 배지가 붙는다.
  - [x] Shift+Enter로 줄바꿈한 질문이 그대로 전송된다.
  - [x] 연속 전송이 차단된다.
  - [x] 기관이 정해지지 않은 질문 → `clarify` 4개 기관 되묻기.
  - [x] "법제처 공무국외출장규정 기준" 질문
    - `unverified` 배지, 적용 문서, 원문 링크, 해설 "검증 안 됨", 256토큰 잘림 안내가 출처 접기 안에 있다.
  - [x] 붙여넣기
    - Esc로 닫히고 입력창으로 포커스가 돌아온다.
    - 3자 입력 후 Ctrl+Enter → 서버 422 오류 "문서는 10~5,000자로 입력해 주세요."가 `role=alert` 빨간 글로 나온다.
    - 정상 입력 후 Ctrl+Enter → 문서 칩이 나오고, 질문하면 응답 표시.
  - [x] ✎ 새 대화
    - fetch 기록이 `/api/text, /api/text, /api/ask, /api/clear` 순서로 남았다.
    - 메시지 0개, 칩 숨김, 빈 첫 화면으로 돌아간다.
  - [x] ＋ 버튼 `aria-expanded`가 true/false로 바뀐다.
  - [x] Tab / Shift+Tab 포커스: 전송·첨부 버튼에 검정 outline과 accent 링이 보인다.
  - [x] 회귀 모델을 선택하면 "실험 · 회귀" 배지가 나오고, 대화상자에 해당 모델 카드만 보인다.
  - [x] Esc로 대화상자가 닫힌다.
  - [x] 콘솔 오류 0건.
- **모바일 390×844**
  - [x] 가로 스크롤 없음. 빈 화면과 대화 후 모두 확인.
  - [x] PDF 업로드
    - 파일: `/tmp/trid-corpus/c021/dl/000`, 2쪽 공개 보도자료.
    - 업로드하면 첨부 메뉴가 닫히고 읽기 순서 경고가 표시된다.
  - [x] 질문 "국비와 지방비는 각각 얼마인가요?"
    - 실제 응답 `quoted`, 배지 "원문 인용 확인됨".
    - 인용 "국비 50%, 지방비 50%"(2쪽)가 본문에 굵은 밑줄로 보인다.
    - 출처 접기 안의 원문에서도 같은 구간이 `<mark>`로 표시된다.
    - 숨긴 해설 문장 6개 안내도 나온다.
- 스크린샷: 데스크톱·모바일의 첫 화면과 대화 화면을 확인했다.
  - 세션 임시 경로에 있고 저장소에는 넣지 않았다(새 문서 파일 금지).

## 확인하지 못한 것
- 실URL 화면: 미배포라서 확인하지 못했다.
- 실제 한글 IME 하드웨어 입력
  - 조합 중 Enter를 전송하지 않는 동작은 headless Chromium의 조합 이벤트로만 확인했다.
  - 실제 macOS나 Windows IME에서는 확인하지 않았다.
- 실제 iOS/Android 기기, Safari, Firefox: 확인하지 않았다. 관리형 Chromium의 모바일 에뮬레이션만 썼다.
- 웹폰트(Wanted Sans/Archivo) 렌더링: 서버가 폰트를 제공하지 않아 시스템 서체로 보인다.
- 스크린리더 실제 낭독
  - `role=log`, `aria-live`, `role=alert`, sr-only 화자 표시는 넣었다.
  - 실제 낭독은 확인하지 않았다.

SUPERVISION_REPORT_READY UI-CHAT-001
