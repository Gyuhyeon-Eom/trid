# B 인벤토리 메모 001: PRISM, 광역 지자체, ALIO 공식 경로

작성: B, 2026-10-30 07:25 KST. 목록 조사만 했다. 원문 수집은 시작하지 않았다.

## PRISM (정책연구관리시스템, www.prism.go.kr)

- **구조:** React SPA다. 백엔드는 `https://api.prism.go.kr/prism-be-{auth,sys,prtl,bscs,asmt}`(`/env.js`)이다.
  - `www.prism.go.kr/robots.txt`는 `Disallow:`(빈 값)이다.
  - `api.prism.go.kr/robots.txt`는 404다. robots.txt가 없다고 허용으로 보지 않는다.
- **공개 과제 조회 경로(번들 `static/js/main.9fad982f.js`에서 확인):**
  - `/asmt/entire/{progress/list, progress/info, lately/info, many/info, recommend/info}`
  - `/asmt/retrieve/{list, listResultInfo, listResultUslz, listInterim, listContarct, listSlctn}`
  - 첨부는 `asmtId`와 `atchFileSn`으로 내려받는다.
  - 요청 형식과 페이지 크기는 아직 호출해 보지 않았다.
- **사용권:** 보고서마다 공공누리 코드가 붙는다.

  | 코드 | 공공누리 유형 |
  |---|---|
  | D0590002 | 1유형(출처표시) |
  | D0590003 | 2유형(+상업적 이용금지) |
  | D0590004 | 3유형(+변경금지) |
  | D0590005 | 4유형 |
  | D0590006 | 자유이용 |

  학습 포함은 D0590002·D0590006만 후보로 두고, 2~4유형과 코드 없음은 격리한다(B 제안. 총괄 결정 필요).
- **다음 단계:** 목록 API 1~2회로 응답 필드(과제 ID, 기관, 연도, 공공누리 코드, 첨부 목록)를 확인한다. 간격은 1 s 이상이다.

## 광역 지자체 17곳

- robots.txt 원문과 상태는 `data/corpus-inventory/robots/`에 있다(2026-10-29T18:36Z).
- 대부분 경로가 Disallow인 곳: seoul, daejeon, sejong, gyeongnam, jeju.
  - 보도자료 게시판 경로가 허용 목록에 있는지 경로별로 확인해야 한다(예: 서울 `Allow: /realmnews`).
- 접근 문제: busan 401, chungbuk 접속 실패, incheon robots 404.
- 보도자료 게시판 URL과 게시물별 공공누리 표시는 아직 매핑하지 않았다.

## ALIO 공식 경로 (B-003-contact.md 요약)

- 공공데이터포털 "재정경제부_공공기관_내부규정 정보"(15068122): PDF, 이용허락범위 제한 없음. 내용은 미확인이다.
- 재정경제부 경영관리과 오픈 API 3종(기관정보 15125287, 채용 15125273, 사업 15125286): 서비스키가 필요하다. 내부규정 원문 API는 없다.
