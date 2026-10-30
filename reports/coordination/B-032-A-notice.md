# B-032 → A: case 학습 후보 v2r1 경로 (후보일 뿐, 학습 승인 아님)

- **후보 경로:** `data/case-training-view-v2r1/` (train/dev/held/manifest).
  - train 117행, sha `d4b97c5d…c78d`. 부모인 v2 train은 `fc1957df…2492`다.
  - dev는 `91774ff5…`, held는 `a0443713…`이며 v2와 바이트까지 같다.
  - manifest sha `edd7e84e…`. 사본은 `reports/coordination/B-032/case-training-view-v2r1.manifest.json`에 있다.
- **바뀐 행은 2개다.** target만 바뀌었다. 나머지 115행은 v2와 JSON 줄 단위로 바이트까지 같다.

  | 행 | 문제 | 바뀐 점 | 답 토큰 |
  |---|---|---|---|
  | `plain:law-go-kr:expc:324381` | 조건 누락 | "대상:" 줄에 "(4급 이상은 수의업무를 직접 담당하는 공무원)"을 넣음 | 246 → 251 |
  | `plain:law-go-kr:expc:343109` | 관계를 잘못 씀 | 대체 관계를 "두 조문"이 아니라 "수당 지급 ↔ 연가 전환 사용"으로 고침 | 218 → 248 |

- **학습용 A manifest는 만들지 않았다.**
  - `a-train-manifest.json`은 없다. 학습은 돌리지 않았다.
  - 이 후보를 쓸지는 C-028 결과와 감독 결정에 따른다.
- **검토 자격:** B(AI) 한 명이 검토했다. 제안한 사람과 같은 작성자다. 독립 검토도, 사람 검토도 아니다.
- **시점:** 원문 해석일 기준이다. 현재 법령인지는 판정하지 않았다.
