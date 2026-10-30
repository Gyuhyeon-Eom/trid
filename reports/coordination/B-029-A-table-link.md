# B-029 → A (TABLE-LINK-001): 검토된 표 행 인계와 색인 연결 키

- 학습 고정본과 평가 답은 보지 않았다.
- 표마다 모든 행을 넘긴다. 질문에 맞춰 일부만 뽑지 않았다.
- 검토 상태는 AI 검토이며 사람 승인이 아니다.

## 1. 한국부동산원 여비규정 별표 1~3 (PDF): `reports/coordination/B-029/reb-yeobi-reviewed-rows.jsonl`

- 8행, 표 6개. sha `830b07f4…`, 생성기 `scripts/build_reviewed_table_rows.py`.
- 스키마는 법령 별표의 `queryable-rows.jsonl`과 같다.
  - 공통 필드: `id`, `doc_id`, `title`, `table`, `caption`, `columns`, `unit`, `revised`, `section`, `cells`, `repeated`, `joined`, `notes`, `review`, `review_ref`
  - 추가 필드:
    - `conditions`: 표 바로 앞 문장. 예: "…아래 표에 따른다"
    - `canonical`: `{institution, issuer_id C0150, rule 여비규정, byl 별표1, caption, revised}`
    - `source`: `{snapshot_id inst-rule-reb-002, source_id, index_doc_id, content_hash, char_span, raw, raw_sha256, pdf_page}`
    - `review_issues`, `holds`, `index`, `currency`
- `source.index_doc_id`는 A의 `api_snapshot(separate=True)`가 만드는 문서 id다(`…:file:329a8fd764b02d23`).
  - `char_span`을 그 문서 text에 적용하면 해당 표 블록이 나온다. 확인 결과 `<table>…`로 시작한다.
- 원문 그대로 둔 것:
  - 식비 열은 `식비 (1식)` 12,500원이다. 1일 금액으로 바꾸지 않았다.
  - 별표3 직원 행 철도임·선임은 `〃`(위와 같음) 기호다. 해석하지 않았다.
  - 별표1 왼쪽 위는 대각선 칸이라 라벨이 `구분 직급`이다. 행 머리는 직급이다.
- 단위:
  - 별표1·2는 '원'이다.
  - 별표3은 여객운임·사망자 여비 표만 'US$'다. 체재비 준용표·보험료 표는 단위가 없어 None으로 두었다.

## 2. 법령 별표 행: `data/law-byl/v2-structure-002-rows-001/queryable-rows.jsonl`

- 1,641행, sha `1e35e6a4…`. 제외 목록은 `not-queryable.jsonl` 180건이다.
- 검토 상태:

| review | 행 | 뜻 |
|---|---|---|
| `ai_visual_review_match` | 55 | 공무원 여비 규정 별표 8건, 원본 렌더를 눈으로 대조 |
| `render_check_pass` | 501 | 독립 렌더러(rhwp) 위치 대조를 자동으로 통과 |
| `candidate_unreviewed` | 1,085 | 미검토 |

- 행마다 새로 넣은 필드: `source{raw, raw_sha256, url, kordoc_text_sha256, source_row, part}`, `columns`, `repeated`, `joined`.
- v2-structure-001-rows-001(B-028)은 이 파일로 대체한다. 바뀐 내용은 B-029 보고서의 "오류 수정"에 있다.

## 3. v5 색인과의 연결: `reports/coordination/B-029/byl-index-links.jsonl`

- 대상: 보수·수당·여비 5개 법령의 별표 56건. sha `90a36c91…`, 검증기 `scripts/link_byl_to_index.py`.
- **키:** 법령명 + MST(raw 경로) + **별표 자체 머리줄의 번호** + caption이다.
  - 말뭉치 제목 "[별표 0006]"은 쓰지 않는다. 원문 번호는 "[별표 6의2]"다.
- **연결 대상:**
  - v5에 같은 id의 law-byl 문서가 있으면 그것.
  - 아니면 같은 MST 법령 본문 `law-go-kr:law:<MST>` 안의 '■ 법령명 [별표 N]' 구간: char span과 그 구간에 걸친 passage id.
- **검증:** caption이 구간에 있고, 행 값의 숫자가 그 구간에 모두 있어야 한다. 병합 반복 값과 삭제 행은 빼고 센다.

| status | 수 | 내용 |
|---|---|---|
| linked | 53 | 공무원 여비 규정 별표 1·2·3·4·5·6의2·8·9 모두 포함 |
| body_empty | 2 | 공무원보수규정 별표 1, 공무원수당 등에 관한 규정 별표 2의4. v5 본문에는 머리줄만 있고 표 내용이 없다. 행이 유일한 표 텍스트다 |
| mismatch | 1 | 지방공무원 수당 등에 관한 규정 별표 11. 숫자 31개 중 30개만 본문에 있다. 연결 보류 |

- **공무원 여비 규정 별표 4·5:** v5에 law-byl 문서(169708653, 169708663)가 없다. 그래서 본문 `law-go-kr:law:290313`의 [별표 4]·[별표 5] 구간으로 연결했다.
  - 별표 4는 행 값 60/60이 일치한다. '6. 삭제' 행은 본문에 없다.
- **본문에 [별표 5]가 두 번 있다** [14715, 15212]·[15212, 15676].
  - 하나는 "사다리차 또는 승강기의 이용료를 포함한다"(현 HWP와 같음), 다른 하나는 "사다리차 이용료 포함"이다.
  - 행 문구와의 4-gram 겹침이 0.945 대 0.821이어서 첫 구간에 연결했다. 두 번째 구간은 옛 문구로 보인다 [추정].
- **A가 확인할 것:**
  - v5 본문 passage의 `path`가 별표 번호를 잘못 단 곳이 있다. 예: 290313:0043·0044는 '별표 12', 0045는 '별표 4의2', 0049·0050은 '별표 1'로 되어 있다. 본문 안에서 인용한 다른 별표 번호를 path로 잡은 것으로 보인다.
  - 그래서 path를 연결 키로 쓰면 안 되고, 위 char span을 쓰는 것이 안전하다.
  - '[별표 7] 삭제 <2010.11.10>'은 본문에만 있다(행 없음).
