# UI-SCOPE-004 → A: 근거 축소 고지용 API 필드 계약안 (B 제안, A 확정 필요)

## 문제 (C-030 L7)

- 문맥 한도 때문에 근거를 줄인 경우, 그 고지가 `context_notes`에 담긴다. 예: "검토된 표 행 9행을 빼었습니다. 이 답은 넣은 행에 한정됩니다."
- 서버는 이 고지를 `scope` 문장 끝에만 붙인다.
- UI는 `scope`를 접힌 "출처/자세히" 안에서만 보여 준다.
- 그래서 일부 근거로 만든 답이 전체 답처럼 보인다.

## 계약안 (A-UI-API-CONTRACT-001 3절에 추가)

| 필드 | 형식 | 뜻 | UI 표시 |
|---|---|---|---|
| `context_notes` (이미 응답에 있음, 계약에는 없던 필드) | `[str]` | 근거 축소의 전체 문구 | 접힌 출처 안. `scope`에 같은 문구가 이미 있으면 다시 넣지 않음 |
| `context_limits` (**신규**) | `{linked_passages:int, table_rows:int, tables:int, own_passages:int}` | `fit_context`가 뺀 개수. 키 4개는 항상 있고, 빼지 않았으면 0 | 답변 본문 바로 아래 한 줄. 0이 아닌 항목만 표시 |

- 키는 `fit_context`의 `dropped` 키와 1:1로 대응한다.

  | `fit_context` `dropped` | `context_limits` 키 |
  |---|---|
  | `linked` | `linked_passages` |
  | `rows` | `table_rows` |
  | `tables` | `tables` |
  | `own` | `own_passages` |

- `context_notes`를 내보내는 두 경로(`/api/ask` 공개 자료 검색과 업로드 문서)에 모두 넣는다.
- 일반 대화(`intent: chat`)에는 넣지 않는다. 넣는다면 모든 값이 0이어야 한다.
- 제안 구현은 A 소유 `scripts/shared_demo.py`에 들어간다. B는 이 파일을 수정하지 않는다.
  ```python
  # fit_context: 반환값에 dropped를 추가
  return selected, notes, dropped
  # 두 호출부
  result.update(checked_chunks=len(selected), context_notes=fit_notes,
                context_limits={'linked_passages': d['linked'], 'table_rows': d['rows'],
                                'tables': d['tables'], 'own_passages': d['own']})
  ```

## UI 동작 (브랜치 `ui-scope-004`, 커밋 `00665bb`, base `3606d1e`, 변경 파일은 `trid/shared_ui/app.js` 하나)

- **한 줄의 위치:** 답변 본문 바로 아래, 상태 배지 위.
- **`context_limits` 표시:** 0이 아닌 항목이 있으면 한 줄을 보인다.
  - 예: `일부 근거만 반영 (표 행 9행 · 다른 별표 구간 3개 제외)`. 390px 화면에서도 한 줄이다.
  - 각 항목의 표기:

    | 키 | 표기 |
    |---|---|
    | `table_rows` | "표 행 N행" |
    | `tables` | "검토된 표 행 전부" |
    | `linked_passages` | "다른 별표 구간 N개" |
    | `own_passages` | "관련도 낮은 구간 N개" |

- **배포 중인 백엔드와도 동작한다.** `context_limits`가 없고 `context_notes`만 있으면 개수 없이 이렇게 표시한다: `일부 근거만 반영 (문맥 한도로 일부 제외)`
  - A 통합 전후 어느 순서로 배포해도 고지가 빠지지 않는다.
- **표시하지 않는 경우:** 두 필드가 모두 비어 있으면 아무것도 표시하지 않는다.
- **긴 문구:** 기술 문구(`scope`, `context_notes` 전체)는 지금처럼 접힌 상태로 둔다.
- **바꾸지 않는 것:**
  - 해설을 검증된 것처럼 표시하지 않는다.
  - 모든 응답에 붙는 상시 경고를 새로 만들지 않는다.

## A에게 요청

1. 위 필드 이름과 형식이 맞는지 확인하거나 수정해 주세요.
   - 이름이 바뀌면 B가 UI 한 곳(`contextLimitLine`)을 고친다.
2. 백엔드에 `context_limits`를 추가하고, 계약 문서 3절에 두 필드를 넣어 주세요.
3. B 브랜치 `ui-scope-004`(`00665bb`)를 통합해 주세요. 백엔드 변경과는 서로 독립이다.
