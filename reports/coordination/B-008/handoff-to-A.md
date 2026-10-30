# B → A 인계 명세: 국회 회의록 clean-v5, 정책브리핑 보도자료 clean-v3

작성: B, 2026-10-30 KST. 근거 보고: `reports/coordination/B-008.md`.

이 문서는 **학습 후보**를 넘기는 명세다. 다음은 이 문서의 범위가 아니다.
- 학습 투입 승인. A003 입력은 바꾸지 않는다. 투입 여부는 총괄이 정하고, 혼합 비율은 A가 정한다.
- 오염 없음 보증. 7절의 한계를 참고한다.

## 1. 입력 (S3, 버킷 `trid-training-148761639313`, ap-northeast-2)

| 출처 | 버전 manifest | manifest sha256 | 학습 후보 샤드 | 행 | 문자 수 |
|---|---|---|---|---|---|
| 국회 회의록 | `corpus/state/assembly-minutes/clean-v5.manifest.json` | `06e23d83c87e7917c180b81a84014f36ed6852873aea8838f2adbe7874a7a2dc` | `corpus/clean/assembly-minutes/clean-v5/docs-000NN.jsonl.gz` (22개) | 1,684 | 130,962,603 |
| 정책브리핑 보도자료 | `corpus/state/korea-kr-briefing/clean-v3.manifest.json` | `5aba0fadd30cd12b58c3ac91d29d6101917d2f6732068d2f9ab1eae701f086e3`, `clean-v3.done` 내용과 같음 | `corpus/clean/korea-kr-briefing/clean-v3/docs-000NN.jsonl.gz` | 281,709 | 1,047,841,584 |

샤드 읽기:
- 각 manifest의 `shards[]`에 key, sha256, bytes, rows가 있다.
- S3 객체는 SHA-256 체크섬을 붙여 올렸다. `head_object(ChecksumMode='ENABLED')`의 `ChecksumSHA256`(base64)과 manifest의 sha256을 비교한다.
- **`quarantine-*.jsonl.gz`는 학습에 쓰지 않는다.** 같은 디렉터리에 있지만 학습 후보가 아니다.

완료 표시:
- **정책브리핑:** `clean-v3.done`이 있다.
- **국회:** `_DONE` 표시가 없다.
  - 22대 목록 2,142회의 중 2건(52354, 52713)을 받지 못했다. 서버가 HTTP 400(24바이트)으로 응답했고, 1회 재시도해도 같았다.
  - 나머지 2,140건은 모두 행이 있다. manifest `not_collected` 필드에 기록했다.
  - 이 2건이 없는 상태로 쓸지는 A·총괄이 판단한다.

사용하지 않는 버전 (덮어쓰지 않고 그대로 남겨 둔다):

| 출처 | 버전 | 사용하지 않는 이유 |
|---|---|---|
| 국회 | clean-v1, v2 | 이전 결함이 있다. B-006·B-007 참고 |
| 국회 | clean-v3 | 1,000회의. 코드 해시 보완 파일이 `clean-v3/provenance-supplement.json`에 있다 |
| 국회 | clean-v4 | 1,140회의를 받은 중간본이다. 연석회의 제목을 찾지 못해 뷰어 UI가 남은 행이 3건 있다 |
| 정책브리핑 | clean-v1 | 예약 부분 겹침을 격리하지 않았다 |
| 정책브리핑 | clean-v2 | URL 훼손 판정의 오탐 7건을 과격리했다 |

## 2. 코드와 재현

| 대상 | 코드 | sha256 | 커밋 |
|---|---|---|---|
| 국회 clean-v5 (22개 샤드 모두 같음) | `scripts/collect_assembly_minutes.py` | `aa6eb1d4b58f939c53d8eb1595e845df8bea748b851be88685f298eb32688895` | `940a83c` |
| 국회 + 정책브리핑 공통 치환기 | `trid/korean_quality.py` | `4d732cf102821152e6dd3ea3bf3e37f4874923b7903129967eb4ed97c11a5c97` | `940a83c` |
| 정책브리핑 clean-v3 | `scripts/build_press_clean.py` | `52e63c724af64b5e1e0a811037b300a3ab7c95bce8634b96211d8d318924e618` | `d7b4b52` |
| 정책브리핑 입력 | `data/korea-kr-briefing-source/docs.jsonl` | `6e16fa1d33f60428c49d6124940ac24a7d4827dfe8507a1ad1b56573b49082ea` | 281,995행 |
| 국회 공식 목록 메타 | `corpus/state/assembly-minutes/official-list-meta.json` | `406246b2231b39291079c1e26aca300666ff6106e055c822f5937abbd3c70ca2` | 머리말 미해석 회의의 제목·날짜 |
| 국회 원문 | `corpus/raw/assembly-minutes/<id>.html.gz` | 행 `meta.raw_gzip_sha256` | 2,140건 모두 현재 저장 객체와 일치 |

재현 절차:
- **국회:** `collect_assembly_minutes.py reprocess --index pilot22-index.json` 다음 `aggregate`. 저장된 원문에서 다시 만들며, 사이트에 요청하지 않는다.
- **정책브리핑:** `build_press_clean.py --version <새 버전> --reserve-quarantine reports/coordination/C-008/quarantine-docs.txt reports/coordination/B-008/press-reserve-partial-quarantine.json`.

ko-pii는 `1.22.15`로 고정한다.

## 3. 사용권

| 출처 | 행 `license` | 근거 | 지켜야 할 것 |
|---|---|---|---|
| 국회 회의록 | `KOGL-Type1` | `https://record.assembly.go.kr/assembly/` 하단의 공공누리 제1유형 표시와 `https://open.assembly.go.kr/portal/policy/copyRightPage.do`. `clean-v5/provenance.json`, `ATTRIBUTION.txt`에도 기록 | 행 `attribution`(출처: 국회사무처 국회회의록, 원문 URL)을 보존한다 |
| 정책브리핑 | `KOGL-Type1` (281,709행 모두) | 원천 `docs.jsonl`의 행별 값을 그대로 옮겼다. B-008에서 행마다 원문 표시를 다시 확인하지는 않았다 [추정: 원천 수집 시점 판정을 따름] | 행 `attribution`(작성 부처, 정책브리핑 출처)을 보존한다 |

## 4. B가 실제로 확인한 것 (2026-10-30)

| 항목 | 국회 clean-v5 (docs+격리 2,140행) | 정책브리핑 clean-v3 (docs+격리 281,808행) |
|---|---|---|
| 샤드 S3 체크섬 = manifest | 44/44 | 16/16 |
| 행 `sha256` = `text` 해시 | 2,140/2,140 | 281,808/281,808 |
| 원문 해시 = 저장 원문 | 2,140/2,140 | 해당 없음 (입력 파일 해시로 고정) |
| 사용권·출처 URL 형식 | 2,140/2,140 | 281,709 docs 모두 KOGL-Type1, www.korea.kr |
| `split` 재계산 일치 | 2,140/2,140 | 이번에 확인하지 않음 |
| 격리 플래그 ↔ 샤드 위치 | 2,140/2,140 | docs에 격리 플래그 0, 격리 샤드 99행 모두 격리 플래그 보유 |
| id 중복 | 0 | 0 |
| 예약 148계열 서명 검사 (`check_reserved_overlap.py`) | 일치 0, 부분 검토 0 | docs 281,709행: 일치 0, 부분 검토 0 |

검사 스크립트(일회용): `/tmp/trid-corpus/verify_candidate.py`. 결과는 `reports/coordination/B-008/verify-assembly-clean-v5.json`에 있다.

## 5. 격리 (학습 금지, 격리 샤드에만 있음)

| 출처 | 사유 | 행 | 판정 방법 |
|---|---|---|---|
| 국회 | `private_person_names` | 456 (감사 198, 위원회 258) | 청원인(`청원(○○○ 외 N인`)이나 증인·참고인 등 사인 발언자가 나오면 회의 전체를 격리 |
| 정책브리핑 | `dev_eval_source` | 50 | 아래 6절 |
| 정책브리핑 | `reserved_overlap` | 46 | 예약 부분 겹침. 7절 |
| 정책브리핑 | `pii_redaction_inside_url` | 3 | 치환 라벨이 URL 범위 안에 들어가 URL이 훼손된 행 |

URL 안 치환 행은 두 문제로 나눠 판단한다.
- **개인정보 측면:** 치환은 이미 됐다. 원래 값이 실제 전화번호였는지 URL id였는지는 판정하지 않았다.
- **내용 훼손 측면:** URL이 깨졌다.
- 처리: 숫자 모양만 보고 URL을 예외로 두지 않고, 행 전체를 격리했다.
- 3건: 156333888(`mId=`), 156477939(`https://[전화].`), 156762411(Zoom 링크의 `omn=`).
- 156762411에는 Zoom 회의 번호와 `pwd=` 값이 그대로 남아 있다. 격리 행이므로 학습에 들어가지는 않는다.
- 국회 2,140행 중 해당 행은 0건이다.

## 6. 개발 평가 제외 50건의 범위와 방법

대상 파일:
- `data/korean-rag-eval-v1/test.jsonl`
- `data/institution-qa-frozen-01/source-docs.jsonl`
- `data/institution-qa-frozen-01/test.jsonl`
- `data/public-document-sft-source/annotations.jsonl`
- `data/public-document-sft-source/documents.jsonl`

일치 방법 (둘 중 하나):
- 평가 파일 어디에든 `korea-kr-press-<newsId>`가 나온다.
- 문서 본문(치환 전 또는 후)의 공백을 모두 지운 문자열이, 평가 행에서 200자 이상인 문자열 필드의 공백 제거 문자열과 정확히 같다.

이 방법으로 잡지 못하는 것:
- 부분 인용
- 다른 출처에 같은 글이 있는 경우
- 위 5개 파일 밖의 평가셋

따라서 A는 tokens 단계에서 기존 개발 평가 8-gram 필터와 출처 간 중복 제거를 그대로 적용한다. 국회 회의록에는 이 제외를 적용하지 않았다. 국회 회의록은 위 평가 파일들의 원천이 아니다 [추정: 평가 파일 출처 필드 기준. 본문 대조는 하지 않음].

## 7. 예약 오염 검사의 한계

사용한 서명: `data/benchmark-reserve-001/signatures-partial-148.jsonl` (sha `0ed3cc07…95e8`). 예약 1,000계열 중 **148계열**뿐이다. 나머지 **852계열은 미확보**라 검사할 수 없었다.

결과:
- **두 코퍼스 전체에서 완전 일치(ID·URL·exact·near)는 0건이다.**
  - 국회는 2,140행 전부.
  - 정책브리핑은 clean-v1 전체 281,808행(docs+격리).
- **부분 검토는 정책브리핑 46건, 국회 0건이다.** 46건은 모두 clean-v3에서 격리했다.
  - C-008이 지정한 `korea-kr-press-156552370`도 여기에 들어 있다. alio-rule 12706·33636과 8-gram 86·79개를 공유한다.

부분 검토 46건의 분류 (`overlap-press-partials-classified.json`):
- 방법: 공유 8-gram을 법제처 현행 법령 본문(`data/law-ko-source`, 384,716건)과 대조했다.

| 분류 | 행 | 내용 |
|---|---|---|
| 공유 8-gram의 80% 이상이 법령 본문에도 있음 | 31 | 개인정보보호법·청탁금지법 정의 조항 인용 등. 156552370은 공유분 100%가 법령 문구다 |
| 20~80%만 법령 본문에 있음 | 11 | 법령 문구 + 징계양정 기준표·표준 서식 등 |
| 20% 미만만 법령 본문에 있음 | 4 | 156213427, 156400548 (블라인드 채용 입사지원서 문구), 156485214 (안전사고 분석 조항), 156613983 (정보보호 점검표). 공공기관 공통 서식·기준 문구로 보인다 [추정] |

이 분류에 대해:
- 법령 문구 공유라고 해서 안전하다는 근거가 되지는 않는다. 46건 모두 격리했다. 법령 문구 공유는 겹침의 원인 설명일 뿐이다.
- **"일치 0건"은 최종 시험이 오염되지 않았다는 뜻이 아니다.** 852계열은 서명이 없다.
- 같은 법령·표준 서식을 인용한 다른 문서도 예약 문서와 부분적으로 겹칠 수 있다. 서명 148계열 기준 임계값 아래의 겹침은 이 도구가 보고하지 않는다.

A가 할 일:
- 서명이 늘어날 때마다 tokens 단계에서 `check_reserved_overlap.py`를 다시 돌린다.
- 부분 검토 행은 학습에서 뺀다. C-007 선례를 따른다.

## 8. 이전 trid 학습 포함 여부

**정책브리핑:**
- docs 중 **38,822행**에 `prior_trid_training` 플래그가 있다. 이 문서들은 다음 파일들에 이미 들어갔던 문서다.
  - `data/mix-v3-dry/train.jsonl`
  - `trid-cpt-kanana-100m/train.jsonl`
  - `public-document-sft-01/train.jsonl`
  - `public-dialogue-shared-ctx-01/train.jsonl`
  - `public-dialogue-mix-01~06/train.jsonl`
- 판정 기준은 `korea-kr-press-<id>` 일치다.
- B는 이 행을 지우지 않았다. 반복 노출 비율은 A가 정한다.

**국회 회의록:**
- 이전 학습 데이터 확인 플래그가 없다.
- 국회 v0 파일럿(버전 없는 clean)은 계약상 학습에 쓰지 않았다 [추정: 이번에 학습 파일을 다시 대조하지 않음].

## 9. 문서 분할

- `split = sha256("split:"+group)` 상위 32비트 % 10000 기준: test 0.5%, valid 0.5%, 나머지 train.
- 분할 단위 `group`:
  - 국회: `assembly-minutes:<회의 id>`
  - 정책브리핑: `korea-kr-briefing:<newsId>`
- 분포:
  - 국회 docs: train 1,665 / valid 9 / test 10
  - 정책브리핑 docs: train 278,915 / valid 1,347 / test 1,447
- **개발용 분할이며 최종 시험이 아니다.** 기관·계열·시점 단위 보류는 총괄 목록으로 A가 tokens 단계에서 적용한다.
- 출처 간 중복(예: 같은 보도자료가 FineWeb에 있는 경우)은 이 분할로 막히지 않는다.

## 10. 알려진 남은 위험

- **원문 덮어쓰기:** 국회 원문 10건(51946, 52032, 52038, 52167, 52171, 52600, 52757, 55606, 55917, 56111)의 바이트가 2026-10-30 09:33 KST 재다운로드로 덮어써졌다.
  - 버킷 버전 관리가 꺼져 있어 이전 바이트는 복구할 수 없다.
  - 새 원문으로 만든 clean-v5 본문은 clean-v3 본문과 10건 모두 같다.
- **치환 오탐:** 전화 치환 오탐이 남아 있을 수 있다. URL 밖의 긴 숫자 id가 `[전화]`로 바뀌는 경우다. 이 경우 내용이 손상될 뿐 개인정보 노출은 없다.
- **공인 이름:** 국회 회의록의 의원·국무위원·공무원 이름은 공적 직무 발언자로 보고 남겼다. 사인 이름이 나오는 회의는 회의 전체를 격리했다. 다만 패턴(`청원(○○○ 외`, 증인·참고인 표기) 밖의 사인 언급은 놓쳤을 수 있다.
- **국회 남은 개인정보 치환 흔적:** `pii_redacted` 플래그가 docs 10행, 격리 6행에 있다. 라벨 합계는 docs와 격리를 합쳐 EMAIL 4, PHONE 61이다.
