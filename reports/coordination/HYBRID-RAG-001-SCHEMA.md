# HYBRID-RAG-001 ↔ API-S3-001 레코드·스냅샷 계약 (A 제안, B 합의 요청)

- **작성:** A (01a18739), 2026-10-30
- **소유:**
  - **B:** 실 API 수집 → private S3 raw + normalized + manifest(API-S3-001)
  - **A:** 임베딩, hybrid 검색, 서비스 통합(HYBRID-RAG-001)
- **변경 절차:** 이 문서를 고쳐 커밋한다. 필드를 추가해도 A 쪽은 깨지지 않는다. 이름 변경이나 삭제는 양쪽이 합의한다.

## 1. 스냅샷 (S3 prefix 하나 = 로컬 디렉터리 하나)

```
<snapshot_id>/                       예: api-20261030T1800Z
  manifest.json                      {snapshot_id, created_at, producer:"B/API-S3-001", sources:[{provider, endpoint, retrieved_at, records, license}],
                                      files:[{path, sha256, records}], records_total}
  normalized/records-00000.jsonl     레코드(아래 2절), 파일당 ≤ 50,000줄
  raw/...                            원 응답(A는 읽지 않음, 감사용)
```

- **해시:** A는 `manifest.json`의 sha256을 **snapshot hash**로 기록한다. 각 파일 sha256이 manifest와 다르면 스냅샷 전체를 거부한다.
- **개발·폐쇄망:** A는 `aws s3 sync`로 받은 로컬 사본(`runs/api-snapshots/<snapshot_id>/`)만 읽는다. 서비스 실행 중에는 외부와 통신하지 않는다. 모델·인덱스·스냅샷 번들만으로 검색이 된다.

## 2. normalized 레코드 (JSON 한 줄)

| 필드 | 필수 | 설명 |
|---|---|---|
| `source_id` | ✓ | 안정 ID `<provider>:<dataset>:<record>` (예: `alio:C0150:travel-rule:2025`). 버전이 바뀌어도 같은 문서면 같은 값 |
| `version` | ✓ | 제공처 개정·수정 식별자. 없으면 `retrieved_at` |
| `content_hash` | ✓ | `text`(NFC, 앞뒤 공백 제거)의 sha256. A는 이 값이 다를 때만 다시 임베딩한다 |
| `kind` | ✓ | `rule`(규정·지침·고시), `manual`(편람), `law`(법령·별표), `interpretation`(법령해석·질의회신), `faq`, `qa_board`(민원 질의응답), `statistic`(통계·공시 값), `press` |
| `title` | ✓ | 문서 제목 |
| `issuer` | ✓ | **발행기관** `{id, name}`. id는 ALIO `apbaId`(예: `C0150`)나 행정기관코드. 없으면 null |
| `applies_to` | | **규정 적용 대상** `{scope:"institution"\|"all_public_institutions"\|"central_government"\|"local_government"\|"unknown", ids:[…]}`. 발행기관과 적용기관을 구분한다. 모르면 `unknown` |
| `dates` | ✓ | `{published, effective_from, effective_to, retrieved_at}` (ISO 날짜, 없으면 null) |
| `rights` | ✓ | `{license, url, restrictions}`. 제한이 있으면 A는 색인하지 않는다 |
| `location` | ✓ | `{url, page, article_path:[…], char_start, char_end}`. 원문 위치 |
| `text` | ✓ | 검색·인용할 본문(원문 그대로) |
| `structured_facts` | | 정형 값 목록. 평균·합계는 A가 이것만 집계한다. 임베딩 결과로 계산하지 않는다. 형식: `[{metric, value(number), unit, period:{from,to}, subject:{id,name}, basis:"regulation_limit"\|"actual_spending"\|"statistic", location}]` |

- **`basis`:** 규정상 지급 기준(`regulation_limit`)과 실제 지출(`actual_spending`)은 반드시 구분해 넣는다. "평균 식비" 같은 질문에서 이 둘을 섞지 않기 위한 필드다.
- **기관 ID:** 공식 기관 목록은 `data/corpus-inventory/alio-agencies.json`(ALIO 355개, 2026-10 수집)이다. A는 이 목록과 기존 색인 메타데이터만으로 기관 별칭(예: 한국부동산원 ↔ 부동산원)을 연결한다.

## 3. A가 하는 일 (HYBRID-RAG-001)
- **임베딩:** 기존 승인 코퍼스(`runs/workspace-index/v2`, 101,014구간)와 스냅샷 레코드를 `BAAI/bge-m3`로 임베딩한다. 로컬 HF snapshot을 쓰며 GPU 작업은 새로 만들지 않는다.
  - 저장소는 내용 해시로 주소를 매긴다(`runs/embeddings/bge-m3`).
  - 중단 후 재시작하면 이어서 진행하고, 변경된 부분만 갱신한다.
- **검색:** 키워드(BM25)와 의미(코사인) 점수를 결합한다. 기관·시점 필터를 유지하고, 규정 질문에서는 권위 자료(규정·편람·법령)를 민원 질의응답보다 우선한다.
- **근거 적합성:** 생성 전에 근거가 질문에 맞는지 검사한다. 맞는 근거가 없으면 무엇이 없는지 말하고, 무관한 문서는 나열하지 않는다.

## 4. B에 요청
1. 이 스키마로 첫 스냅샷을 만든다. 출장·여비 규정이 있는 기관을 우선한다(한국부동산원 포함).
2. 스냅샷 경로와 manifest sha256을 `reports/coordination/API-S3-001.md`에 남긴다.
3. A가 읽을 수 있는 로컬 사본 위치를 알려 준다.
4. S3 권한이 막히면 같은 형식의 로컬 스냅샷으로 먼저 넘긴다. 이 경우 A는 "S3 완료"라고 주장하지 않는다.
