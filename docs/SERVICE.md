# trid 서비스 계층 (개발 중 · 현재 버전 0.1)

폐쇄망 서버 한 대에서 기관 문서를 근거로 답하는 패키지. `trid/service/`가 전부이고, 모델 밖에서 권한 필터와 감사 로그를 맡는다.

```
문서 폴더 ──ingest──▶ Document(sha256, acl, text, passages, refs) ──index──▶ BM25 + 구조 그래프 (폴더 저장)
질문(user) ──ACL──▶ BM25 top-k ──그래프 확장(참조·상위 조)──▶ rag_prompt [D1..D5] ──모델──▶ 값 [Dn] | 확인 불가 | 근거 충돌
                                                                              └──▶ 감사 로그 (문서 sha256 + 글자 위치)
```

답은 인용한 문단의 글자열만 될 수 있다. 백엔드가 어떤 문자열을 내놓든 엔진이 그 문단에서 다시 찾아 문서 안 글자 위치로 바꾸고, 못 찾으면 답을 내지 않고 오류를 낸다 (`RuntimeError`).

## 설치 (오프라인 서버)

네트워크가 되는 곳에서 wheel을 모은다. 전부 CPU용이고 외부 가중치 다운로드는 없다.

```sh
uv pip download --python 3.12 -d wheels '.[service]'      # torch, tokenizers, safetensors, numpy, bm25s, kiwipiepy(+model), ko-pii
```

서버로 옮길 것: `wheels/`, 이 저장소(`trid/` 패키지), 모델 폴더(`config.json`, `model.safetensors`, `tokenizer.json`, 있으면 `training.json`), 필요하면 `rhwp` 바이너리(.hwp)와 poppler `pdftotext`(.pdf). 서버에서:

```sh
uv venv --python 3.12 .venv && uv pip install --no-index --find-links wheels '.[service]'
```

의존성과 라이선스는 `pyproject.toml`의 `service` 항목에 적혀 있다 (bm25s MIT, kiwipiepy Apache-2.0, ko-pii MIT, rhwp MIT, poppler GPL은 바이너리 호출만).

## 색인

```sh
.venv/bin/trid ingest docs/ --index index/        # txt md hwp hwpx pdf, 하위 폴더 포함
```

- 문서마다 원본 파일 sha256을 기록한다. 텍스트는 NFC 정규화 후 `trid.korean_quality.redact`로 전화·주민번호·계좌 등을 `[전화]` 같은 토큰으로 바꾼다 (`--no-redact`로 끈다). 색인에 저장되는 글자 위치는 이 정규화·치환된 텍스트 기준이다 (`index/documents.json`에 그대로 있다).
- 규정(제N조 heading이 둘 이상)은 편/장/절/조/항/호/목/별표/서식/부칙 구조로 자른다. 문단 하나는 550자 이하이고, 긴 조는 항 단위, 다시 호·목·줄·문장 단위로 쪼개 앞뒤를 550자까지 묶는다. 경로 예: `("출장여비규정", "제2장", "제12조", "②~③")`. 보도자료 같은 일반 문서는 문단을 550자까지 묶는다.
- 참조는 문서 안에서 해석한다: `제13조에 따른`, `제12조제2항`, `별표 2`, `별지 제1호서식`. `「○○규정」 제5조`는 제목이 같은 다른 문서로 잇는다. 검색은 BM25로 찾은 문단에서 참조 문단과 그 조의 첫 문단(상위 조 문맥)을 1홉 더 가져온다.

### 권한 (deny by default)

문서 태그와 사용자 태그가 하나라도 겹쳐야 보인다. 태그가 없는 문서는 아무도 못 보고, 모르는 사용자는 아무것도 못 본다.

```
docs/acl.json                      {"default": ["staff"], "files": {"급여규정.txt": ["hr"]}}   # 그 폴더 아래 전체, files는 폴더 기준 상대경로
docs/급여규정.txt.acl.json          {"acl": ["hr"]}                                           # 파일 옆 사이드카가 우선
index/users.json                   {"kim": ["staff"], "hr_lee": {"tags": ["staff", "hr"], "token_sha256": "…"}}   # 목록형은 CLI 전용
```

## 질문

```sh
.venv/bin/trid ask --index index/ --run models/trid-reader --user kim '숙박비의 상한은 1박당 얼마인가?'
.venv/bin/trid serve --index index/ --run models/trid-reader            # 127.0.0.1:8321
.venv/bin/trid token --users index/users.json --user kim      # 토큰은 한 번만 출력; users.json에는 sha256만 저장
curl -s -X POST localhost:8321/ask -H 'Authorization: Bearer <token>' -H 'Content-Type: application/json' -d '{"query": "..."}'
curl -s localhost:8321/health
```

- `--run`의 `training.json`이 `extractive_reader_v1`이면 추출 헤드(`trid.extractive`, forward 1회), 아니면 구간 단위 점수화(`trid.span_scoring`, 후보 수백~수천 개를 전부 점수화하므로 91M·CPU에서 질문당 수 초)를 쓴다.
- 옵션: `--k` BM25 상위 문단 수(기본 3), `--hops` 그래프 확장 홉(기본 1, 0이면 BM25만), `--threads` torch 스레드(기본 4), `--audit` 감사 로그 경로(기본 `index/audit.jsonl`), `--host`/`--port`.
- 프롬프트에는 문단을 최대 5개 넣는다. 추출 헤드는 답 토큰을 생성하지 않아 여유분 0으로 계산한다. 구간 점수화는 실제 후보 답을 해당 토크나이저로 인코딩한 최대 길이와 EOS 1토큰을 확보한다. 고정 48토큰 가정은 사용하지 않는다.
- 응답 JSON: `text`(모델 출력 그대로), `kind`(`answer`·`확인 불가`·`근거 충돌`), `value`, `source`(문단 id·경로·문서 안 글자 범위), `char_offset`(문서 텍스트 기준), `score`, `candidates`(프롬프트에 준 문단과 BM25 점수, 그래프로 온 문단은 `ref←<id>`·`head←<id>`), `seconds`.
- 표·목록의 줄바꿈과 공백을 문단 원문 그대로 전달한다. 추출 헤드가 고른 정확한 글자 위치를 감사 로그까지 전달하므로, 같은 값이 반복돼도 첫 출현으로 바꾸지 않는다. 문자열만 고르는 구간 점수화 백엔드는 첫 출현을 기록한다.
- 예약 제어 토큰이나 `\n[Dn] `처럼 근거 경계로 오인될 내용을 가진 문단은 프롬프트에서 제외한다. 제외 이유를 별도 응답 필드로 제공하지는 않는다. 질문의 공백은 한 칸으로 정규화하며 예약 제어 토큰이 든 질문은 거부한다. 이 처리는 직렬화 경계 보호이지 문서 내 악성 지시의 의미를 판별하는 방어는 아니다.
- `score`는 보정된 확률이 아니다. 답 있음의 구간 점수와 보류의 분류 확률을 같은 임계값으로 비교하지 않는다.

## 감사 로그

`index/audit.jsonl`에 한 줄씩 덧붙인다. 문서 본문이나 질문 원문은 쓰지 않는다.

```json
{"ts": "2026-10-28T05:56:42.258+00:00", "user": "kim", "query_sha256": "49ac…", 
 "candidate_docs": [{"doc_id": "출장여비규정", "sha256": "6480…"}],
 "candidates": [["출장여비규정:0008", 2.2158, null], ["출장여비규정:0004", 1.557, null]],
 "kind": "answer", "score": -25.33,
 "source": {"doc_id": "출장여비규정", "sha256": "6480…", "passage_id": "출장여비규정:0005", "path": ["출장여비규정", "제2장", "제13조"], "char_start": 339, "char_end": 340},
 "seconds": 11.88}
```

검증은 `index/documents.json`의 해당 문서 `text[char_start:char_end]`로 한다.

## 네트워크

색인·검색·추론 어디에도 소켓을 열지 않는다 (`tests/test_service_engine.py::OfflineTest`가 `socket.socket`을 막고 전체 경로를 돈다). `trid serve`만 지정한 주소에 바인드한다. `trid.cli`는 `HF_HUB_OFFLINE=1`을 강제한다.

## 테스트

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_service_*.py'    # 23 tests, CPU, 2초
```

## 측정 (2026-10-28, M-series CPU 4스레드, `/tmp/trid-base-snap` 91M 사전학습 체크포인트 = 구간 점수화 백엔드, 가상 문서 2개)

| 질문 | 근거 문단 | 답 | 시간 |
|---|---|---|---|
| 숙박비의 상한은 1박당 얼마인가? (kim) | 별표 2 · 제12조 · 제13조 | `제 [D3]` (제13조, 문자 339) | 11.9초 |
| 서울특별시 출장의 숙박비 상한은 1박당 얼마인가? (HTTP) | 별표 2 · 제13조 · 제12조 | `제 [D3]` (제12조, 문자 201) | 10.9초 |
| 급여는 매월 며칠에 지급하나? (hr_lee) | 급여규정 제3조 · 제2조 · 제1조 | `제 [D2]` | 1.6초 |
| 같은 질문, 권한 없는 사용자 | 없음 | `확인 불가` (모델 호출 없음) | 0.0초 |

사전학습만 된 체크포인트라 값은 틀리지만(길이 편향으로 한 글자 구간), 어느 경우에도 답은 인용 문단의 글자열이고 감사 로그에 위치가 남는다. 시간은 후보 수(문단 길이)에 비례한다. 추출 헤드 run을 `--run`에 주면 forward 1회로 끝난다.
