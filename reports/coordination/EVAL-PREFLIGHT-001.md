# EVAL-PREFLIGHT-001: 오프라인 서비스 평가의 필수 artifact 사전 검사 (재현성)

작성: C(테스트 전담, 세션 01a17c28), 2026-10-31.
- 새 GPU·비용 0.
- 생성 모델 추론 없음. 평가 문항 원문 사용 없음(중립 프로브 4개만).

## 1. 문제
- C-030의 로컬 평가는 배포 커밋을 git worktree로 꺼내 실행했다.
- 서비스 코드가 읽는 **git 미추적 artifact**가 worktree에는 없었다. 코드는 `exists()`로 확인하고 없으면 조용히 기본값으로 넘어간다.
- 그 결과 기관 약칭 "회귀 4건"과 "부동산원 다른 기관 규정 적용"을 잘못 보고했다(C-031 3절 정정).
- 실서비스와 같은 artifact를 읽는지 실행 전에 확인하는 장치가 없었다.

## 2. 도구: `scripts/eval_artifact_preflight.py`
- **수집 방법:**
  - 평가할 코드(worktree)의 `shared_demo.Workspace`를 실제로 만들고, 고정 중립 프로브 4개("안녕하세요", "출장 여비 지급 기준", "연가 사용 원칙", "별표 지급표 금액")로 `select()`를 실행한다.
  - 그동안 `open`/`io.open`/`Path.exists`/`os.path.exists`/`Path.is_file` 호출을 기록한다.
  - 생성 모델(llama-server)은 호출하지 않는다. 임베딩 모델은 CPU 적재만 한다.
- **판정 (코드 루트 아래 경로마다):**

| 상태 | 뜻 | 결과 |
|---|---|---|
| `ok` | 평가 checkout에 있음. sha256·크기·git 추적 여부 기록. 미추적 파일은 배포 저장소 같은 경로와 sha 비교 | 통과 |
| `missing` | 평가 checkout에는 없고 배포 저장소에는 있음(**실수 누락**) | **실패, exit 2** |
| 미추적 sha 불일치 | 있지만 배포본과 내용이 다름 | **실패, exit 2** |
| `absent_intentional` | 배포 저장소에도 없음(그 버전에서 원래 선택적인 artifact. 예: 정형값 이전 색인의 `facts.jsonl`) | 기록만 |

- 코드 루트 밖 경로(색인 디렉터리, 임베딩 가중치 캐시)도 sha256을 기록한다. 그 버전 코드가 아예 읽지 않는 파일은 대상이 아니다.
- **평가 실행 차단:** C의 로컬 평가 스크립트(`runs/c-private/c027_e2e.py`, `c027_retrieval.py`)가 시작할 때 이 검사를 돌린다. 실패하면 **exit 3으로 문항을 하나도 실행하지 않고** 종료한다. manifest는 실행마다 남는다.

## 3. 검증 (실제 실행)

| 경우 | 코드·색인 | 결과 |
|---|---|---|
| **별칭 등 누락 fixture**(새 worktree, 데이터 연결 없음) | 9256df4 + v5 | **fail, exit 2.** missing 4건: `data/agency-aliases-v1/aliases.jsonl`, `data/corpus-inventory/alio-agencies.json`, `data/law-byl/v2-structure-003-rows-001/queryable-rows.jsonl`, `data/official-abbreviations/abbreviations.jsonl` |
| 같은 fixture로 **실제 평가 실행** | `c027_e2e.py` | **"PREFLIGHT FAILED (exit 2); evaluation not run" → exit 3.** 결과 파일을 쓰지 않음. 모델 서버에 접속하지 않음(일부러 잘못된 포트 지정) |
| **내용 변경 fixture**(별칭 파일 마지막 줄 삭제) | 9256df4 + v5 | **fail, exit 2**, untracked_differs: `aliases.jsonl` |
| 이전 버전, 데이터 연결 없음 | 242dbf2 + v3 | **fail.** missing은 `alio-agencies.json` 1건뿐. 이 버전은 별칭·약칭·검토 행을 읽지 않으므로 누락으로 잡지 않음(버전별 구분). `v3/facts.jsonl`은 absent_intentional |
| **복구 후 성공 경로**(데이터 연결) | 9256df4 + v5 / v3 | **pass**, 39개 ok. 미추적 6개 파일 sha가 모두 배포 저장소와 같음 |
| 같은 성공 경로로 실제 실행 | `c027_retrieval.py` v5 | 사전 검사 통과 뒤 정상 실행. 결과가 C-031과 같음(hybrid doc@1 12, MRR 0.643). 재현 확인 |
| C-031 재실행 환경 | 242dbf2 v3, 679004f v3·v5 | 모두 pass(32 / 37 / 37개). **C-031 정정 결과는 올바른 artifact로 측정됐음을 사후 확인** |
| 배포 저장소 자체 | aa106e4 + v5 | pass, 39개 |

**미추적 필수 artifact (9256df4가 읽는 것, sha 앞 12자리):**
- `data/agency-aliases-v1/aliases.jsonl` `4e75f11f50f2`, `manifest.json` `947c2e553630`
- `data/corpus-inventory/alio-agencies.json` `5e79c5339f9f`
- `data/law-byl/v2-structure-003-rows-001/queryable-rows.jsonl` `6413a1ccfbf3`
- `data/official-abbreviations/abbreviations.jsonl` `ae41cb3ccc64`, `manifest.json` `6464d6f363de`

**추가로 드러난 사실:**
- C-030 로컬 실행에는 별칭뿐 아니라 **검토된 별표 행(queryable-rows)과 공식 약칭 파일도 없었다.**
- 그러므로 C-030의 679004f 로컬 결정·해설 수치(1.2·1.3절)는 이 artifact가 빠진 상태의 값이다. 신뢰할 수 있는 값은 C-031의 재실행 값이다.
- C-030 정정 절에 이 내용을 덧붙였다.

## 4. 산출물
- 도구: `scripts/eval_artifact_preflight.py`
- manifest(경로, sha, 프로브만 담고 평가 원문 없음): `reports/coordination/EVAL-PREFLIGHT-001/preflight-*.json`
  - fixture 3종(missing, modified, 242-v3)
  - 성공 경로 5종(9256df4 v3·v5, 242dbf2 v3, 679004f v3·v5)
  - 배포 저장소 1종
- fixture worktree(`pf-missing`, `pf-242`)와 C-030 중단 실행용 `0ee996d` worktree는 삭제했다. 남은 worktree(242dbf2·679004f·9256df4)는 데이터를 연결한 검증 통과 상태다.

## 5. 한계
- 추적 대상은 `open`, `exists`, `is_file` 호출이다. 디렉터리 목록(`listdir`/`glob`)으로만 고르는 파일은 놓칠 수 있다. 다만 디렉터리 자체의 존재는 기록된다.
- 프로브 4개로 닿지 않는 지연 적재 경로는 이 검사에 잡히지 않는다. 예: ALIO C0150 기관 질문에서만 여는 기관 별도 색인(`inst-reb-002`).
  - C의 로컬 평가는 그 경로를 쓰지 않고 실 UI로만 검증하므로 현재 평가에는 영향이 없다.
  - 그런 경로를 로컬 평가에 넣을 때는 해당 프로브를 추가해야 한다.
- 이 도구가 막는 것은 오프라인 재현 평가다. 실 UI 평가는 배포된 서버 자체를 쓰므로 해당하지 않는다.

SUPERVISION_REPORT_READY EVAL-PREFLIGHT-001
