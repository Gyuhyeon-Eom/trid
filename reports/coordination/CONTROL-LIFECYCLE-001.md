# CONTROL-LIFECYCLE-001: stop 후 llama-server 고아 프로세스 수정

**상태:** 패치와 격리 smoke를 마쳤다. 실서비스에는 아직 적용하지 않았다(배포 보류).

- 실서비스는 C 평가 중이라 재시작하지 않았다.
  - 실서비스: Worker `506ecc2b`, 포트 51873과 51912/51917/51918
  - C 평가: 포트 51923
- 위 실서비스·C 평가 프로세스와 포트는 건드리지 않았다. smoke 전후로 같은 PID가 그대로 떠 있는 것을 확인했다.
- 추가 AWS GPU 0, 학습 0. 모델·평가·훈련 sha는 바꾸지 않았다.
- B-031/B-032 학습 후보(v2r1)는 받기만 했고 학습하지 않았다.

## 원인 (격리 재현)

- **재현 방법:** 실서비스와 같은 `shared_demo.py`를 격리 포트로 띄웠다.
  - 포트 51975, llama 51962/51967/51968
  - versions 파일 `/tmp/versions-lifecycle.json`, 런타임 `/tmp/trid-lc-raw`
- 모델 서버만 SIGKILL로 죽이면 llama-server 3개가 부모 PID 1인 고아로 남아 포트를 계속 잡았다(PID 74581/74595/74602). 실사이트에서 관찰한 증상과 같다.
- **경로:** 이전 supervisor의 `stop_child`는 모델 서버에 SIGTERM을 보내고 5초를 기다린 뒤 SIGKILL한다. 모델 서버의 `State.close()`는 llama 서버를 하나씩 종료하며 서버당 최대 5초를 기다렸다. 그래서 종료가 5초를 넘기면 모델 서버가 먼저 죽고 llama 서버가 고아로 남는다.
- **실사이트 발생:** 23:08과 23:11 두 번 발생했다. 51918, 51917·51918이 고아로 남았다.
- **확인하지 못한 것:** 부하 없는 격리 정지는 4초 안에 끝나 이 경로를 타지 않았다(1회 확인). 실사이트에서 정확히 무엇이 종료를 늦췄는지는 확인하지 못했다.
- **정리:** 재현으로 생긴 고아는 PID·시작 시각·격리 포트를 대조한 뒤에만 종료했다.

## 수정

**`scripts/share_demo_control.py`**

모델 서버를 자체 프로세스 그룹으로 실행하도록 바꿨다(`start_new_session`). llama 서버들이 이 그룹을 물려받는다.

| 동작 | 변경 내용 |
|---|---|
| supervisor 종료 | `stop_group`: 그룹 전체에 SIGTERM → 최대 20초 대기 → 남으면 SIGKILL → 5초 대기. 모델 서버가 먼저 죽어도 같은 그룹의 llama 서버까지 정리된다. 남은 것은 `state.json`의 `stop_leftover`에 기록한다 |
| 기록 | `state.json`에 `backend_pgid`와 그룹 리더 `backend_started`(ps lstart)를 남긴다 |
| `stop` | supervisor 종료와 그룹 비움을 최대 90초 기다린 뒤 "Demo is stopped"를 출력한다. 남은 프로세스는 PID·시작 시각을 출력하고 실패로 끝난다. 포트를 쥔 프로세스를 추측해서 죽이지 않는다 |
| `start` | 기록된 이전 그룹이 살아 있으면 거부한다. 리더의 시작 시각이 기록과 다르면 그룹 id가 재사용된 것으로 보고 이전 그룹으로 치지 않는다(`owned_group`) |
| 격리 실행 | 환경 변수 `TRID_DEMO_RUNTIME`, `TRID_DEMO_PORT`, `TRID_DEMO_VERSIONS`, `TRID_DEMO_TUNNEL=0`. 기본값은 실서비스 그대로다 |

**`scripts/shared_demo.py`**

- `State.close`: llama 서버 전부에 SIGTERM을 한 번에 보내고 대기 10초를 함께 쓴다. 그 뒤 남은 것은 kill하고 회수한다.
- llama 로그는 인스턴스의 런타임 폴더에 쓴다. 격리 실행이 실서비스 로그를 덮어쓰지 않는다.

## 검증

**자동 테스트**
- `tests/test_share_demo_control.py` 2개:
  - SIGTERM을 무시하는 서버와 그 자식을 `stop_group`이 모두 종료한다.
  - 리더 시작 시각이 다르면 그룹을 자기 것으로 보지 않는다.
- 전체 54개 통과.

**실제 명령 격리 smoke** (포트 51974, llama 5196x, 터널 없음)

1. **모델 서버 정지 상태(SIGSTOP)에서 `stop`:**
   - llama 3개는 SIGTERM으로 종료됐다. 모델 서버는 20초 뒤 SIGKILL로 종료됐다.
   - `stop` 소요 21초. 포트가 비었고 `stop_leftover []`.
2. **start → 모델 질문(repair-002) → stop → 곧바로 start → 질문 → stop:**
   - 두 차례 모두 "Demo is stopped (supervisor and model server process group exited)"가 나오고 포트가 비었다.
   - 두 번째 start는 대기 없이 성공했다.
3. **이전 그룹 생존 시 start:** 시뮬레이션 그룹(`sleep`)이 살아 있는 상태에서 start하면 "Previous model server processes are still running … 75530 (started …)"로 거부됐다.

## 남은 것

- **실서비스 적용은 다음 재시작부터다.** 지금 실서비스는 이전 코드로 떠 있다(`state.json`에 그룹 기록 없음).
  - 그래서 다음 한 번의 `stop`은 이전 supervisor 로직으로 실행된다. 고아가 남을 수 있다.
  - 그 경우 새 `stop`은 기록이 없어 그룹을 모른다. 남은 llama 서버는 PID·시작 시각을 확인한 뒤 수동으로 정리해야 한다.
  - C 평가가 끝나면 재시작한다.
- `alive(supervisor_pid)`는 PID 재사용을 시작 시각으로 확인하지 않는다(기존 동작). 명령줄 문자열 확인은 그대로다.

## TABLE-LINK 후속 (같은 커밋)

- 법령 본문에 같은 "[별표 N]" 구간이 두 개 있으면 본문 구간 연결을 하지 않는다. 사유: "statute body holds 2 sections; which one is current is unconfirmed".
- 별표 5가 해당한다. B가 첫 구간을 현행판으로 본 추정은 현행 근거로 쓰지 않는다. 별표 5 행은 sha가 같은 원문 문서가 v5에 없어서 넣지 않고, "정보 부족"을 표시한다.
- 테스트가 추가돼 있고, 적용은 다음 재시작부터다.

SUPERVISION_REPORT_READY CONTROL-LIFECYCLE-001
