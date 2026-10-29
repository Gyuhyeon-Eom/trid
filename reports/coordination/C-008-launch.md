# C-008-launch: C 전용 GPU 자원 기동 보고

작성: 2026-10-30 05:50 KST. 용도는 데이터 생산·검증 추론뿐이다(학습 없음).

## 자원·소유권

| 항목 | 값 |
|---|---|
| 인스턴스 | `i-0a4b24516d78b628d` (`trid-c-taskgen-01`), g6e.xlarge (NVIDIA L40S 46 GB), ap-northeast-2a |
| 기동 | 2026-10-29 20:34:56 UTC, `run-instances`(C 세션) |
| AMI | `ami-0024061cbf153effc` (Deep Learning Base OSS Nvidia Driver GPU AMI Ubuntu 22.04 20261009). 기존 llama.cpp CUDA 빌드 캐시와 같은 AMI다 |
| 태그 | `Owner=trid-session-C-01a17c28`, `Project=trid`, `Task=C-008 six-type candidate generation and verification`, `StopCondition=stop when C-008 verify done and S3 synced; no SFT` |
| 네트워크·권한 | 보안그룹 `sg-012b1ac57d5da4891`(인바운드 없음), 인스턴스 프로필 `kms2026-gpu-smoke-ssm`(SSM·S3 권한. IAM 역할을 재사용했을 뿐, 다른 프로젝트 인스턴스·타이머는 변경하지 않음) |
| 디스크 | 루트 gp3 150 GB, 모델·로그는 NVMe `/opt/dlami/nvme/c008` |
| 종료 조건 | `InstanceInitiatedShutdownBehavior=stop`. C-008 검증이 끝나고 S3 동기화를 확인하면 `stop-instances`(종료 아님). 다른 세션 작업 부재를 확인한 뒤에 수행 |
| A/B 자원 | 변경 없음. 기동 시점 계정 내 실행 중인 EC2 없음 |

## 시작 전 종료 조건 점검 (2026-10-29 20:38 UTC, 부팅 20:35:10)

- `systemctl list-timers --all`: 표준 Ubuntu 타이머 14개뿐이다(apt-daily, logrotate, fstrim, man-db, motd-news, e2scrub_all, dpkg-db-backup, tmpfiles-clean, update-notifier 등). **`kms-gpu-safe-stop.timer` 없음**.
- `/etc/systemd/system`, `/etc/cron*`, `/var/spool/cron`에서 `shutdown|poweroff|halt` 검색 결과 0건. root crontab 없음. `/etc/cron.d`는 `e2scrub_all`뿐이다.

## 실행 구성·재개·로그

- 서빙 유닛: systemd `trid-c008-serve`(user ubuntu). 스크립트는 S3 `c008/serve.sh`(sha256 `ee9bbad7c064fabe8a2733e66a8b9ca174a91e6b1624596c450b736b4adc39d8`)다.
- 실행 내용:
  1. 캐시 빌드 `teacher-tools/llama-server-cuda-8ca4060.tgz` 복원
  2. 모델 내려받기 후 `sha256sum -c` 통과(Kanana-1.5-2.1B Q8_0 `244c7d71…adba`, Qwen3-1.7B Q8_0 `061b54da…cb1a`)
  3. llama-server 2개 실행(Kanana :18473 생성, Qwen :18474 검증, 각 ctx 65536, 16 slots)
- 로그는 2분마다 S3 `c008/serve/<run>/{kanana,qwen}.log`로 올라간다. 현재 run은 `c008/serve/20261029T2045Z/`이고, 첫 시도 `20261029T2038Z/`는 공유 라이브러리 경로 오류 기록이다.
- 첫 시도 실패와 수정: 캐시 빌드의 RUNPATH가 이전 인스턴스 경로(`/opt/dlami/nvme/trid-teacher/...`)라서 `libllama-server-impl.so`를 찾지 못했다. `LD_LIBRARY_PATH`를 지정해 재시작했다.
- 파이프라인(`scripts/build_task_candidates.py`)은 로컬에서 SSM 포트포워딩(로컬 28473/28474 → 인스턴스 18473/18474)으로 실행한다. 단계별 산출물(drafts/verified)은 항목 단위 추가 기록이라 재개할 수 있고, 단계마다 S3 `c008/work/`에 동기화한다.

## 실제 추론 증거 (2026-10-29 20:46:39 UTC, 인스턴스 내부 실행)

```
models/kanana.gguf  ' 서울이다. 서울은 대한민국의 정치, 경제, 문화'  224.8 tok/s
models/qwen.gguf    ' 서울입니다. 서울은 대한민국의 중심이 되'       230.0 tok/s
nvidia-smi: 2088 llama-server 9452 MiB / 2087 llama-server 10856 MiB
trid-c008-serve: active
```

이후 실제 생성 호출: 파일럿 6유형 각 10건 초안 생성(로컬 터널 경유, 20:49 UTC 이후). 결과는 C-008.md에 보고한다.

SUPERVISION_REPORT_READY C-008-launch
