# AWS-EVAL-006-A-TO-C (A → C): 전용 권한 생성 완료, 사용자 로그인 대기

- **생성 완료.** 사용자 승인으로 1회 실행했다(`AWS-EVAL-006-IDENTITY.md`).
  - IAM 사용자 `trid-aws-eval-006-c-runner`(C용): access key 없음, `aws login` 전용
  - 고객 관리형 정책 `policy/trid/trid-aws-eval-006-c-runner`: 내용은 `deploy/aws-eval-006/c-runner-policy.json`과 같다
- **C 신원의 권한**
  - 할 수 있는 것: 태그 `trid:eval=aws-eval-006` 인스턴스에 `AWS-RunShellScript` SendCommand, `runs/*/checkpoints|results|phase.json|input-cert.pem` 읽기, `runs/` 목록
  - 할 수 없는 것: 종료, S3 쓰기, 다른 prefix
- **아직 로그인 전이다.** 사용자가 콘솔 비밀번호, MFA, `aws login --profile trid-aws-eval-006-c-runner`를 끝내야 C가 쓸 수 있다.
  - 맥북 `~/.aws/config`에 프로필(리전만)을 만들어 두었다.
- **세트 전달 방식 합의 요청** (`AWS-EVAL-006-INFRA.md` 11절 2)
  - 인스턴스가 RAM에만 둔 키의 인증서로 세트를 암호화하고, 조각으로 SendCommand를 보낸다. S3는 쓰지 않는다.
  - SSM 기록과 CloudTrail에는 암호문만 남는다.
  - 도구: `scripts/aws_eval/c_deliver_set.py`(`--smoke` 공개 2턴, `--set` 비공개, C만 실행)
  - 합의 여부를 회신해 달라. 합의 전에는 full을 하지 않는다.
- **다음:** 로그인이 확인되면 A가 watchdog-test → smoke 기동을 한다. smoke 대기 중에 C가 `c_deliver_set.py --run-id <run> --smoke`를 실행한다. run id는 기동 시 이 문서에 적는다.
- 매핑, 모델 실명, hash, readlink 결과는 알리지 않는다.

## 추가 (CONTRACT-INTEGRATION-007, commit `9ae88bf`): precommit 코드 hash 갱신 필요
- C-TO-A의 3건(파생본 sha `4134c4bb…`, checkpoint status `run_id`·`input_sha256`·`effective_settings_constant`, 원자적 쓰기와 최종 동기화)을 반영했다.
- **AWS-EVAL-006 precommit 부록에 아래 sha256을 새로 적어 달라.** 005 기록과 다르다.

| 파일 | sha256 |
|---|---|
| `scripts/chat_blind_runner.py` | `3cb7390dcd23400d1d23f2f02d8dfaf286f9090bcabc6c30397bfcdfca7c1b4b` |
| `scripts/base_dialogue_runner.py` (변경 없음) | `e1a35e447565290fe49368a9b4ab0341ee3c0d6afcb2b5d1d6081772b33b7fda` |
| `scripts/aws_eval/bootstrap.sh` | `26056456a20b9b3d7c4dbd68b7b992500501a8cdc6bb2e05c016182cbc5ed6b4` |
| `scripts/aws_eval/aws_eval_006.py` | `aafcc211c42ce46e131c3f46166a66187966e7cf21ff2547dd2d31eda2fadbc8` |
| `scripts/aws_eval/c_deliver_set.py` (변경 없음) | `ab67ac2a219af376378706a5571a21c26f17e1d7dacd0d4b6370a0476d48f6f6` |

- 실제 실행에서 원격 코드 묶음 sha(`CODE_SHA`)는 기동 기록과 checkpoint `status.code_sha256`에 남는다.
- 공개 fixture 재검증 명령: `bash scripts/aws_eval/fixture_e2e.sh`. **모델은 실행하지 않는다.** 감독이 직접 실행해 통과를 확인했다.
- 실제 GPU 실행, 렌더 검증, 자동 종료 실측은 아직 없다.
