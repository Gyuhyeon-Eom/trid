# trid 임시 웹 공유

공유 주소: https://test.tridatum.co

이 배포는 Cloudflare Worker → 임시 Cloudflare Tunnel → 이 Mac의 CPU 모델 서버로 연결됩니다. 별도 클라우드 머신으로 모델을 옮긴 배포가 아닙니다. Mac 전원과 인터넷 연결이 필요하며 덮개를 닫거나 재부팅하면 중단될 수 있습니다. 화면은 꺼져도 되도록 `caffeinate -i`를 서버 실행 동안에만 사용합니다.

2026-10-29 새 추출 모델로 재배포하고 24시간 실행을 시작했습니다. 현재 자동 종료 예정은 **2026-10-30 14:11 KST**입니다. 중단되면 도메인은 임시 서버 종료 안내를 표시합니다. 소규모 테스트용이며 운영 서비스의 가용성을 보장하지 않습니다.

## 사용

Windows를 포함한 브라우저에서 주소를 열고 터미널 입력줄에 `/sample`을 입력합니다. `/pdf`로 파일 선택, `/paste`로 문서 붙여넣기, `/source`로 원문 확인, `/clear`로 문서와 화면 기록 삭제가 가능합니다. 예시 질문은 해외 식비, 해외 숙박비, 담당 부서, 기관명입니다. 문서 붙여넣기(5,000자)와 텍스트 PDF(2MB·20페이지·5만 자)도 지원합니다. 모델 파일은 방문자에게 내려주지 않습니다.

추론 모델은 `runs/reader-factorial-01-cpt-v1-v2-v4`의 완료된 91M 기반 추출 독해기(CPT 몸통·v2 헤드·양방향 attention)입니다. 가중치 SHA256은 `90ccb60fcb98b839e5d74f0b3cd0781f30609654e88c660c3778167e33c9317c`이며 S3 보존본과 일치합니다. 기존 short-v3-hangul 모델로 우회하지 않습니다. 화면과 `/healthz`에는 실제 로드된 모델 이름을 표시합니다. 모델 파일은 공개 경로로 제공하지 않습니다.

기존 문서 검색·세션 처리는 유지하고, 검색된 최대 6개 구간을 각각 새 독해기에 전달합니다. 프롬프트 기준 예측 위치를 문서 구간·페이지 위치로 변환하고 원문 일치를 검사한 뒤 표시합니다. 따라서 고정 근거를 제공한 기관문서 probe 20/40은 이 웹 서비스의 정확도가 아닙니다. 단일 구간의 근거 충돌 판정은 검토 안내로 표시하며, 서로 다른 후보 답도 기존처럼 원문 검토를 요청합니다. 일반 대화·문서 간 종합 추론 서비스가 아니며 보류와 오답이 남아 있습니다.

PDF와 질문은 Cloudflare HTTPS 중계를 거쳐 운영자의 Mac으로 전송됩니다. 비공개·민감한 자료 대신 공개 가능한 테스트 자료만 사용하세요. 앱은 문서와 질문을 파일·학습 자료로 저장하지 않습니다. 문서는 HttpOnly/Secure/SameSite 세션별 메모리에 저장되며, 지우기 또는 15분 미사용 후 정리됩니다. Cloudflare 자체 네트워크 처리·메타데이터 정책과는 별개입니다.

## 중지 / 재실행

프로젝트 폴더에서:

```sh
.venv/bin/python scripts/share_demo_control.py status
.venv/bin/python scripts/share_demo_control.py stop
```

다시 24시간 열기:

```sh
.venv/bin/python scripts/share_demo_control.py start --hours 24
.venv/bin/python scripts/share_demo_control.py deploy
```

새 실행마다 터널 주소가 바뀌므로 `deploy`까지 실행해야 도메인이 새 서버로 연결됩니다. 시작이 실패하면 `.shared-demo/server.log`, `supervisor.log`, `tunnel.log`를 확인합니다. 비밀키는 `.shared-demo/proxy-secret`(0600)에 있고 Cloudflare Worker secret으로도 저장됩니다. 이 폴더를 공유하지 마세요. Cloudflare OAuth 자격 증명은 기존 Wrangler 저장소를 사용하며 프로젝트에 복사하지 않습니다.

2026-10-27 연결 복구: 터널 프로세스가 살아 있어도 실제 연결이 0개인 상태가 발생했습니다. 이제 시작 시 터널의 `/ready`와 외부 HTTPS 모델 응답을 모두 확인하며, 실행 중에는 30초마다 검사합니다. 두 번 연속 실패하면 모델 서버를 유지한 채 터널을 재시작하고 Worker의 연결 주소를 재배포합니다. 새 터널의 DNS 음성 캐시를 피하기 위해 외부 상태 검사는 요청별 DNS-over-HTTPS를 사용하며 Mac의 DNS 설정은 변경하지 않습니다. 복구 중 잠시 중단될 수 있고, 네트워크·Cloudflare 인증 문제가 지속되면 자동 복구도 완료되지 않을 수 있습니다. `status`의 `connection_status`, `last_health_at`, `recoveries`로 확인할 수 있습니다.

서버는 127.0.0.1에만 바인딩하며 인터넷 터널 직접 접근은 프록시 비밀키 없이 거절합니다. 요청 원본과 CSRF 토큰을 검사하고, 브라우저가 보내는 프록시 신원 헤더는 Worker가 덮어씁니다. Worker는 IP별 분당 30회, 서버는 세션별 분당 20회를 제한합니다. 최대 32개 세션, HTTP 작업 8개, 모델·PDF 처리 1개를 허용합니다. PDF는 별도 프로세스에서 15초 제한 및 CPU/메모리 감시하에 읽습니다. 이 조치들은 정식 보안 심사나 운영용 격리 환경을 대신하지 않습니다.

## 검증

- `tests/test_shared_demo.py`: 방문자별 문서 격리, CSRF/Origin/프록시 차단, 만료, 잘못된 입력, 크기 제한.
- `deploy/shared-demo/worker.test.mjs`: 경로 차단, 프록시 헤더 교체, 비밀키 전달 범위, 연결 종료 화면.
- `reports/shared-demo-local-smoke.json`: 이전 short-v3-hangul 모델의 샘플 4항목, PDF 담당 부서 응답에 대한 역사적 검증.
- `reports/shared-demo-public-smoke.json`: 이전 모델의 HTTPS 공개 추론과 방문자 문서 접근 차단 검증. 현재 추출 모델의 새 측정 결과가 아닙니다.
- `reports/shared-demo-extractive-update.json`: 새 가중치 해시, 배포 버전, 272개 회귀 테스트 통과, 공개 브라우저 `/sample`·`/paste` 담당 부서 응답과 정확한 원문 하이라이트 확인. 실제 모델의 로컬 샘플 해외 식비·숙박비 질문은 보류했으며 정답으로 집계하지 않았습니다. 이번 공개 UI 검증은 PDF 업로드를 재실행하지 않았습니다.

새 DNS 레코드는 일부 네트워크의 이전 NXDOMAIN 캐시 때문에 약 30분 늦게 보일 수 있습니다. 권한 DNS, Cloudflare 1.1.1.1, Google 8.8.8.8에는 새 주소가 반영됐습니다. 접속이 안 되면 잠시 후 또는 다른 네트워크에서 다시 시도하세요.

참고: [Cloudflare 임시 터널](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/), [Workers Custom Domains](https://developers.cloudflare.com/workers/configuration/routing/custom-domains/).
