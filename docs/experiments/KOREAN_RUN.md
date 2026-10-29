# trid 한국어 개선 실행 기록 — 2026-10-27

## 범위

기존 자체 무작위 초기화 모델의 계보를 유지한다. 새 외부 가중치나 외부 생성 API는 사용하지 않는다. 현재의 기관 문서 응답 문제를 기초 언어 학습, 문서 독해, 후속 대화로 나누어 실험한다. 원본 일반 대화·기관 모델의 활성 선택은 평가 후 별도로 판단한다.

## 고정한 진단 평가

`data/korean-eval-v1/cases.jsonl`: 가상 문서 16개, 40턴. 기관명·부서·금액·기간 구분, 부정, 경계 조건, 모호한 질문, 자료 부족, 개정·충돌, 후속 대화, 오답 정정을 포함한다. 12개 시나리오는 실제 모델 답변을 다음 질문의 이력에 넣는다. 4개 정정 검사는 잘못된 이전 답을 의도적으로 제공하는 통제된 검사로 명시한다. 정답 예시는 입력에 넣지 않는다.

현재 `trid-grounded-v2` 기준선은 의미 검토상 4/40이다. 올바른 단답 및 적절한 `자료없음`/`규정충돌`은 인정했다. 문맥 때문에 이력이 잘린 질문은 0개다. 개발자가 작성하고 Codex가 검토한 작은 진단 세트이며 독립된 인증이나 한국어 전체의 성능 수치가 아니다. 학습 자료·체크포인트 선택에 사용하지 않는다.

- 원시 출력: `reports/korean-eval-v1-baseline.json`
- 의미 검토: `reports/korean-eval-v1-baseline-review.json`
- 고정 SHA256: `32e3b8c3712c1aebb4175828970e698a5194b76405d1f8f9457f5dcf5b69e707`

## 기초 학습 자료

이미 보관한 한국어 Wikipedia 원본 샤드에서 원래 학습 10,777개에 고정 해시 우선순위로 선정한 새 학습 문서 12,000개를 추가했다. 기존 검증 606개·시험 617개는 파일 내용까지 그대로 유지했다. 새 학습 자료는 22,777문서, 32,191,291토큰이다.

NFC·공백 정규화 동일 문서 검사와 어절 5개 묶음의 근사 MinHash 후보 검색 후 실제 Jaccard 0.8 이상 중복 검사를 적용했다. 이번 표본에서 제거된 문서는 0개다. 근사 검색의 누락 가능성이 있으며 과거 모델이 이미 본 유사 문서의 영향을 되돌리는 것은 아니다. 여전히 Wikipedia 중심 자료다. 다양한 한국어 웹 문서를 완전히 확보했다고 표현하지 않는다.

기존 8,192어휘 토크나이저를 바이트 단위로 그대로 복사했다. 검증·시험 토큰 파일의 SHA256도 기존과 같다. 출처·원문 이력·라이선스 문서를 보존했다.

```sh
.venv/bin/python scripts/prepare_korean_continuation.py
.venv/bin/trid prepare-corpus \
  --source data/korean-foundation-v1 --output data/korean-foundation-tokens-v1 \
  --tokenizer runs/trid-pretrain-pilot/tokenizer.json
```

이미 생성된 경로에는 다시 실행하지 않는다. 재현 시 새 출력 경로를 사용한다.

## 추가 사전학습

`pretrain --init-run`은 기존 사전학습의 최적 가중치·구조·토크나이저를 이어받고 새 옵티마이저·학습 일정을 시작한다. `--resume`은 현재 실행의 원래 일정·옵티마이저·난수 상태를 그대로 복원한다. 두 옵션은 함께 사용할 수 없다.

새 모델은 `runs/trid-korean-base-v1`이다. 부모는 `runs/trid-pretrain-pilot`의 최적 2,000단계 가중치다. 부모가 처리한 4,096,000토큰에서 추가 12,649단계 × 8배치 × 256토큰, 누적 30,001,152토큰 처리를 완료했다. 본 실행의 학습 시간은 2,131.5초였다. 이는 처리한 위치 수이며 고유 텍스트 양이나 학습 완료 수준을 뜻하지 않는다.

```sh
.venv/bin/trid pretrain \
  --data data/korean-foundation-tokens-v1 --output runs/trid-korean-base-v1 \
  --init-run runs/trid-pretrain-pilot --steps 12649 --stop-after 1000 \
  --batch-size 8 --lr 0.0002 --eval-every 250 --validation-batches 32
```

첫 1,000단계의 같은 검증 표본에서 손실은 5.4095 → 5.2219였다. 이후 동일 체크포인트에서 12,649단계까지 완료했으며 마지막 표본 검증 손실은 4.080963이다. 완료 여부와 실제 단계는 `runs/trid-korean-base-v1/training.json`을 확인한다.

```sh
.venv/bin/trid pretrain \
  --data data/korean-foundation-tokens-v1 --output runs/trid-korean-base-v1 \
  --resume --stop-after 11649
```

사전학습은 대화 학습이 아니다. 검증 손실이 줄었다는 이유로 일반 `chat`의 기본 가중치를 교체하지 않는다.

## 독해·대화 학습 자료

공식 KLUE MRC v1.1을 고정 Git revision에서 내려받고 파일 해시·라이선스·원본 출처를 저장했다. 모델 가중치를 받은 것은 아니다.

- 전체 문서가 512토큰 학습 예산 안에 들어가는 질문만 유지한다. 정답 위치를 이용한 문서 자르기는 하지 않는다.
- 공식 dev와 제목·본문이 겹치는 학습 후보 문서 2,623개를 제외했다.
- 공식 train 안에서 제목 및 동일 본문 연결 집합 단위로 학습·검증을 나눈다.
- 학습 5,171개: 답 있음 3,475개, 답 없음 1,696개.
- 내부 검증 292개, 별도 공식 dev 유래 시험 2,362개. 짧은 문서 부분집합이므로 공식 KLUE 점수로 보고하지 않는다.
- 원래 사전학습의 Wikipedia와 공개 독해 자료가 겹쳤을 가능성은 남는다. 오염 없는 공개 벤치마크라고 주장하지 않는다.

```sh
.venv/bin/python scripts/fetch_korean_mrc.py
.venv/bin/python scripts/prepare_korean_mrc.py
```

원본: https://github.com/KLUE-benchmark/KLUE

추가로 직접 작성한 한국어 대화는 학습용 문서 대화 92행, 정정 8행, 기본 대화 20행이다. 검증에는 다른 문서·표현을 사용한다. 이전 답변을 무작위로 붙이지 않고 실제 이어지는 대화 이력을 쓴다. 단순 문구 치환으로 행 수를 늘리지 않았다. 최종 지도학습 자료는 학습 5,291행·검증 324행이다. 표본 추출 확률은 독해 65%, 문서 대화·정정 25%, 기본 대화 10%로 정했다.

```sh
.venv/bin/python scripts/prepare_korean_sft.py
```

`train --context-length 512`로 별도 학습 실행의 문맥 한도를 확장할 수 있다. RoPE를 사용하는 현재 구조에는 학습된 위치 임베딩 표가 없으므로 가중치와 토큰 번호는 유지한다. 설정을 늘렸다는 사실만으로 장문 이해력을 확보했다고 하지 않는다. 추가 학습에서 검증 손실이 더 나빠지면 시작 가중치를 최적 가중치로 유지하도록 수정했다.

## 검증 도구

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/evaluate_pretraining.py \
  --run runs/trid-korean-base-v1 --output reports/korean-base-full-valid.json
.venv/bin/python scripts/evaluate_korean_dialogue.py \
  --run runs/검사할-대화모델 --output reports/새-대화평가.json
.venv/bin/python scripts/evaluate_korean_mrc.py \
  --run runs/검사할-대화모델 --output reports/새-독해평가.json
```

기초 모델 전체 검증은 1,174,066개 다음 토큰 위치를 모두 평가한다. 250단계마다 쓰는 고정 32배치 표본과 구분한다. 부모 모델 전체 검증 손실은 5.405495, 추가 학습 후 전체 검증 손실은 4.119327이다. 동일 토크나이저·동일 검증 토큰 파일·동일 256토큰 문맥으로 비교했다.

데이터 분리·문맥 길이·토크나이저 동일성 점검 결과는 `reports/korean-prepared-data-audit.json`, `reports/korean-sft-data-audit.json`에 있다. 자동화 테스트는 추가 학습의 부모 보존, 새 일정의 정확한 재개, 토크나이저 불일치 차단, 유사 중복 검사, 문맥 확장, 악화된 지도학습의 최적 가중치 보존을 포함한다.

## 독해·대화 후보 실행

`runs/trid-korean-dialogue-v1`은 위 기초 모델에서 아래 설정으로 실행한 별도 후보이다. 완료 상태·선택 단계는 해당 실행의 `training.json`으로 확인한다. 문장형 문서 지시는 `trid/prompts.py`에서 학습 자료·평가·기관 채팅에 공통 적용하며, 예전 값형 모델의 입력 형식은 유지한다. 답을 규칙으로 대체하지 않는다.

```sh
.venv/bin/trid train \
  --init-run runs/trid-korean-base-v1 --data data/korean-sft-v1 \
  --output runs/trid-korean-dialogue-v1 --context-length 512 \
  --steps 1200 --batch-size 12 --lr 0.0001 \
  --eval-every 200 --validation-limit 0
```

## 최종 결과와 채택 판단

기초 사전학습 12,649단계와 독해·대화 지도학습 1,200단계를 모두 완료했다. 현재 실행 중인 학습 프로세스는 없다. 새 기초 가중치는 보존하지만 새 대화 후보는 평가 미달로 채택하지 않는다. 기존 `active-model.json`의 `trid-v0.2`, `active-institution-model.json`의 `trid-grounded-v2`는 유지했다.

| 항목 | 기존 기준선 | 새 후보 |
|---|---:|---:|
| 기초 모델 전체 검증 손실, 낮을수록 좋음 | 5.405495 | 4.119327 |
| 고정 가상 문서 대화 의미 통과 | 4/40 | 1/40 |
| 일반 대화 16문항 의미 통과 | 6/16 | 1/16 |
| 사용자 실패 5질문, 실제 출력 이력 | 전부 정확한 답변 없음 | 0/5 |

새 후보의 독해 검증에서는 답 있는 질문 135개 중 정확 일치 0개, 잘못된 답변 보류 116개였다. 답 없는 질문은 62개 중 53개를 보류했다. 답변 길이 예산 때문에 292개 검증 질문 중 95개를 제외해 197개를 검사했다. 공개 벤치마크 전체 성능이 아니다. 문서 대화 의미 검토와 실제 CLI에서도 오류를 확인했으므로 정확 일치 지표만의 문제로 해석하지 않는다.

고정 문서 대화에서 이전 모델의 적절한 `자료없음`/`규정충돌` 단답을 인정한 것과 같은 기준으로 새 후보의 C02 기관명 부재 답변 하나를 인정했다. 정정 설명은 부족하다. 반복 문자열은 40턴 중 6턴에서 탐지했다. `trid chat --mode institution --run runs/trid-korean-dialogue-v1`에서도 식비를 2만원으로 잘못 답하고 기관명을 지어내는 결과를 재현했다.

기대 답변 토큰 비중 계산에서 120개 작성 대화가 약 58%, 정답 있는 실제 독해 3475개가 약 22%를 차지했다. 예제 추출 비율을 65% 독해로 정했더라도 답변 길이와 토큰 평균 손실 때문에 실제 학습의 균형이 달라질 수 있다. 이는 정확한 경사 기여도나 유일한 실패 원인을 입증한 수치는 아니다. 상세 계산은 `reports/korean-sft-target-mass-audit.json`에 있다.

다음 실험은 같은 혼합 학습의 단순 반복이 아니라, 개선된 기초 모델에서 **독해만 먼저 분리하고 예제별 손실 정규화를 비교**하는 것이다. 정답 있는 질문의 답변 정확도와 잘못된 보류 비율을 통과한 뒤 대화 문장을 다시 결합해야 한다. 이번 실행에서 이 다음 실험까지 수행했다고 주장하지 않는다. 1억 토큰 확대·모델 크기 확대·양자화·Raspberry Pi 실측도 아직 실행하지 않았다.

- 전체 결과: `reports/korean-development-v1-summary.json`
- 새 후보 의미 검토: `reports/korean-eval-v1-candidate-review.json`
- 사용자 질문 재현: `reports/institution-followups-korean-v1.json`
- 실제 CLI 출력: `reports/korean-institution-cli-smoke.txt`
- 자동화 테스트: `reports/korean-code-tests.txt`, 23개 통과
- 채택 판단: `runs/trid-korean-dialogue-v1/acceptance.json`

기초 모델의 개선과 재현 가능한 학습·평가 절차는 확보했지만, 한국어 대화·기관 응답 정확도 개선이라는 최종 목표는 아직 달성하지 못했다.


## 후속: 대화 제외, 문서 독해 분리

`trid-korean-base-v1`에서 `data/korean-mrc-v1`만 사용한 1,200단계 example 평균 손실 학습을 완료했다. 정답 2/135, 잘못된 보류 90/135, 답 없음 처리 47/62로 아직 실사용 기준에 미달한다. 활성 모델은 변경하지 않았다. 세부 실행법·한계·비교는 [KOREAN_READING.md](KOREAN_READING.md)에 기록했다.


## 후속: 정답 구간 직접 추출

자체 기초 모델에서 한국어 시작·끝 구간 및 답 존재 여부 학습 1,200단계를 완료했다. 전체 검증 손실로 고른 600단계 모델은 같은 197문항에서 답 있음 정답 11/135, 답 없음 보류 0/62다. 생성 독해 대비 구간 정답은 늘었지만 전체 정답은 49/197 → 11/197로 줄어 채택하지 않았다. [KOREAN_SPAN.md](KOREAN_SPAN.md)에 실행 방법·평가·한계를 기록했다.


## 후속: 짧은 문서 구간·존재 여부 분리

각각 자체 기초 모델에서 800단계씩 학습한 독립 모델 두 개를 만들었다. 새 짧은 합성 시험은 104/600 → 358/600으로 개선됐지만, 항목별 편차와 기존 문서 실패가 남아 기본 모델로 채택하지 않았다. 전체 크기는 두 모델 합계 34,736,642파라미터다. [KOREAN_SHORT.md](KOREAN_SHORT.md)에 구간·분류·연결 결과를 구분해 기록했다.


## 후속: 항목 대조·한글 문자·연산 최적화

화면 작업을 멈추고 자체 독해 모델 학습에 집중했다. v2/refined 추가 학습 뒤, v3 대조 자료와 고유 한글 문자 특징·이름 경계 학습을 진행했다. 최종 후보는 `runs/trid-korean-short-v3-hangul`이며 개발 회귀 998/1,120, 새 합성 진단 75/96이다. 새 진단에서 기관명 없음은 0/12로 실패해 실무 배포용으로 채택하지 않았다. 일반 대화 능력이 확보됐다는 의미가 아니다.

CPU/MPS에서 RoPE 계산을 재사용하는 최적화도 적용했고 96개 예측의 답·보류·구간 동일성을 검사했다. 46개 코드 테스트 통과. 학습은 모두 종료했다. 상세 비교와 실행법은 [KOREAN_MODEL_OPTIMIZATION.md](KOREAN_MODEL_OPTIMIZATION.md), 앞선 v2 과정은 [KOREAN_SHORT_V2.md](KOREAN_SHORT_V2.md)에 기록했다.
