# trid 한국어 RAG 생성 모델

목표는 파일 없이 한국어 대화와 공통 업무 설명을 수행하고, 기관 검색 시스템이 근거를 전달하면 그 근거로 정확히 답하는 자체 모델이다. 개별 기관의 모든 규정을 가중치에 암기시키거나, 사용자가 매번 첨부해야만 작동하는 제품을 목표로 하지 않는다.

## 현재 학습 범위

`runs/trid-korean-rag-curriculum-v1`은 기존 자체 대화 모델 `runs/trid-v0.2`에서 시작한다. 일반 대화 퇴행을 보인 첫 기본 후보 대신 다른 초기화 경로를 시험하는 별도 실행이다. 외부 사전학습 가중치나 원격 모델 API는 사용하지 않는다. 생성 모델은 17,308,032 파라미터, 문맥 길이는 512토큰이다.

7,632개 학습 행과 304개 검증 행을 사용한다. 기존 기본 대화 자료에 직접 작성한 가상의 한국어 근거 읽기 예제를 섞었다. 새로운 RAG 예제는 금액, 사람 이름, 방해 자료, 표, 정보 없음, 충돌, 현행·폐지 구분, 두 근거 결합의 8개 유형이다. 답은 짧은 값과 근거 번호이며 자유로운 장문 설명 능력을 검증하는 과제가 아니다.

학습 비중은 기본 대화 35%, 공통 업무 20%, 정보 부족·한계 10%, 문장 작성 5%, 후속 대화 5%, RAG 근거 읽기 25%다. 기본 대화 기능을 함께 유지하도록 구성했다. 학습 자료의 표현 수가 실제 독립적인 사실 수를 의미하지 않는다.

```python
train('data/korean-rag-curriculum-v1', 'runs/trid-korean-rag-curriculum-v1',
      steps=1800, batch_size=24, lr=.0001, device='mps', seed=271027,
      init_run='runs/trid-v0.2', eval_every=100, validation_limit=0,
      intent_weight=0, context_length=512, loss_normalization='example')
```

체크포인트는 검증 손실로 선택한다. 테스트 점수를 보고 학습 가중치를 고르지 않는다. 학습 로그는 `reports/korean-rag-curriculum-v1-training.log`다.

## 평가 범위와 한계

- RAG 95문항은 생성된 근거를 직접 제공하고 정답과 출처의 일치를 함께 검사한다. 공백·유니코드·마지막 마침표만 정규화한다. 모델 출력은 덧씌우거나 정답으로 교체하지 않는다.
- 새로운 RAG 자료의 숫자와 이름을 split별로 나눴다. 이 값이 과거 일반 자료나 사전학습에 전혀 없었다는 뜻은 아니다. 질문 표현도 일부 나눴으나 합성 문서 구조는 공유한다.
- 새 12문항으로 파일 없는 기본 기능을 확인한다. 기존 56턴은 이미 실패를 보았으므로 개발·회귀 진단으로만 사용한다.
- 실제 기관 PDF, 검색 정확도, OCR, 긴 문서, 문서 접근권한, 프롬프트 주입 방어를 검증한 결과가 아니다. 512토큰 모델에 긴 PDF 처리가 가능하다고 주장하지 않는다.
- 기본 후보의 RAG 기준선은 0/95였다. 학습 완료 후 동일 프로토콜로 새 후보를 비교한다.

```sh
.venv/bin/python scripts/evaluate_rag_curriculum.py --run runs/trid-korean-rag-curriculum-v1 --output reports/korean-rag-curriculum-v1-candidate.json
.venv/bin/python scripts/evaluate_base_dialogue.py --run runs/trid-korean-rag-curriculum-v1 --data data/korean-rag-curriculum-v1 --suite base-check --output reports/korean-rag-base-check-candidate.json
```

## RAG 구축 방식의 비교 계획

아래는 이후 검색 시스템 평가 계획이며 이번 모델 학습으로 구현됐다는 뜻은 아니다. 동일한 기관 문서와 질문에서 정확도·출처·지연·메모리를 비교하여 결정한다.

| 방식 | 비교할 역할 |
| --- | --- |
| 키워드 검색 | 조문 번호, 기관명, 코드 등 명시적인 표현 검색의 기준선 |
| 의미 검색 | 표현은 달라도 뜻이 가까운 질문 검색의 기준선 |
| 하이브리드 검색 + 재정렬 | 두 검색 결과를 결합하고 질문에 맞는 근거를 앞에 배치하는 후보 |
| 문서 계층을 이용한 검색 | PDF의 제목·조항·표·앞뒤 조건을 함께 보존하는 후보 |
| GraphRAG | 여러 문서의 관계나 전체 주제에 대한 질문에서 추가 비교할 후보 |

RAG 방식 이름만으로 최고 품질이 정해지지는 않는다. 검색이 정답 근거를 못 찾는 실패와, 올바른 근거를 줬는데 모델이 틀리는 실패를 분리한다. 폐쇄망 제품은 검색·문서 처리·생성·로그를 모두 내부에서 운영하고 권한을 검증해야 한다. 현재 공개 Mac 데모는 그 검증을 대신하지 않는다.

## 참고 근거

- [RAFT 연구](https://arxiv.org/abs/2403.10131): 관련 근거와 방해 문서를 함께 주는 학습의 참고. 이번 짧은 합성 실험은 논문 전체의 재현이 아니다.
- [Ko-LongRAG](https://aclanthology.org/2025.findings-emnlp.938/): 한국어 긴 문맥 평가의 참고. 현재 실행하지 않았으며 점수를 주장하지 않는다.
- [Elastic 검색·재정렬 문서](https://www.elastic.co/docs/solutions/search/ranking): 키워드·벡터 결과 결합과 재정렬 방식의 참고. 특정 제품 도입 결정은 아니다.
- [GraphRAG 공식 개요](https://microsoft.github.io/graphrag/query/overview/): local/global 등 질문 범위에 따른 검색 방식 구분의 참고.

## 소형 후보 결과와 채택 여부

1,800단계 학습을 완료했다(510.1초). 검증 손실로 선택한 체크포인트는 300단계, 손실 1.0996이었다. RAG 진단은 **15/95**였지만 충돌 12/12, 근거 없음 3/12에서만 통과했다. 실제 값과 출처를 요구하는 나머지 **71문항은 0/71**이었다. 이를 근거 읽기 성능 향상으로 해석하지 않는다.

새 파일 없는 기본 질문은 기존 4/12에서 후보 5/12였고, 이미 확인한 56턴 회귀 진단은 21/56이었다. 판정은 Codex의 의미 검토다. 마지막 단계도 추가로 검증 split의 RAG 정확도를 확인했지만 12/91로, 손실 최적 체크포인트의 23/91보다 낮았다. 따라서 마지막 체크포인트를 대신 채택하지 않았다.

CPU와 MPS의 실제 입력 logits 차이는 최대 약 0.000018, 그 입력의 greedy 토큰은 모두 같았다. 동일 예제의 생성 오류도 두 장치에서 같았으며, 기존 rotary/cache 검사도 통과했다. 이 검사는 제한된 구현 점검이며 모든 구현 결함을 배제하는 증명은 아니다.

후보는 `runs/trid-base-rag-v0.3-candidate`로 내보냈지만 기본 활성 모델과 공개 웹 서비스는 교체하지 않았다. 보고서는 `reports/korean-rag-curriculum-v1-candidate.json`, `reports/korean-rag-base-check-candidate*.json`, `reports/korean-rag-base-regression*.json`에 있다.

## 정확도 우선: 더 큰 자체 모델 실행

사용자는 2026-10-27에 Raspberry Pi급 크기 제한보다 Mac 32GB에서 정확도를 우선하고, 이후 경량화하는 방향을 선택했다.

현재 Mac의 짧은 무작위 토큰 계산 실험에서 45,101,568 및 91,245,312 파라미터 모델의 학습 연산을 확인했다. 91,245,312 파라미터, 문맥 1,024, batch 2, FP32 AdamW는 약 2,637 tokens/s였고 backward 직후 관측한 Metal driver 할당량 최대는 5.52GB였다. 이는 보장된 최대 메모리나 지속 부하 속도가 아니다. 상세 수치는 `reports/mac-model-capacity-v1.json`에 있다. 무작위 토큰 측정은 언어 학습 결과로 저장하지 않았다.

새 사전학습 자료는 기존 로컬 공개 Wikipedia shard에서 60,000문서를 추가 선택해 총 70,777개 학습 문서, 79,900,979개 토큰으로 확장했다. 기존 606개 검증 문서와 617개 테스트 문서는 그대로 유지했다. 출처·라이선스·변경 기록은 `data/korean-foundation-v2/`에 있다. 근사 중복 필터는 완전한 중복 부재를 증명하지 않는다. 위키 중심이라는 다양성 한계는 여전히 남는다.

`scripts/train_medium_pipeline.py`는 다음 작업을 한 번의 유한한 실행으로 수행한다.

1. 새 자체 무작위 가중치에서 91,245,312 파라미터·1,024토큰 문맥으로 사전학습한다. 기존 자체 tokenizer를 유지하며 외부 사전학습 모델을 사용하지 않는다.
2. 사전학습은 최대 48,829단계, 100,001,792토큰이다. 250단계마다 검증·체크포인트를 저장한다. 재개 시 optimizer와 sampling RNG를 복구한다.
3. 완료 후 혼합 대화·RAG 자료로 별도 2,000단계 SFT를 실행한다.
4. 진단 결과와 `runs/trid-medium-v0.4-candidate`를 저장하고 종료한다. **자동 활성화나 웹 배포는 하지 않는다.** 이전에 관찰한 개발 진단이므로 독립 테스트 성능으로 광고하지 않는다.

첫 실제 250단계(512,000토큰) 체크포인트와 optimizer 저장을 확인했다. 해당 구간은 약 2,491 tokens/s였다. 이 속도 기준 사전학습에는 약 11시간 이상이 예상되지만, 열·다른 작업·저장 시간에 따라 달라진다. 이 문서 작성 시 실행은 진행 중이며 완성된 대화 모델이나 RAG 품질을 주장하지 않는다. Mac이 실행 상태여야 한다. 실행 중에는 해당 프로세스에 한해 idle sleep을 막고 종료하면 해제한다. 덮개 닫기나 재부팅 이후 자동 재시작 서비스는 설치하지 않았다.

```sh
# 상태와 최근 체크포인트
.venv/bin/python scripts/train_medium_pipeline.py --status
# 현재 구간 종료 후 중단 요청
.venv/bin/python scripts/train_medium_pipeline.py --stop
# 중단된 실행 재개 (현재 실행이 종료된 뒤)
.venv/bin/python -u scripts/train_medium_pipeline.py --resume
```

로그: `reports/medium-training-pipeline.log`. 상태: `reports/medium-training-job.json`. 사전학습 원본: `runs/trid-korean-medium-base-v1`. 대화 학습 도중 중단된 시도는 보존하며, 명시적으로 재개하면 새 SFT 시도로 시작한다. 모든 평가 완료 후에도 근거 일치, 기본 대화, 실제 문서 평가를 통과해야 배포를 검토한다.

## RAG SFT v2와 독립 평가셋 (2026-10-27)

### 진단
v1 RAG 자료는 항목 4개, 금액 10개, 이름 8개, 문서당 사실 1개로 구성됐고 중복 제거 후 RAG 학습 행이 2,716개였다. 기존 17M 후보(`runs/trid-base-rag-v0.3-candidate`)를 새 독립 평가셋으로 측정하면 **0/310**이며, 모든 질문에 대부분 `2만원 [D5]`처럼 학습 자료의 값을 그대로 출력했다(`reports/rag-eval-v1-baseline-v0.3-candidate.json`, 512토큰 문맥 초과 118개 포함). 근거를 읽고 복사하는 대신 학습 값을 기억하는 실패로 판단한다.

### 독립 평가셋 `data/korean-rag-eval-v1` (310문항)
- 직접 작성 160문항: 4개 영역(공공 행정, 교육, 지자체 서비스, 기업·병원)별 40문항. 값 32, 이름·부서 24, 방해 근거 24, 표 20, 정보 없음 20, 충돌 12, 개정·폐지 16, 두 근거 결합 12. 학습 생성기와 별도로 작성한 가상의 기관·수치이며 원본은 `data/korean-rag-eval-v1-source/`다.
- KLUE MRC 공식 dev 150문항(답 있음 100, 답 없음 50), 50% 확률로 다른 문서 1개를 방해 근거로 추가. trid 학습에는 쓰지 않았지만 이전 독해 진단에서 사용한 자료다.
- 채점(`scripts/evaluate_rag_eval.py`): strict(값과 출처 모두 정답), value(출처 무시), lenient_value(생성 값이 정답을 포함), source, 보류 여부를 따로 기록한다. 문맥을 초과한 문항은 오답으로 세고 `skipped_context`로 표시한다.
- **학습이나 체크포인트 선택에 사용하지 않는다.** v2 생성기는 이 평가셋의 근거·질문 문장이 학습 프롬프트에 들어가지 않았는지 검사한다.

### SFT v2 자료 `data/korean-rag-curriculum-v2`
`scripts/prepare_rag_curriculum_v2.py`로 생성. 학습 22,116행, 검증 693행.
- 합성 근거 11,200행(7종 × 1,600): 16개 영역 × 6–7개 항목, 금액·날짜·시각·기간·비율·인원·전화·사람·부서·장소·기관 값을 넓은 범위에서 생성해 기억이 아닌 복사를 학습시킨다. 문서마다 3–5개 사실을 문장·목록·조문·표·FAQ·메일 형식으로 넣는다. 방해 문서는 같은 항목을 다른 대상(예: 정규직/계약직)에 대해 담는다. 정보 없음은 인접 항목만 있거나 다른 대상의 값만 있는 대조형이다. 개정·폐지 표시는 여러 표현을 사용한다.
- KLUE MRC train 실제 문단 6,000행(답 있음 4,133, 답 없음 1,867). 공식 dev와 제목·문단이 겹치는 문서는 제외. 출처와 라이선스는 `ATTRIBUTION.txt`에 기록.
- v1의 기본 대화·업무·한계 행은 그대로 재사용한다. 표본 비중: 기본 30%, 업무 15%, 한계 10%, 작성 5%, 후속 5%, 합성 RAG 23%, KLUE 12%.
- 답 형식은 v1과 같다(`값 [Dn]`, `확인 불가`, `근거 충돌`). 프롬프트 형식은 `trid.prompts.rag_prompt`에서 공유한다.

### 비교 실행
`scripts/train_medium_sft_v2.py`가 medium 파이프라인 완료(`reports/medium-training-job.json`의 status `complete`)를 5분 간격으로 확인한다. 완료되면 같은 base 가중치와 같은 SFT 설정(2,000단계, batch 16, lr 1e-4, seed 271029, 문맥 1024)으로 데이터만 바꿔 학습하고, v1·v2 SFT를 독립 평가셋으로 평가한다. 결과는 `reports/medium-sft-v1-v2-comparison.json`, 기본 대화 회귀 보고서는 `reports/medium-base-*-sft-v2.json`에 저장된다. 모델 내보내기·활성화·배포는 하지 않는다. 파이프라인이 실패·중단되면 대기 작업도 실패로 종료한다.

```sh
.venv/bin/python scripts/train_medium_sft_v2.py --status   # 로그: reports/medium-sft-v2.log
```

### 근거 복사 제한 디코딩 (`trid/constrained.py`)
재학습 없이 출력 형식을 제한한다. 답은 `확인 불가`, `근거 충돌`, 또는 `값 [Dn]`(최대 2개)만 허용한다. 값은 인용한 근거 문서에 그대로 있는 문자열이어야 하고, 문서 안에서 글자 경계에서 시작해 글자 경계에서 끝나야 한다(UTF-8 바이트 조각 금지). 출처 번호는 프롬프트에 주어진 번호만 쓸 수 있다. 바이트 수준 BPE 토큰마다 이 조건을 검사해, 조건을 만족하는 토큰 중 점수가 가장 높은 것을 고른다. 각 접두어마다 답을 완성하는 최소 바이트 수를 계산하고, 그 수가 남은 출력 토큰 안에 들어가는 토큰만 받는다(모든 단일 바이트가 토큰이므로 바이트 수가 토큰 수의 상한). 그래서 막다른 길이나 토큰 한도 소진으로 미완성 답이 나오지 않고, 생성은 항상 완성된 답과 EOS로 끝난다. EOS는 답이 완성된 뒤에만 허용한다. 따라서 근거에 없는 값은 생성되지 않는다. 다만 어떤 구간을 고를지와 보류할지는 여전히 모델이 정하므로, 이 제한이 정답을 보장하지는 않는다.

`scripts/evaluate_rag_eval.py --decoding evidence`로 측정한다. 기존 17M 후보에서는 자유 생성 0/310, 제한 디코딩 3/310이었고, 출처 일치는 14에서 114로 늘었다(`reports/rag-eval-v1-baseline-v0.3-candidate-evidence.json`). 문맥 안에 든 192문항의 답이 모두 문법에 맞고 깨진 글자가 없다. 이 모델은 근거 안에서 정답 위치를 찾지 못한다. 값 부분 문자열 중 `2`로 시작하는 쪽을 주로 골랐는데, 학습 자료 값을 기억한 데서 오는 편향으로 보인다. `scripts/train_medium_sft_v2.py`는 v1·v2 SFT를 자유 생성과 제한 디코딩 두 방식 모두로 평가한다.

### 구간 단위 점수화 (`trid/span_scoring.py`, 2026-10-28)
토큰 탐욕 선택 대신 근거의 후보 구간(어절·구분자 경계에서 시작, 내부 숫자·영문 연속은 쪼개지 않음, 최대 8어절·64자)을 `값 [Dn]`으로 만들어 전체 로그우도로 고른다. 아래 수치는 초기 경계 규칙으로 측정한 과거 기록이다: 평가셋 단일 값 216문항 중 정답 후보 포함 210/216, 두 값 결합 12문항은 미지원. 17M 후보의 strict 22/310은 전부 보류 문항이고 답 있는 228문항은 0이다(`reports/rag-eval-v1-baseline-v0.3-candidate-span-{sum,mean}.json`). 보류 후보를 빼고 구간만 고르게 해도 authored 80문항에서 출처는 62/80, 값은 0/80이었다. 추출 헤드로 가는 근거다(`reports/closed-network-architecture-survey.md` §7–8). 최신 경계 규칙은 아래 `span_boundary_v2`이며 과거 평가 수치를 새 규칙의 결과로 해석하지 않는다.

### 추출 헤드 (`trid/extractive.py`, 2026-10-28)
입력은 `trid.prompts.rag_prompt` 프롬프트 그대로이고, trid 몸통(전체 가중치 학습) 위에 헤드 둘을 얹는다. (1) 마지막 은닉 상태에서 3-way 존재(답 있음 / 확인 불가 / 근거 충돌), (2) 글자 단위 시작·끝 점수(질의는 마지막 은닉, 키는 글자가 속한 토큰 은닉 + 글자 특징 + 토큰 내 오프셋). `[Dn] ` 표지·공백·질문 글자는 점수에서 제외하고, 출처는 고른 구간이 놓인 문단에서 결정한다. 같은 문단·최대 64자·대괄호 미포함 조건과 `trid/span_boundaries.py`의 공통 경계 규칙을 지키는 (시작, 끝) 중 점수 최대를 고른다. 구간 점수화의 추가 8어절 제한과 추출 헤드의 64자 제한은 별개다. 손실은 CE(존재) + CE(시작) + CE(끝)(답 있는 행만). 존재 확률은 아직 보정되지 않았으며 두 값 결합 답은 지원하지 않는다.

자료 매핑(`data/korean-rag-curriculum-v2`, 기존 medium-base 토크나이저·문맥 1024): 정답이 반복되면 인용 문단 안의 출력 가능한 출현을 선택하도록 고친 뒤 사용 가능 행은 학습 15,385개·검증 437개였다. 이어 `span_boundary_v2`에서 괄호·따옴표·구분자 뒤 시작과 숫자 뒤의 분리 쉼표를 허용해 **학습 15,593개·검증 440개**로 늘었다(학습 208개·검증 3개 복구). 남은 경계 제외 학습 6개는 실제 단어 내부 정답이나 쉼표만 정답인 행이다. combine 제외(학습 1,600·검증 40), 인용 문단에서 찾을 수 없는 학습 1개, 비-RAG 제외는 유지한다. 이는 학습 라벨 정합성 개선이지 정답률 상승 측정이 아니다.

학습 라벨과 추론 구간은 같은 허용 조건을 따른다. 값 내부에 `[`·`]`가 들어가 서비스 답변 문법을 깨는 구간은 고르지 않는다. 유효 구간이 하나도 없으면 임의의 프롬프트 위치를 인용하지 않고 `확인 불가`와 `reason=no_valid_span`을 반환한다. 재현 입력의 잘못된 형식·원문 밖 구간 문제가 수정 후 사라지는 것을 CPU에서 확인했다. 이 보류는 표현 가능한 구간의 부재에 대한 처리이며, 의미적 확신도 보정은 아니다.

스크립트: `scripts/train_extractive.py`(rag 행만, `reader.safetensors`), `scripts/evaluate_extractive.py`(`evaluate_rag_eval.py`의 `score`를 그대로 써 같은 보고서 형식), `scripts/train_medium_extractive.py`(SFT v2 작업 완료 후 medium base에서 2,000단계, batch 16, lr 1e-4, seed 271029, mps; `reports/medium-extractive-v1.json`과 `reports/medium-extractive-job.json`에 기록; 내보내기·활성화 없음).

현재 추출 학습은 **내부 검증셋의 실제 출력 정답률 최대**, 동률이면 검증 손실 최소로 체크포인트를 선택한다. 답 있음은 값과 출처가 모두 맞아야 하고, 답 없음·충돌은 해당 보류 문구가 맞아야 한다. 같은 문단에서 같은 값의 다른 출현을 고른 경우는 정답이다. 시작점·끝점만 맞고 존재 판단을 틀리면 오답이다. 초기 체크포인트도 비교에 포함하며, 독립 310문항 평가셋은 선택에 쓰지 않는다. 검증 배치의 한 번의 forward 결과를 손실과 출력 지표에 함께 사용한다. 전체 strict와 답 있음·없음·충돌별 strict 및 존재 손실을 기록하고, 기존 존재 정확도·위치 일치는 진단용으로만 남긴다. 이미 끝난 학습의 가중치나 과거 측정값이 이 변경으로 개선되는 것은 아니다. 회귀 테스트는 `tests/test_extractive.py`, `tests/test_train_extractive.py`.

CPU 경로 확인(학습 200행 표본, 20단계, batch 4, medium base 30,500단계 스냅샷): 검증 40행 손실 17.55 → 11.78, 존재 정확도 0.10 → 0.70, 구간 일치 1/28. 독립 평가셋 앞 40문항은 strict 2/40, value 2, lenient_value 5, source 22/40, 보류 1(답 없음 8문항은 모두 미보류), `skipped_context` 0(`reports/extractive-smoke-cpu.json`). 거의 무작위 수준이며 경로가 끝까지 도는지만 확인한 것이다.

#### 공유 경계 규칙과 개발셋 전후 확인

`span_boundary_v2`는 괄호·따옴표·가운뎃점·물결표 등 구분자 뒤의 값을 선택할 수 있게 한다. 단어 내부 시작은 막고, `1,000`·`30.4`·`foo.bar`의 내부 쉼표·마침표 양쪽은 자르지 않는다. 반면 문장 끝 마침표나 항목 구분 쉼표 앞에서는 답을 끝낼 수 있다. 원문과 글자 위치는 변경하지 않는다. 후보 열거·학습 라벨·추출 디코더가 같은 함수를 쓰고, 새 학습 메타데이터와 추론 결과에 `boundary_policy`를 기록한다. 과거 모델의 학습 정책을 알 수 없으면 보고서에 `null`로 남긴다.

`scripts/evaluate_extractive_dev.py`는 curriculum의 **valid.jsonl만** 읽는다. 비-RAG 213행을 제외한 480행 전체를 평가하며, 두 값 결합 40행과 출력 불가능한 라벨·실행 오류도 분모에서 빼지 않는다. 잠긴 310문항 평가셋은 이 변경의 설계·검증에 사용하지 않았다.

| 동일한 20-step CPU 시험용 reader | 수정 전 | 수정 후 |
|---|---:|---:|
| 개발 문항 수 | 480 | 480 |
| 출력·출처·보류까지 정확 | 23 | 23 |
| 원래 정답 문자열을 출력할 수 없는 라벨 | 43 | 40 |
| 실행 예외 | 0 | 0 |

남은 40개는 모두 두 값 결합 문항이다. **같은 가중치에서 정답률 개선은 없었다.** 수정 후에도 답 없음 103문항 중 95개, 충돌 40문항 전부를 답 있음으로 판단했다. 원시 `P(present)`로 보류 임계값을 정하기에는 이 모델이 너무 미학습 상태라, 측정한 risk–coverage 곡선을 서비스에 적용하지 않는다. 곡선은 이미 답 있음으로 분류한 출력 중 `P(present) ≥ 임계값`인 것만 채택하며 같은 확률은 함께 처리한다. coverage의 분모는 실패·미지원 문항을 포함한 480개이고, 채택 답이 0개일 때 risk는 0이 아닌 `null`이다.

측정 가중치는 `/tmp/trid-extractive-smoke`의 `cd8d234a52a65620d0ce09cbae1393e3d4add88abfeb97f59d06b3b91197e4c8`이며, 200행·20단계 시험 학습 결과다. 과거 체크포인트 선택에 사용한 40행도 이 개발셋에 속하므로 독립 성능평가가 아니다. `reports/extractive-dev-boundary-{before,after}.json`에 문항별 결과·코드/모델/자료 해시·분류 오류를 남겼다. 전후 실행 모두 CPU 2스레드이며 가중치 변경은 없었다. 개발 문항 150개에서 구간 열거 후보 수는 435,207 → 479,097로 늘어 구간 점수화 비용은 증가할 수 있다. 빠른 배포 경로의 목표는 여전히 추출 헤드 1회 추론이다.

```sh
.venv/bin/python scripts/evaluate_extractive_dev.py \
  --run /tmp/trid-extractive-smoke \
  --output reports/extractive-dev-new-measurement.json --threads 2
```

전체 회귀 테스트 217개 통과. 기존 MPS SFT v2 작업은 유지했으며, 대기 중인 본 추출 학습이 시작할 때 새 경계 규칙과 정책 메타데이터를 사용한다. 학습 실행 전후 코드를 계속 바꾸면서 같은 결과라고 취급하지 않도록 보고서의 소스 해시를 함께 확인한다.

### 검색 헤드 (`trid/retrieval_head.py`, 2026-10-28)
조사 보고서 §2 ColBERT 행. 외부 임베딩 모델 없이 trid 몸통으로 late interaction 검색기를 만든다. 질문(`질문: ` 접두)과 문단(`문단: ` 접두)을 따로 인과 몸통(`hidden_only=True`)에 넣고, 토큰마다 128차원 선형 투영 + L2 정규화(bos 제외). 점수는 MaxSim(질문 토큰마다 문단 토큰 최대 코사인의 합). 문단은 한 번 인코딩해 fp16 배열(`PassageIndex`, 421문단 55 MB, `npz` 저장)로 두고, 질의는 짧은 몸통 통과 한 번 + 행렬곱 한 번이다. 손실은 MaxSim(질문 길이로 나눔, temperature 0.05) softmax CE: 배치 안 다른 문단이 음성(positive 중복 없는 배치만 뽑음) + 쌍마다 hard negative 1개(같은 프롬프트의 방해 문단 ∪ 학습 문단 말뭉치 BM25-kiwi 상위 8개 중 비정답). 몸통은 학습하거나 `--freeze-backbone`으로 투영만 학습한다. 양방향 attention 전환은 하지 않았다(§1.4 b는 추출 헤드 결과 뒤).

자료 쌍(`data/korean-rag-curriculum-v2`): 답이 `값 [Dn]` 하나인 행은 인용 문단, `rag_klue_missing` 중 문단이 하나뿐인 행은 그 문단(KLUE 질문이 그 문단으로 쓰였으므로). combine·conflict·missing·history 행은 제외. 학습 12,092쌍(klue 4,132 / klue_missing 1,560 / value·revision·noise·table 각 1,600; 문단 11,318개, 프롬프트 방해 문단 4,553개), 검증 348쌍(문단 335개, 정답 + 방해). 검증 지표는 검증 문단 전체에서 Recall@1/3, 체크포인트는 Recall@3 최대(동률이면 손실 낮은 쪽).

스크립트: `scripts/train_retrieval_head.py`(`retrieval.safetensors` + `training.json`), `scripts/evaluate_retrieval_head.py`(`benchmark_retrieval.py`의 kiwi 토큰화·BM25·RRF·지표 함수를 import; 같은 421문단·228질의; head / bm25_kiwi / hybrid(RRF k=60) → `reports/retrieval-head-<name>.json`), `scripts/train_medium_retrieval_head.py`(`reports/medium-extractive-job.json`이 complete가 된 뒤 medium base에서 2,000단계, batch 16, lr 1e-4, seed 271030, mps, 몸통 학습; `reports/retrieval-head-medium-v1.json`·`reports/medium-retrieval-head-job.json`; 내보내기·활성화 없음). 테스트는 `tests/test_retrieval_head.py`(형상, MaxSim = 완전 탐색, 소형 모델 40단계 후 정답 문단 1위, 저장·로드). bm25s·kiwipiepy는 학습 시 hard negative에만 쓰고 제품 검색 경로에는 들어가지 않는다.

CPU 경로 확인(학습 500쌍 표본, 30단계, batch 8, 몸통 고정, medium base 30,500단계 스냅샷, 검증 128쌍): Recall@1 0.578 → 0.570, Recall@3 0.703 → 0.734, 손실 3.11 → 2.87. 독립 평가셋 421문단·228질의: head R@1 0.478 / R@3 0.583 / R@5 0.636 / MRR 0.565, 질의당 37 ms(bm25_kiwi 0.6 ms); bm25_kiwi R@3 0.947; hybrid R@3 0.934(결합이 BM25보다 낮음). 유형별 head R@3: klue 0.88, authored 0.35(표 0.10, 이름 0.25). 문단 62개는 512토큰에서 잘렸다(`reports/retrieval-head-smoke-cpu.json`). 투영만 30단계 학습한 수치라 BM25에 한참 못 미치고, 무작위 투영도 R@1 0.58이 나오는 것은 같은 토큰의 은닉 상태가 비슷해 MaxSim이 어휘 일치를 그대로 재기 때문이다. 결론은 medium 2,000단계(몸통 학습) 결과를 본 뒤다.

### Qwen 비교와 Astra·Fable 교차 검토

`scripts/evaluate_external_chat.py`는 개발용 로컬 비교기이며 제품 추론 경로에 들어가지 않는다. Astra가 구현하고 Claude Fable(`anthropic/claude-fable-5-1`)이 프로토콜과 코드를 독립 검토했다. 검토 과정에서 평가 분모를 198이 아닌 **답 있음 228 / 답 없음·충돌 82**로 교정했고, 기존 결과에 없는 종료·오류 계측은 0이 아니라 `null`로 표시하도록 고쳤다. 모델끼리 동의했다는 사실보다 원시 출력과 실행 증거를 우선한다.

비교 대상은 [공식 Qwen3-0.6B-GGUF](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/tree/23749fefcc72300e3a2ad315e1317431b06b590a)의 Q8_0이다. 파일 SHA256은 `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`, 크기는 639,446,688바이트. CPU-only llama.cpp `8ca4060`, 스레드 2개, 문맥 4,096토큰으로 실행했다. trid는 완료된 SFT v1 후보 `runs/trid-medium-v0.4-candidate`를 PyTorch CPU 4스레드·문맥 1,024토큰으로 실행했다. 폴더의 과거 `v0.4` 표기는 배포 버전 승격이 아니며 프로젝트 버전은 0.1이다.

고정 조건: 동일한 310개 근거·질문 원문, 각 모델의 입력 형식, greedy·출력 최대 64토큰. Qwen은 네이티브 템플릿의 `enable_thinking=false`가 실제 빈 think 접두부로 렌더되는지 검사했다. 추가 시스템 지시가 없는 조건과 고정된 답 형식 지시를 주는 조건을 **둘 다 실행 전 고정하고 모두 보고**한다. 질문별 실패·문맥 초과를 분모에서 빼지 않고, 생성 제한 종료·형식 오류·원시 출력·렌더 프롬프트 해시를 남긴다. 첫 추론 자체가 실패하면 해당 실행은 중단하며 유효한 전체 평가로 취급하지 않는다. 기존 `score()` 결과도 유지하지만, `strict_product`는 전체 문자열이 답 형식에 맞아야 정답으로 센다. 기존 scorer가 정답 뒤의 임의 설명을 무시하던 문제를 제품 지표에서 차단한다.

| 조건 | 제품 형식까지 맞힌 정답 / 310 | 형식 오류 / 310 | 출력 한도 종료 |
|---|---:|---:|---:|
| trid 91M SFT v1 · 자유 생성 | 1 | 160 | 미계측 |
| trid 91M SFT v1 · 근거 제한 | 3 | 0 | 미계측 |
| Qwen3-0.6B Q8_0 · 원래 프롬프트 | 0 | 307 | 38 |
| Qwen3-0.6B Q8_0 · 형식 지시 추가 | 2 | 307 | 7 |

Qwen 두 조건 모두 문맥 누락·실행 오류·컨텍스트 잘림은 0이었다. 하지만 많은 출력이 요구한 `값 [Dn]` 대신 `[Dn] 값`이나 설명문이었다. **이 결과는 형식 준수 실패에 지배되므로 trid의 Qwen급 능력이나 우위를 입증하지 않는다.** 양쪽 모두 제품 기준에서 바닥 수준이며, 값·출처·보류의 실제 신뢰성과 별도 과제 일반화가 필요하다. `lenient_value`도 기존 pair 파서에 의존하므로 형식에 독립적인 의미 정확도로 읽으면 안 된다. 이 결과를 보고 프롬프트를 고쳐 같은 310문항에서 최고 점수를 고르지 않는다. 속도는 런타임·스레드 수·동시 학습 부하가 달라 직접 우열을 비교하지 않는다. 보고서의 Wilson 구간도 문항 독립을 가정한 참고값이며 분야 일반화나 동등성 검정이 아니다.

실행 기록: `reports/rag310-qwen3-0.6b-q8-comparison.json`, `reports/rag310-medium-sft-v1-{free,evidence}.json`. Qwen 서버는 측정 후 종료했으며 비교용 가중치는 제품·학습에 연결하지 않았다. 아래 명령은 내려받은 고정 GGUF와 별도로 빌드한 CPU llama.cpp가 필요하다.

```sh
# 별도 터미널: 비교용 로컬 서버, GPU 미사용
OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 nice -n 10 /tmp/llama.cpp/build/bin/llama-server \
  -m /tmp/trid-qwen-baseline/Qwen3-0.6B-Q8_0.gguf --alias qwen3-0.6b-q8 \
  --host 127.0.0.1 --port 18473 -c 4096 -t 2 -tb 2 -np 1 -ngl 0 \
  --no-context-shift --jinja --chat-template-kwargs '{"enable_thinking":false}'

PYTHONPATH=. .venv/bin/python scripts/evaluate_external_chat.py \
  --model-file /tmp/trid-qwen-baseline/Qwen3-0.6B-Q8_0.gguf \
  --model-id Qwen/Qwen3-0.6B-GGUF --revision 23749fefcc72300e3a2ad315e1317431b06b590a \
  --compare-report reports/rag310-medium-sft-v1-free.json \
  --compare-report reports/rag310-medium-sft-v1-evidence.json \
  --output reports/rag310-qwen3-0.6b-q8-rerun.json
```

서비스 교차 검토에서는 추출 위치를 첫 번째 동일 문자열로 바꾸던 문제, 표 줄바꿈 손실, 고정 답 토큰 여유분을 수정했다. 추출 위치는 원문·감사 로그까지 보존하고, 구간 점수화는 실제 후보 토큰 길이를 기준으로 문맥을 채운다. 자세한 입력 제약과 점수 해석은 `docs/SERVICE.md`에 기록했다. 전체 회귀 테스트 204개와 실제 CPU 서비스 실행을 확인했다. 모델의 업무 정확도 향상이나 배포 승인을 뜻하지 않는다.

사후 실패 분석에서는 두 모델에 **동일한** 느슨한 조건도 적용했다: 출력과 정답 후보를 NFKC·공백 제거·끝 마침표 제거로 정규화하고, 요청된 모든 값 그룹마다 허용된 정답 후보 중 하나가 출력 어디엔가 포함되는지 확인했다. 답 있음 228문항에서 Qwen 원래 프롬프트 147개 / 형식 지시 102개, trid 자유 생성 4개 / 제한 디코딩 9개였다. 이는 출처·순서·부정 표현을 확인하지 않고 긴 설명이나 근거 전체 복사에도 점수를 주므로 **사후 진단이지 정답률·사전 등록 지표·선택 기준이 아니다.** 서로 다른 파서에 의존하는 지표를 모델 간에 직접 비교하지 않았다. 주 비교 대상은 SFT 생성 후보이며, 서비스의 추출 헤드·구간 점수화 백엔드 결과로 대체해 읽으면 안 된다.

### 학습 메모리 제한

SFT v2의 100단계 기록은 누적 6,467.14초였다. 이 시간에는 초기·100단계 검증도 포함되므로 순수 업데이트 속도로 나누면 안 된다. 실행 중인 PID 15477을 1초간 샘플링한 결과 역전파의 MPS 행렬곱·softmax와 GPU 명령 대기가 관측됐다. `sample`의 프로세스 메모리 표기는 38.3G(최대 38.6G), 시스템 RAM은 32 GiB, 시스템 swap 사용량은 23,336.75M였다. 메모리 압박이 강한 원인 후보지만 과거 경과 시간 전체를 설명한다고 단정하지 않는다. 관측값은 `reports/sft-v2-memory-investigation.json`에 기록했다.

이를 줄이기 위해 SFT와 추출 학습에 `--micro-batch-size`를 추가했다(기본 4, 실제 batch보다 크면 batch 크기로 제한). 예를 들어 batch 16은 원래와 같은 16행을 뽑은 뒤 4행씩 역전파하고, gradient clipping과 optimizer update는 한 번만 한다. 샘플링 순서·실효 batch·학습률 일정은 바꾸지 않는다. SFT의 example/token 평균과 선택적 intent 평균을 각각 전체 batch의 분모로 나눠 누적한다. 추출 헤드는 존재 손실을 전체 행 수로, 시작·끝 손실을 답 있음 행 수로 나눈다. 단순히 각 작은 batch의 평균을 다시 평균내지 않는다.

SFT에서는 정답 토큰 위치만 어휘 로짓으로 투영해, 손실에서 버릴 프롬프트·padding 위치의 어휘 분포를 계산하지 않는다. 모든 위치가 학습 대상인 사전학습은 기존 dense 투영을 유지한다. 모델 구조와 체크포인트 텐서 형식은 그대로다.

검증셋도 작은 batch로 전부 평가한다. 특히 추출 검증 손실은 전체 검증셋의 존재/구간 분모로 합산하도록 고쳐, 답 있음 비율과 마지막 batch 크기에 따라 손실이 달라지던 문제를 없앴다. 정답률 우선·손실 차선 체크포인트 선택은 유지하되, 과거의 batch별 평균 검증 손실과 새 손실을 같은 값으로 취급하지 않는다.

CPU에서 실제 SFT CLI와 추출 학습 스크립트를 각각 2단계 실행했다. 실효 batch 7·micro-batch 2로 마지막 조각이 1행인 경우와 SFT intent 손실을 함께 확인했고, SFT 최종 가중치 변경·학습 완료·메타데이터 기록을 확인했다. 별도로 dense 사전학습의 eager 실행과 `torch.compile(backend="eager", fullgraph=True)` 실행의 손실·전체 파라미터 gradient가 일치했다. 이는 경로와 수치 정합성 확인이지 MPS 속도 측정이나 학습 품질 평가가 아니다.

메모리 최적화는 이미 실행 중이던 SFT v2에 소급 적용되지 않았다. SFT 체크포인트에는 optimizer/RNG 상태가 없어 진행 상태를 완전히 보존한 재시작이 불가능하다. 최초 CPU 검증 때는 기존 학습을 유지했으나, 이후 독해 성능을 우선해 SFT v2와 두 대기 큐를 정상 인터럽트로 중단했다. 저장된 best/last 가중치는 보존하고 수집 작업은 건드리지 않았다. 추출 헤드는 SFT v2가 아닌 완료된 medium base를 사용하므로, `runs/trid-medium-reader-dev-01`에서 2,000단계·실효 batch 16·micro-batch 4 학습을 별도로 시작했다. 잠긴 평가셋을 자동 실행하는 기존 큐 대신 학습 명령만 실행한다. 실제 MPS 메모리 감소량과 속도 향상 배수는 아직 측정하지 않았다.

최종 회귀 테스트는 227개 통과했다. 작은 batch 분할 전후의 실제 모델 손실·전체 파라미터 gradient·optimizer 업데이트와, 정답이 없는 batch에서 구간 헤드에 gradient가 생기지 않는 경우를 함께 확인한다. 과거 코드 복제본과 비트 단위 동일성을 비교하던 구현 종속 테스트는 제거하고, 학습 재개·실효 batch 동등성 등 동작 검증을 유지했다.

첫 본학습 결과를 보기 전에 `reports/reader-dev-01-protocol.json`에 개발 목표를 고정했다: 전체 480문항 strict ≥80%, 답 없음·충돌 각각의 잘못된 답 있음 판단 ≤5%, 충돌·KLUE·복수 문단 답 있음 과제 각각 strict ≥70%, 출력 가능한 답 있음 297문항의 잘못된 보류 ≤15%, 실행 예외 0건. 두 값 결합 40문항은 미지원으로 남겨 분모에서 빼지 않는다. 따라서 현 단일 구간 모델의 구조적 상한은 440/480이다. 이 값들은 내부 개발 목표이며 배포 승인이나 Qwen 동등성 기준이 아니다. 개발셋은 체크포인트 선택에도 사용되므로 독립 성능 검증을 대신하지 않는다.

### 개발 판정과 실제 기관 문서 점검

`scripts/analyze_extractive_dev.py`는 평가 결과를 다시 집계하기 전에 자료 해시·문항 ID·정답·과제·전체 분모를 확인한다. 누락·중복 문항으로 점수를 올릴 수 없고, 다른 실행의 결과는 등록한 후보의 합격으로 처리하지 않는다. 답 없음과 충돌의 false-presence를 따로 보며, 충돌을 `확인 불가`로 잘못 보류한 경우도 별도 오류다. 값 오류는 출처 오류와 구분하고, 값이 너무 길거나 짧게 잘린 경우를 나눈다. 실제 과거 시험 결과 480행으로 CLI를 실행해 23개 정답과 135개 false-presence를 재현했다. 이는 분석기 검증이지 새 모델의 성능 개선 결과가 아니다.

```sh
.venv/bin/python scripts/analyze_extractive_dev.py \
  --report reports/extractive-dev-medium-01.json \
  --protocol reports/reader-dev-01-protocol.json \
  --output reports/extractive-dev-medium-01-analysis.json
```

curriculum 검증셋에는 학습과 동일한 질문 문자열을 쓰는 합성 문항이 182/480개 있다. 근거 문서와 KLUE 질문 ID의 정확한 중복은 발견되지 않았지만, 템플릿에 익숙해진 효과와 새로운 기관 문서를 읽는 능력은 구분해야 한다.

이를 확인할 별도 소규모 자료 `data/institution-qa-frozen-01`을 후보의 해당 문항 예측을 보기 전에 고정했다. 실제 기관 보도문 30건에서 단일 사실 30문항과 근거 부재 10문항을 작성했고, 긍정 문항 중 10개에는 다른 문서의 방해 근거가 있다. 신청·환불 기한, 일반/청소년 추천 마감 구분, 지원 한도, 의무 대상 조건 등을 포함한다. 원문·출처 URL·기관 표시·이용 조건·문자 위치를 보존했다. 최대 입력은 653토큰이고 원문 위치와 프롬프트 왕복 변환을 확인했다.

이 자료는 Fable 작성, Astra의 과제 구성·위치 검토, 별도 Fable의 의미 검토를 거친 **LLM 작성 점검 자료**다. 사람이 주석한 독립 벤치마크나 대표성 있는 표본으로 부르지 않는다. 기본 모델의 기록된 사전학습 출처는 Wikipedia이며 보도자료 수집본을 사용하지 않았지만, 오래된 보도문이 Wikipedia에 인용됐을 가능성까지 배제하지는 못했다. 충돌·다중 값·검색·권한 통제는 이 40문항의 검증 범위가 아니다.

동결된 `test.jsonl` SHA-256은 `35803b6d474f5dd48b299610b007a1d6d405d675a54add4097c5ff1eed97d6d6`이다. 주 지표는 출력 전체가 사전에 승인한 `accepted_answers` 중 하나와 정확히 일치하는지다. 기존 평가기의 공백·문자 정규화 점수는 보조 지표로 구분한다. 후보 선택은 개발셋에서 끝낸 뒤 이 자료를 사용하며, 예측을 보고 정답 변형을 추가하거나 학습 자료에 섞지 않는다.

개발 판정기의 빈 과제 그룹 회귀까지 포함해 전체 테스트 234개 통과. 자료 검증은 토크나이저·문자 위치 확인만 수행했으며, 위 40문항에서 모델 예측이나 정답률을 아직 측정한 것은 아니다.

### 추출 헤드 양방향 attention 옵션 (2026-10-28)

`reports/extractive-dev-medium-01-analysis.json`에서 합성 과제는 거의 다 맞고 `rag_klue` 16/137·`rag_klue_missing` 28/63·`rag_noise` 19/40이 낮았다. 코드상 원인은 RAG 프롬프트에서 문단이 질문보다 앞에 오는데 몸통이 인과 attention이라 문단 토큰의 hidden state(구간 key)가 질문을 보지 못하고 마지막 토큰 query만 질문을 아는 것이다(조사 보고서 §1.4 b, 실험 1). `trid.model`의 `Attention/Block/Trid.forward`에 `attention='causal'|'bidirectional'`과 `lengths`(오른쪽 padding key 마스크, `[b,1,1,t]` bool 브로드캐스트) 인자를 추가했다. 기본은 causal이며 기존 호출자·생성·KV 캐시 경로는 그대로다(`bidirectional` + 캐시는 ValueError). RoPE는 바꾸지 않는다.

`ExtractiveReader(cfg, attention)`가 이 값을 몸통에 넘기고, `scripts/train_extractive.py --attention {causal,bidirectional}`(기본 causal)이 `training.json`의 `reader_attention`에 기록한다. `load_reader`는 이 키로 복원하며 키가 없는 과거 실행은 causal이다. `reader.safetensors`의 텐서 형식은 같다(모드는 가중치가 아니라 메타데이터). `read()` 결과에 `attention`을 넣는다. 회귀 테스트(`tests/test_extractive.py`): padding 배치와 행별 단독 forward 일치, causal 모드의 기존 경로와 비트 동일, 마지막 토큰 교란이 앞 토큰에 전파되는지(양방향만), `load_reader` 왕복, 캐시 거부. 3행 합성 자료 2단계 CPU 시험으로 경로 확인만 했으며 성능 수치는 아직 없다.

첫 본학습 `runs/trid-medium-reader-dev-01`(causal, 2,000단계, 2시간 6분)의 결과: 내부 검증 440행 strict 최고 0.589(2,000단계), 개발 전체 480문항 strict 259/480(53.96%). 사전 등록 목표(전체 ≥80%, KLUE ≥70%, 복수 문단 ≥70%, 답 없음 false-presence ≤5%)는 미달이다. 유형별: rag_value 38/40, rag_table 40/40, rag_revision 40/40, rag_conflict 40/40, rag_missing 38/40, rag_noise 19/40, rag_klue 16/137, rag_klue_missing 28/63, combine 0/40(미지원). 오류 137건 중 값 오류(출처 맞음) 97건(불일치 84·과장 9·절단 4), 답 없음에 답함 37건(KLUE 35), 잘못된 보류 23건(KLUE 21), 출처·값 모두 오류 24건(noise). 즉 합성 템플릿은 외웠고 실제 문장 독해와 방해 문단 선택이 안 된다. 사용자 목표는 내부 검증 strict 95%이며, 같은 설정에 `--attention bidirectional`만 바꾼 `runs/trid-medium-reader-dev-02-bidir`를 학습 중이다.

`runs/trid-medium-reader-dev-02-bidir` 결과(2,000단계, 2시간 6분): 내부 검증 strict 최고 0.632(1,900단계, causal 0.589), 개발 전체 278/480(57.9%). `reports/extractive-dev-medium-02-bidir-analysis.json`. 유형별: rag_noise 19→39/40(방해 문단 선택이 거의 해결), rag_klue 16→24/137, rag_klue_missing 28→29/63, rag_missing 38→30/40(악화), rag_conflict 39/40, value 37/40, table·revision 40/40. 남은 오류 162건 중 KLUE가 147건: 값 불일치 68, 답 없음에 답함 44, 잘못된 보류 30. 양방향 전환은 "어느 문단인가"를 고쳤고 "문단 안에서 어떤 구간인가"는 거의 못 고쳤다. KLUE 행이 200/440이므로 내부 검증 95%(418/440)는 KLUE ≥ 약 180/200을 요구한다. 참고로 KLUE-MRC EM은 RoBERTa-base(110M, 수십 GB 사전학습)에서 68.7이며, 이 몸통은 1억 토큰 Wikipedia 사전학습이다. 다음 실행은 KLUE 학습 행을 6,000→11,447로 늘린 `data/korean-rag-curriculum-v3`(valid 동일, `scripts/extend_rag_curriculum_klue.py`, seed 300326)로 4,000단계 양방향 학습 `runs/trid-medium-reader-dev-03-bidir-v3`다.

`runs/trid-medium-reader-dev-03-bidir-v3` 결과(4,000단계, 4시간 29분): 내부 검증 strict 최고 **0.539**(2,600단계)로 v2 자료 2,000단계(0.632)보다 낮다. 검증 손실은 1,800단계(4.13) 이후 4.98까지 상승했고 답 없음 정확도는 0.14~0.55 사이를 오갔다. KLUE 학습 행을 1.9배로 늘려도 개선이 없었으므로 병목은 자료량이 아니라 몸통이다(1억 토큰 Wikipedia 사전학습, 91M). **현 몸통으로는 KLUE 200행이 포함된 내부 검증 95%는 달성할 수 없다**고 판정한다. 선택지는 (1) 사전학습 규모 확대(조사 보고서 §8 실험 3, 10억 토큰 이상, 수 일), (2) 95% 목표를 제품 과제(합성 기관 문서 과제 240행 + 고정 기관문서 40문항)로 두고 KLUE 일반 독해는 별도 지표로 분리. 현재 최선 후보는 `runs/trid-medium-reader-dev-02-bidir`(합성 과제 235/240 = 97.9%, KLUE 53/200).

### 고정 기관문서 40문항 첫 측정과 사전학습 확대 결정 (2026-10-29)

사용자 결정: 제품 과제 기준 검증과 사전학습 확대를 함께 진행한다. `scripts/evaluate_institution_probe.py`(주 지표: 출력 전체가 `accepted_answers`에 정확히 포함; 보조: 기존 정규화 strict)로 `runs/trid-medium-reader-dev-02-bidir`를 `data/institution-qa-frozen-01`에 처음 평가했다(`reports/institution-probe-reader-02-bidir.json`, CPU 4.9초, 예외 0). **13/40(32.5%)**: 답 있음 12/30(잘못된 보류 5, 값 오류 11, 출처·값 오류 2), 답 없음 1/10(9건을 답 있음으로 판단, 대부분 P(present)>0.8). 두 지표는 40행 모두 일치했다. 경계 과장 3건(`15일 이내에`, `…(야간 전시는 9.30.`, `15억원, … 17억원`)은 정답 변형을 추가하지 않고 오류로 둔다.

결론: 합성 과제 97.9%는 템플릿 암기이며 실제 기관 문서 독해로 옮겨가지 않는다. KLUE 결과와 같은 원인(몸통)이므로 제품 과제 95%도 현 몸통으로는 불가능하다. 따라서 몸통의 이어 사전학습을 시작한다. 조사(PretrainPlan): 토큰화된 혼합 말뭉치 `data/mix-v3-dry-tokens`(법령 197M·보도 49M·위키 205M = 452M 학습 토큰)는 ds-8192 토크나이저라 base-v1과 호환되지 않아 base 토크나이저로 재토큰화한다(`data/mix-v3-basetok-tokens`). `trid.pretraining.pretrain(init_run=base-v1)`은 가중치만 잇고 optimizer·일정은 새로 시작하며 `stop_after`/`resume`으로 중단·재개된다. 측정 처리량 2,184 tok/s(fp32) → 10억 토큰에 5.3일, bf16+compile 추정 ~3일(부하 중 측정 +79%, 보장 아님). 10억 토큰은 혼합 말뭉치 약 2.2회 통과다.

재토큰화 결과 학습 485,434,084·검증 5,254,591·시험 5,027,236 토큰(토크나이저 sha `73551e65…`, base와 동일). 20단계 파일럿(`bf16` autocast + `torch.compile`, batch 2 × accumulate 32 = 65,536 토큰/단계, lr 3e-4, WSD 20% 감쇠): 검증 손실 3.173 → 3.012, **8,081 tokens/s**(MPS 단독 점유 시). 이 속도면 10억 토큰(15,259단계)은 약 34시간이다. 본 실행은 `scripts/train_medium_cpt_v1.py`가 `runs/trid-korean-medium-cpt-v1`에 250단계 세션을 반복하며(50단계마다 optimizer·RNG 포함 체크포인트, `reports/medium-cpt-stop.flag`로 정지, Ctrl-C 후 재실행으로 재개) `reports/medium-cpt-v1.log`에 기록한다. base-v1은 수정하지 않는다. 완료 후 할 일: 새 몸통에서 `--attention bidirectional` 독해 헤드 2,000단계 재학습 → 개발셋 480·기관문서 40문항 재평가. 사전학습 손실이 내려가도 독해 정답률이 오른다는 보장은 없으며, 그 결과로 판정한다.

같은 학습을 임대 GPU(AWS g6e L40S·g5 A10G)에서 돌리려면 `scripts/train_medium_cpt_gpu.py`를 쓴다. v1 스크립트와 같은 세션 반복·정지 플래그·재개 구조이며 데이터·초기 실행·출력·단계·배치·누적·lr·장치(`--device cuda`)·정밀도·컴파일을 인자로 받는다. 기본 batch 16 × accumulate 4 = 65,536 토큰/단계로 Mac 실행과 토큰/단계가 같다. `pretrain()`은 단계마다 batch × accumulate 행을 한 번에 뽑고 micro 손실을 accumulate로 나누므로 곱이 같으면 batch/accumulate 분할은 기대 갱신을 바꾸지 않고 메모리만 바꾼다. 런타임은 `cuda`를 인식하며(`auto`는 cuda → mps → cpu 순), GPU에서의 실제 실행은 아직 검증하지 않았다.

### 추출 헤드 v2: 이웃 글자 키·근거를 보는 존재 헤드·가중 손실·macro 선택 (2026-10-29)

독해 논리 감사(ReaderLogicAudit)에서 찾은 세 가지 구조 문제: (1) 구간 키가 글자별 특징에 선형이라 토큰 안의 경계 선택이 이웃 글자와 무관했다(경계 과장·절단), (2) 존재 헤드가 마지막 hidden state만 보고 구간 점수를 모르며, 학습 자료의 67/22/10 비율에 가중 없는 CE를 썼다, (3) 전체 strict로 체크포인트를 고르면 답 없음 행의 흔들림이 가려졌다.

`trid/extractive.py`의 `ExtractiveReader(cfg, attention, head)`에 `head='v1'|'v2'`(`HEADS`)를 추가했다. v1은 기존 `trid.reader.Reader` 배치·연산과 동일하며 state_dict 키도 같다. v2는 `encode()`가 더한 `prev_ids`/`next_ids`(마스크 안 이웃 글자 id + 1, 마스크 밖·공백·문단 끝은 0; `NEIGHBOURS`=513)를 `batch()`가 함께 넘기고, 키 = MLP(concat(토큰 hidden, 글자 임베딩 + 토큰 내 오프셋 + 이전 글자 + 다음 글자))(`key_mlp`: dim+32→256→GELU→128, 2×64), 존재 = MLP(concat(마지막 hidden, detach한 구간 근거 3개 — `span_evidence`: MAX_SPAN 창 안 start+end 최대, start·end logsumexp))(`presence`: dim+3→64→GELU→3). 구간 근거는 detach하므로 존재 CE는 구간 점수를 바꾸지 않는다. `training.json`의 `reader_head`로 복원하며 키가 없는 과거 실행은 v1이다(`runs/trid-medium-reader-dev-02-bidir`는 그대로 로드되고 기관문서 40문항 13/40이 재현됐다). `read()` 결과에 `head`를 넣는다.

`loss_terms`/`loss`에 `weights`(PRESENCE 순 3개)를 더했다. 행별 존재 CE에 kind 가중을 곱하고 분모는 행 수 그대로이므로 chunk 분할 불변성은 유지된다. `scripts/train_extractive.py`의 `presence_weights()`가 학습 행의 kind 역빈도를 평균 1로 정규화해 `presence_class_weights`로 기록하고, 학습 단계와 검증 손실에 똑같이 적용한다. 체크포인트 선택은 `macro_strict_accuracy`(답 있음·없음·충돌 strict의 단순 평균, 행이 없는 kind 제외; `selection_metric`)로 바꾸고 동률은 `valid_loss`가 낮은 쪽이다. 전체 `strict_accuracy`는 진단으로 남는다. `--head {v1,v2}` 기본은 v2. 회귀 테스트: `tests/test_extractive.py`(v1 동일성, v2 형태와 이웃 민감도, 가중 손실 chunk 불변, head 복원), `tests/test_train_extractive.py`(macro 순서, 가중치, v2 양방향 2단계 CPU 스모크). 이미 끝난 실행의 수치는 바뀌지 않는다.
