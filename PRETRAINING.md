# trid 한국어 기초 사전학습

이 문서는 최초 12,000문서·2000단계 파일럿의 기록입니다. 2026-10-27에 추가한 문서 확대·고정 토크나이저·`pretrain --init-run` 이어 학습은 [한국어 개선 실행 기록](KOREAN_RUN.md)에 있습니다. `--resume`은 여전히 원래 일정을 유지하며, 완료한 실행에서 새 일정을 시작할 때는 다른 출력 경로와 `--init-run`을 사용합니다.

현재 반복 출력의 주된 배경은 1,011개의 가상 업무 예제만으로 무작위 모델을 학습했기 때문입니다. 반복 차단이나 정해진 인사말로 결과를 가리는 대신, 한국어 문장 자체를 학습하는 경로를 추가했습니다. 단, 적은 학습량만으로 반복과 오답이 해결된다고 보장할 수는 없습니다.

## 진행 순서

1. **기초 사전학습:** 한국어 문서의 모든 다음 토큰을 예측하도록 학습합니다.
2. **대화 학습:** 직접 사전학습한 가중치와 토크나이저를 그대로 이어 받아 검수된 질문·답변을 학습합니다.
3. **회사 업무 학습:** 일반 대화 성능이 확인된 뒤 권한이 있는 회사 자료와 업무 사례를 추가합니다.
4. **경량화:** 정확도와 실제 Raspberry Pi 성능을 확인하며 양자화를 적용합니다.

어느 단계에서도 Qwen·Llama 같은 외부 사전학습 가중치는 사용하지 않습니다. 사전학습 데이터가 공개 문서라는 것과 기존 모델을 사용한다는 것은 서로 다른 의미입니다.

## 이번에 준비한 자료와 설정

- 공개 한국어 Wikipedia 문서 12,000개, 2023-11-01 스냅샷.
- 학습 10,777개 / 검증 606개 / 시험 617개. 문서 전체 텍스트의 해시로 먼저 분리했습니다.
- 학습 문서만 사용해 새로 학습한 byte-level BPE, 어휘 8,192개.
- 학습 20,252,886토큰 / 검증 1,174,067토큰 / 시험 1,049,115토큰.
- 모델: 8층, hidden 384, head 6, SwiGLU 1,024, 문맥 256. 약 1,730만 파라미터.
- 실제 회사 자료는 읽거나 학습하지 않았습니다.

첫 번째 한국어 샤드의 앞쪽에서 조건에 맞는 문서를 골랐으므로 대표성 있는 무작위 한국어 자료가 아닙니다. 완전히 같은 문서는 제거하지만 유사 문서 제거는 아직 없습니다. 백과사전 문체만 학습하므로 일상 대화·업무 대화는 별도 데이터가 필요합니다.

원문 링크와 기여자 이력 링크는 각 JSONL 행에, 소스 revision과 SHA256은 `data/wiki-ko-pilot/provenance.json`에 보존했습니다. 데이터 카드에는 CC BY-SA 3.0/GFDL, 현재 Wikimedia 안내에는 CC BY-SA 4.0이 표기되어 있어 두 출처를 함께 기록했습니다. 이 로컬 시험을 기업 배포에 대한 라이선스 판정으로 해석하지 마세요. 원문을 재배포할 때는 출처와 해당 이용 조건을 유지해야 합니다.

출처: https://huggingface.co/datasets/wikimedia/wikipedia

Wikimedia 이용 조건: https://dumps.wikimedia.org/legal.html

## 완료된 사전학습과 재개 방식

기본 실행은 **2,000단계까지 완료**했습니다. 현재 가중치의 검증 손실은 5.358951이며, 다음 `--resume` 명령은 완료된 스케줄에 단계를 추가하지 않습니다. 새로 시작했거나 중간에 멈춘 실행에서만 남은 단계가 이어집니다.

```sh
cd '/Users/eomgyuhyeon/사업/trid'
.venv/bin/trid pretrain --resume --stop-after 200
```

`--stop-after 200`은 **이번 실행에서 최대 200단계를 추가**한다는 뜻입니다. 저장된 전체 2,000단계 스케줄과 배치·학습률을 유지합니다. 끝나면 체크포인트를 저장하고 프로세스가 종료됩니다. 백그라운드 실행이나 자동 재실행은 하지 않습니다.

전체 목표 2,000단계는 약 409만 6천 개의 토큰 위치를 학습하는 초기 실험 설정이며, 모델 완성 기준이 아닙니다. 약 2,025만 개의 학습 토큰에서 임의 구간을 뽑으므로 문서 전체를 한 번 읽은 것과도 다릅니다. 이것만으로 자연스러운 대화가 된다고 기대해서는 안 됩니다. 더 긴 실험은 학습 속도·검증 손실·출력 품질을 함께 보고 별도 스케줄로 잡습니다.

50단계마다 검증 손실과 처리 속도를 기록하고, 가중치·AdamW 상태·배치 난수 상태를 저장합니다. Ctrl+C로 중단하면 **마지막으로 완전히 저장된 체크포인트**에서 재개합니다. 최근 두 개의 학습 체크포인트와 검증 손실이 가장 낮은 추론용 가중치를 유지합니다. 옵티마이저도 Safetensors로 저장하며 pickle을 읽지 않습니다.

현재 빠른 검증은 고정 seed로 선택한 검증 문서 토큰 구간 8배치(총 16,384개 토큰 위치)에서 수행합니다. 전체 검증 말뭉치를 모두 평가한 값은 아닙니다.

## 문장 이어쓰기 확인

```sh
.venv/bin/trid complete '우리 회사는' --max-tokens 40
```

이 명령은 질문에 답하는 대신 주어진 문장 뒤를 생성합니다. 초기 가중치에서는 반복과 비문이 계속 나올 수 있습니다. 사전학습 모델에 `chat`을 지정하면 대화 학습 전이라는 오류를 표시합니다. 새 모델의 성능을 기존 `chat`으로 판단하지 마세요.

## 대화 학습

사전학습 성능을 확인한 뒤 별도의 검수된 대화 데이터가 필요합니다. `data/private/dialogue/train.jsonl`와 `valid.jsonl`에 다음 형식으로 준비합니다.

```json
{"prompt":"안녕", "answer":"안녕하세요. trid입니다."}
```

인사 예제 몇 개를 복제하는 방식은 피합니다. 다양한 표현, 질문 의도, 짧고 긴 답변, 자료 부족 상황, 회사 업무와 무관한 질문 등을 포함하고, 독립적인 평가용 표현을 남겨야 합니다. 현재는 선택적인 `history` 배열로 여러 차례의 대화도 학습합니다. 각 항목은 `{"user":"앞선 질문","assistant":"앞선 답변"}` 형식입니다. 앞선 대화는 입력으로만 사용하고 현재 답변에만 손실을 계산합니다.

```sh
.venv/bin/trid train \
  --init-run runs/trid-pretrain-pilot \
  --data data/private/dialogue \
  --output runs/trid-dialogue-01 \
  --steps 300 --batch-size 8 --lr 0.0001
```

위 수치는 초기 실험용이며 검증 결과로 조정해야 합니다. 자체 사전학습 가중치에서 대화 학습을 실행했습니다. 진행 기록과 평가 방법은 [DIALOGUE.md](DIALOGUE.md)에 있습니다. `--init-run`은 가중치를 새로 초기화하지 않고 로컬 사전학습 결과를 이어 받으며 토크나이저도 유지합니다.

## 처음부터 재현

현재 다운로드와 토큰화는 이미 끝났으므로 다시 할 필요가 없습니다. 새 환경에서는:

```sh
uv sync --frozen --extra corpus --python 3.12
.venv/bin/trid fetch-wiki
.venv/bin/trid prepare-corpus
.venv/bin/trid pretrain --steps 2000 --stop-after 200
```

다운로드는 공개 텍스트 샤드 약 400MB이며 고정 revision과 SHA256으로 검증합니다. 외부 모델 파일이나 회사 자료를 전송하지 않습니다. 이후 토큰화·학습·추론은 로컬에서 수행합니다. `runs/`와 `data/`의 결과를 공유할 때는 학습 데이터와 원문 출처를 함께 관리하세요.

## 품질 판단

검증 손실 감소는 다음 단어 예측이 나아진다는 신호일 뿐, 대화·추론 성능을 보장하지 않습니다. 다음 항목을 따로 확인해야 합니다.

- 학습에서 보지 않은 한국어 문장의 예측 손실.
- 처음 보는 표현에 대한 문법·의미 일관성.
- 반복·중간 종료·문장 붕괴 빈도.
- 대화 학습 이후의 지시 따르기와 회사 업무 정확도.

이번 사전학습에서 시험 split은 학습과 체크포인트 선택에 사용하지 않습니다. 기존 가상 업무 시험 93.7%는 이 모델의 지표가 아닙니다.

## 다음 사전학습 말뭉치 준비 (2026-10-27)

91M 모델(`runs/trid-korean-medium-base-v1`)은 위키 약 8천만 토큰으로 학습하고 있습니다. 이 크기에 비해 적은 양이고, 문체도 RAG 대상인 공문체와 다릅니다. 다음 실행에 쓸 약 10억 토큰을 출처와 라이선스가 분명한 공공 자료로 모읍니다. 출처별 조사 결과와 근거 링크는 아래 표에 요약했습니다.

| 출처 | 이용 근거 | 수집기 | 시범 수집 | 전체 추정 |
|---|---|---|---|---|
| 국가법령정보센터 (법령·행정규칙·자치법규·판례·헌재·해석례·행정심판) | 저작권법 제7조 비보호 저작물. 법제처: "영리 목적의 이용을 포함하여 자유로운 활용이 보장됩니다" ([이용안내](https://open.law.go.kr/LSO/information/guide.do)) | `scripts/collect_law_go_kr.py` | 7종 × 100건, 640건 유지, 문자/토큰 1.68 | 약 11억 토큰, 요청 45만 건, 약 5.7일 |
| 정책브리핑 보도자료 2015년~ | 각 쪽의 "이 자료는 텍스트에 한하여 공공누리 제1유형(출처표시)의 조건에 따라 이용할 수 있습니다". 이 문구가 있는 자료만 유지 | `scripts/collect_korea_briefing.py` | 148건 중 공공누리 1유형 98%, 137건 유지, 문자/토큰 1.60 | 약 5.3억 토큰, 약 13.7일 |
| 한국어 위키백과 전체 (3개 샤드) | CC BY-SA 4.0 / GFDL ([dumps legal](https://dumps.wikimedia.org/legal.html)) | `scripts/collect_wiki_ko_full.py` | 전체 수집 완료: 379,068문서, 5.47억 자 | 약 3.08억 토큰 |

토큰 수는 현재 8k 토크나이저 기준입니다. 다음 실행은 새 모델 계열이라 토크나이저를 다시 학습하므로 실제 토큰 수는 달라집니다. AI Hub와 모두의 말뭉치는 학습한 모델을 공개하려면 별도 확인이나 승인이 필요해서 제외했습니다. 웹 말뭉치(HPLT, FineWeb-2)는 원문 저작권이 남아 있으므로, 쓰더라도 공공 도메인으로 걸러 적은 비율만 보충용으로 넣습니다.

### 공통 형식
수집기는 모두 `DIR/docs.jsonl`, `raw/`, `provenance.json`, `ATTRIBUTION.txt`, `pilot-stats.json`을 씁니다. 각 행에는 `id, title, source, text, sha256, license, attribution, group, date, kind`가 있습니다. `group`은 한 항목의 판·사본을 묶는 키(법령 ID, newsId)입니다. 전화번호와 이메일은 지우고, 200자 미만 문서는 버립니다. 시범 수집 통계는 `pilot-stats-100.json`(법령)과 `pilot-stats-150.json`(정책브리핑)에 따로 보관했습니다.

### 섞기와 중복 제거 (`scripts/build_pretraining_mix.py`)
```sh
.venv/bin/python scripts/build_pretraining_mix.py \
  --source data/law-ko-source:350M --source data/korea-kr-briefing-source:300M \
  --source data/wiki-ko-full-source:250M --output data/korean-foundation-v3
.venv/bin/trid prepare-corpus --source data/korean-foundation-v3 --output data/korean-foundation-tokens-v3
```
- `--source`의 순서가 우선순위입니다. 중복이 있으면 먼저 적은 출처의 문서를 남깁니다.
- 중복 제거 전에 품질·PII 단계(`trid/korean_quality.py`, 의존성 `uv pip install --python .venv/bin/python kiwipiepy==0.24.1 ko-pii==1.22.15`)를 거칩니다. `--no-quality`로 끄고, `--quality-profile fineweb2|public-docs`(기본 `public-docs`)로 임계값을 고릅니다.
  - 줄 정리(C4식, 한국어 조정): javascript 안내 줄, 40자 이하의 사이트 메뉴 줄(로그인·개인정보처리방침 등), 1,000자 초과 단어가 있는 줄만 지웁니다. 종결 부호 없는 줄은 개조식 문장 때문에 지우지 않습니다.
  - PII: ko-pii(MIT)로 주민·외국인등록번호, 전화, 이메일, 계좌, 카드, 여권, 운전면허를 찾아 `[주민번호] [전화] [이메일] [계좌] [카드] [여권] [운전면허]`로 바꿉니다. 이름·주소는 오탐이 많아 끕니다. ko-pii는 공백을 접어 매칭하므로 표를 펼친 숫자열(`2010 2011 2012 2013`)을 카드·전화로 잡는데, 원문 구간이 형식과 정확히 맞고 앞뒤에 숫자·영문이 붙지 않은 것만 바꿉니다(3,000건 샘플에서 전화 오탐 238건 → 5건).
  - 품질: FineWeb-2 `kor_Hang.yml`의 Gopher 반복(dup_line_frac 0.374, top n-gram, dup n-gram)·FineWeb(line_punct 0.062, char_dup 0.1, new_line_ratio 0.196)·Gopher 품질(평균 어절 길이 1~6, 알파벳 어절 ≥0.791, 불용어 2개 이상, 50~100k 어절) 값을 kiwipiepy 형태소 단위로 그대로 적용합니다. GlotLID 언어 점수는 한글·한자·로마자 글자 비율(≥0.6, ≤0.3, ≤0.3)로 대신합니다.
  - `public-docs` 프로파일은 법령·보도자료 파일럿 측정에 맞춰 셋만 완화합니다: dup n-gram 임계값 2배(원값은 법률 73%·법령해석례 92%를 상투구 때문에 지움), 알파벳 어절 비율 0.7(원값은 보도자료 26%를 숫자·기호 때문에 지움), 최대 어절 100만(원값 10만은 공직선거법·관세법을 지움). 제거 사유별 건수는 `provenance.json`의 `filter_counts`(`출처:quality:사유`, `출처:pii:라벨`, `출처:lines_removed`)에, 제거 문서 목록은 `quality-removed.json`에 남습니다.
  - 수집본 3,000건씩 샘플(법령 2,948건 4,950만 자·보도자료 3,000건 910만 자)에서 법령 3.5%, 보도자료 25.6%가 제거됐습니다. 보도자료 제거는 line_punct_ratio 401·dup_line_frac 221·list_ratio 70이 대부분이며, 검토한 사례는 HWP 추출이 깨져 줄이 토막 나거나 본문이 4번 반복된 문서였습니다. 처리 속도는 이 5,948건·5,860만 자에 4분(Kiwi 4스레드, 수집기 2개·학습 1개와 동시 실행)이었습니다.
- 공백을 지운 NFC 본문으로 출처 간 완전 중복을 제거합니다.
- 유사 중복은 5단어 shingle MinHash(64개, 4개씩 16밴드)로 후보를 찾습니다. 추정치가 0.6 이상인 후보만 본문을 다시 읽어 정확한 Jaccard를 계산하고, 0.8 이상이면 제거합니다. 위키 전체로 시험했을 때 추정치만 쓰면 제거 237건 중 52건이 실제 Jaccard 0.70~0.80인 다른 문서였습니다. 정확한 값으로 확인하도록 바꾼 뒤에는 214건을 제거했고, 모두 0.8 이상이었습니다.
- train/valid/test는 `group` 해시로 나눕니다(기본 각 1%). 같은 법령의 여러 판이 학습과 시험에 나뉘어 들어가지 않습니다.
- 출처별 토큰 상한까지만 해시 우선순위로 고르고, 반복해서 채우지는 않습니다. 모자란 양은 `provenance.json`의 `shortfall_tokens`에 기록합니다.
- 위키 전체 37.9만 문서 기준으로 106초, 최대 메모리 0.79GB였습니다. 본문은 메모리에 두지 않고 파일 위치만 기억합니다.

### 전체 수집 실행
- **정책브리핑:** 2026-10-27에 시작했습니다. 첨부 파일은 본문을 뽑은 뒤 지웁니다. 첨부를 남기면 약 416GB가 필요합니다. 로그는 `reports/korea-briefing-crawl.log`, 상태 확인은 `.venv/bin/python scripts/collect_korea_briefing.py --status`입니다. 중단되면 같은 명령으로 다시 실행하면 이미 받은 항목은 건너뜁니다. HWP 첨부는 2026-10-28부터 `rhwp`(MIT, `~/.local/bin/rhwp`)로 추출합니다. 이전의 `hwp5txt`(pyhwp, AGPL-3.0)와의 비교는 `reports/hwp-extractor-parity.json`에 있습니다.
  ```sh
  nohup nice -n 15 .venv/bin/python scripts/collect_korea_briefing.py --output data/korea-kr-briefing-source --drop-attachments > reports/korea-briefing-crawl.log 2>&1 &
  ```
- **법령:** 공개 시험 키 `test`로 45만 건을 요청하지 않습니다. open.law.go.kr에서 Open API를 신청해 승인받은 키(OC)를 넣어 실행합니다.
  ```sh
  TRID_LAW_OC=<발급받은 ID> nohup nice -n 15 .venv/bin/python scripts/collect_law_go_kr.py --output data/law-ko-source > reports/law-crawl.log 2>&1 &
  ```

## 학습 루프 옵션과 다음 실행 레시피 (2026-10-28)

`reports/korean-oss-opportunities.md` 3절의 조사 결과를 `trid/pretraining.py`의 `pretrain()` 선택 인자로 옮겼습니다. 모두 기본값이 기존 동작과 같고(기본값으로 3단계를 돌린 metrics.json과 가중치가 옵션 추가 전 코드와 비트 단위로 같음을 `tests/test_pretraining.py`에서 확인), 체크포인트 형식은 바뀌지 않았습니다. 값은 모두 `training.json`에 기록됩니다.

| 인자 | 기본 | 동작 |
|---|---|---|
| `accumulate` | 1 | gradient accumulation 횟수. 옵티마이저 1스텝 = `batch_size × accumulate × context_length` 토큰. 손실은 1/accumulate로 나눠 더하므로 `accumulate=2, batch 2`와 `accumulate=1, batch 4`의 학습 손실 차이는 FP32에서 1e-5 이내(테스트). 배치 난수는 스텝마다 한 번만 뽑아 쪼개므로 두 설정의 토큰 순서가 같습니다. `tokens_seen`·`lineage_tokens_seen`·`tokens/s`가 이 크기를 반영하고, `--resume`은 저장된 값을 복원합니다(`--init-run`의 `inherited_tokens_seen`도 부모의 accumulate를 곱함). |
| `precision` | `fp32` | `bf16`이면 학습 forward·손실과 검증을 `torch.autocast(device_type, dtype=bfloat16)`로 실행합니다(검증도 같은 autocast — 2026-10-29부터). master weight·옵티마이저 상태·저장은 FP32 그대로입니다. 실행마다 지정하며 `--resume`이 복원하지 않습니다. |
| `compile` | `False` | `torch.compile(model)`을 학습 forward에만 씁니다. 검증과 체크포인트는 원래 모듈을 쓰므로 `model.safetensors` 키에 `_orig_mod.`가 붙지 않습니다. 첫 스텝에 컴파일 시간(작은 모델 CPU 기준 약 15초)이 듭니다. |
| `schedule` | `cosine` | `cosine`: 기존대로 warmup 후 코사인으로 최대 lr의 10%까지. `wsd`: warmup → `lr` 고정(stable) → 마지막 `decay_fraction` 구간에서 0까지 선형 감소(warmup–stable–decay, [MiniCPM arXiv 2404.06395](https://arxiv.org/html/2404.06395v2) §4.2 "디케이 10% 스텝이면 충분", SmolLM2는 20%). `--resume`은 `planned_steps`와 함께 복원합니다. |
| `decay_fraction` | 0.1 | WSD의 디케이 비율 (0, 1]. 100스텝·0.1이면 90스텝까지 lr, 95스텝 0.5·lr, 100스텝 0. |
| `z_loss` | 0 | PaLM식 $z\_loss \cdot \log^2 Z$ 항 ([arXiv 2204.02311](https://arxiv.org/pdf/2204.02311)). bf16으로 바꿀 때 PaLM 1e-4 또는 Kanana 5e-6을 권장. 0이면 기존 `loss_for` 경로 그대로입니다. |

현재 CLI(`trid pretrain`)에는 이 인자가 아직 없으므로 Python에서 `pretrain(..., accumulate=32, precision="bf16", compile=True, schedule="wsd")`처럼 호출하거나 파이프라인 스크립트에 넘깁니다. 91M 학습 job이 MPS를 쓰는 동안 위 검증은 모두 CPU에서만 했고, MPS에서의 bf16·compile 수치(보고서 측정 +79% 처리량)는 다음 파일럿에서 다시 잽니다.

## 체크포인트 주기·epoch 샘플링·소스 가중치·weight decay 그룹 (2026-10-29)

`pretrain()`에 인자 네 개를 더했습니다. 모두 `training.json`에 기록되고 `--resume`이 복원하며, 키가 없는 이전 실행(예: 진행 중인 cpt-v1)은 괄호의 옛 동작으로 재개합니다. 새 실행의 기본값은 `decay_norms`만 옛 동작과 다릅니다.

| 인자 | 기본 | 옛 실행 재개 시 | 동작 |
|---|---|---|---|
| `checkpoint_every` | `eval_every` | `eval_every` | 검증은 `eval_every`마다, 옵티마이저 체크포인트(약 1.1 GB 쓰기)는 `checkpoint_every`(`eval_every`의 배수)마다. best 가중치(`model.safetensors`)도 검증 손실이 개선된 **체크포인트 스텝**(또는 세션 마지막 스텝)에만 씁니다. 그래서 `best_step`은 항상 저장된 체크포인트와 일치합니다. |
| `sampling` | `replacement` | `replacement` | `epoch`: `train.bin`을 겹치지 않는 `context_length` 창으로 잘라 epoch마다 `seed + epoch`로 시드한 순열로 한 번씩 봅니다(`trid.pretraining.EpochSampler`). 위치 `{"epoch", "cursor"}`는 체크포인트 `state.json`의 `sampler`에 저장되어 재개 시 그 자리에서 이어집니다. `training.json`의 `epoch_windows`가 epoch당 창 수, metrics의 `epoch`가 현재 epoch. replacement로 만든 실행은 재개해도 replacement를 유지합니다(바꾸려면 새 실행). |
| `source_weights` | `None` | `None` | `sampling='epoch'` 전용. `{kind: weight}`로 창 시작 토큰이 속한 문서의 `kind`별 배수를 지정합니다. $w$ 배수는 창을 $\lfloor w \rfloor$번 반복하고 소수 부분 확률로 한 번 더(시드 고정) — `2`는 두 배, `0.5`는 절반, `0`은 제외. 데이터 폴더의 `sources.json`(아래)이 필요합니다. |
| `decay_norms` | `False` | `True` | `False`: 2차원 이상 파라미터(임베딩 포함)에만 weight decay 0.1, RMSNorm gain 등 1차원 파라미터는 0. `True`: 모든 파라미터 decay(옛 동작). 체크포인트 형식은 파라미터 순서 기준이라 그룹 수와 무관합니다. |

같이 바뀐 커널 수준 동작(인자 없음): CUDA에서 `AdamW(fused=True)`와 `torch.set_float32_matmul_precision("high")`, 검증도 학습과 같은 autocast, micro-batch마다 하던 `isfinite` 동기화를 스텝당 한 번(누적 손실)으로 — 비유한 손실이면 옵티마이저 갱신 전에 그대로 `RuntimeError`.

`sources.json`은 `trid.corpus.prepare`가 토큰 파일 옆에 쓰는 사이드카로 split별 문서 시작 오프셋(`<|bos|>` 위치)과 JSONL의 `kind` 색인(`kinds` 이름표; `kind`가 없는 행은 `unknown`)을 담습니다. `.bin`과 원본 JSONL만 읽으므로 이미 만든 폴더에도 `trid.corpus.write_sources(source_jsonl_dir, token_dir)`로 추가할 수 있습니다(`manifest.json`을 건드리지 않아 재개 지문이 바뀌지 않음). mix-v3의 kind: `law`, `admrul`, `ordin`, `press_release`, `wikipedia`.

`scripts/train_medium_cpt_gpu.py`는 `--eval-every 100 --checkpoint-every 1000 --sampling epoch --no-decay-norms --stop-after 5000`이 기본이고 `--source-weights press_release=2 ordin=0.5`(또는 JSON 하나)를 받습니다. 테스트(`tests/test_pretraining.py`): epoch 샘플러가 창을 epoch당 정확히 한 번 보고 커서에서 재개, 파라미터 그룹 분리, 체크포인트 주기, 새 키 없는 `training.json` 재개가 옛 동작과 비트 단위로 같음, 사이드카 작성과 가중 샘플링, GPU 스크립트 CPU 2스텝 smoke.

### 다음 실행 제안 (`runs/trid-korean-medium-base-v1` 이후)

현행 91M 실행은 스텝당 2,048토큰(batch 2×1,024), lr 3e-4, cosine입니다. 비교 레시피는 llm.c GPT-2 124M 재현이 **스텝당 524,288토큰·lr 6e-4·warmup 700·선형 디케이**([llm.c #481](https://github.com/karpathy/llm.c/discussions/481)), SmolLM2-135M이 **스텝당 1,048,576토큰·lr 3e-3·warmup 2,000·WSD(20% 선형 디케이)**([config](https://raw.githubusercontent.com/huggingface/smollm/main/text/pretraining/smollm2/config_smollm2_135M.yaml))입니다. 제안 설정:

- `batch_size=2, accumulate=64` → 스텝당 131,072토큰(메모리는 micro-batch 2 기준 그대로). 10억 토큰이면 약 7,600스텝.
- `schedule="wsd", decay_fraction=0.1` — 수집이 길어져 총 토큰이 유동적인 상황에서 stable 구간의 어느 체크포인트에서든 디케이를 붙일 수 있습니다. `--resume`은 `planned_steps`를 고정하므로 디케이 시점을 바꾸려면 `--init-run`으로 새 실행을 만듭니다.
- lr는 128k 토큰 배치 기준 6e-4·1e-3·2e-3 세 값으로 20M 토큰 파일럿을 돌려 도메인별 BPC로 고릅니다(보고서 §3.2). warmup은 현행 `min(100, steps//10)`이 유지됩니다.
- `precision="bf16", compile=True`는 250스텝 파일럿에서 FP32와 검증 손실 차이를 `reports/`에 기록한 뒤 채택하고, 채택 시 `z_loss=1e-4`(PaLM) 또는 `5e-6`(Kanana)을 함께 켭니다.

### CPT 이후 독해기 factorial 비교 (2026-10-29)

- 실행기: `scripts/run_reader_factorial_gpu.sh`, S3 `trid-training-148761639313/src/reader-ablation-8.sh`; 원격 systemd 단위 `trid-reader-factorial-01`.
- `trid-cpt`와 재시작 단위 `trid-cpt-resumed` 종료 및 CPT 메타데이터 `status=complete`를 모두 확인한 뒤 시작한다. 기존 CPT 실행 중에는 소스를 교체하지 않는다.
- 몸통 base-v1/CPT-v1 × 헤드 v1/v2 × curriculum v2/v4의 8조합. 모든 조합은 bidirectional, 2,000 steps, batch/micro-batch 16, lr 1e-4, seed 271029, eval 100으로 고정한다.
- 개발셋은 동일한 v2 validation으로 평가한다. 기관문서 probe는 진단용이며 반복 사용했으므로 새로운 독립 최종 평가로 주장하지 않는다. v4와 v2의 데이터량이 달라 같은 step 수는 같은 epoch 수가 아니다.
- 결과 이름은 `reader-factorial-01-{backbone}-{head}-{data}`. 기존 미완료 디렉터리는 덮어쓰지 않고 중단한다. 실행별 모델·보고서를 S3에 보존하고 원격 가중치의 SHA256을 로컬과 비교한 뒤 이번 실행의 로컬 safetensors만 회수한다. 실행 전 여유 공간 1.5GiB 미만이면 중단한다.
- 이 비교는 세 묶음의 효과를 분리한다. 헤드 내부 변경·합성 hard negatives·TyDi 및 사전학습 최적화 각각의 기여도를 분리하는 실험은 아니다.

후속 데이터 제거 비교는 `scripts/ablate_reader_data.py --remove {tydi,synthetic-hard,klue-hard} --output data/<새경로>`로 준비한다. v4의 해당 task만 제거하며 남은 행의 바이트·순서와 검증셋은 보존한다. synthetic-hard는 긍정/부정 예제를 함께 제거한다. 실제 생성 결과는 각각 train 35,063 / 33,661 / 32,161행이며 검증 SHA256은 모두 v2와 같다. 이는 데이터 준비 결과이지 성능 측정 결과가 아니다.

결과를 로컬 reports에 내려받은 후 `scripts/summarize_reader_factorial.py --output reports/<새파일>.json`으로 집계한다. dev/probe 모델 해시, 실행 이름, 평가 데이터 해시, 480/40행 분모와 정답 집계를 확인한다. 미완료 조합은 0점 대신 pending으로 남긴다. 역사적 실측 보고서의 임시 복사본으로 278/480 및 13/40 집계를 재현했고, 모델 해시 불일치는 거부했다. 후속 제거 실험은 몸통·헤드·학습 설정을 고정해 비교하며, probe 결과로 자동 모델 선택하지 않는다.

`scripts/run_reader_removals_gpu.sh`는 systemd `trid-reader-removals-01`로 실행한다. CPT와 8조합 서비스 종료 후 8개 보존 완료 마커를 확인한 다음, CPT-v1 몸통·v2 헤드를 고정하고 TyDi / synthetic-hard / klue-hard 제거 데이터를 각각 2,000단계 학습한다. 다른 조건과 보존 절차는 factorial 실행과 같다. 비교 기준은 factorial의 CPT-v1/head-v2/data-v4이며, 결과를 보고 몸통·헤드를 다시 선택하지 않는다. 최초 8조합 중 실패가 있으면 제거 실험도 시작하지 않는다.
