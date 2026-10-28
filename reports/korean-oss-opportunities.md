# trid에 도입할 수 있는 한국어 NLP·소형 LM 오픈소스 조사

작성일: 2026-10-28. 대상: `trid` 91M 모델(`runs/trid-korean-medium-base-v1`, 12층·hidden 768·문맥 1,024·자체 8,192 byte-level BPE)과 다음 약 10억 토큰 사전학습(법령·행정규칙·판례·보도자료·위키) → RAG SFT → 자체 평가셋 파이프라인.

전제(프로젝트 정책): **외부 사전학습 가중치는 쓰지 않는다.** 이 문서는 생성 모델 가중치가 아닌 **코드·도구·데이터 처리 방법·평가 벤치마크·토크나이저 설계·학습 레시피**만 다룬다. 5절의 검색(retrieval) 모델만 예외적으로 "정책 적용 여부가 결정되지 않은 질문"으로 표시한다.

표기 규칙
- 라이선스는 1차 출처(GitHub LICENSE, HF 모델·데이터 카드, 논문)를 2026-10-28에 직접 확인한 값이다. 상업적 이용이 가능한지 여부는 "코드·도구를 제품 파이프라인에 쓰는 경우" 기준이다.
- "(측정)"은 이 조사에서 로컬 Mac으로 직접 측정한 값이다. 토크나이저 비교에 쓴 외부 tokenizer.json 파일들은 측정 후 삭제했고 trid 코드·데이터에 넣지 않았다.
- 판정: **도입**(다음 실행에 넣을 것), **실험**(소규모 비교 후 결정), **보류**(가치 있으나 지금은 아님), **제외**(라이선스·규모·정책상 부적합).
- 추정·추론은 `[INFERENCE]`로 표시한다.

---

## 0. 요약

| 영역 | 핵심 발견 | 판정 |
|---|---|---|
| 토크나이저 | 자체 8k BPE의 한국어 글자/토큰(1.60–1.89)은 이미 kanana(=Llama-3 128k, 1.54–1.62)·EXAONE(102k, 1.70–1.79)·HyperCLOVA X SEED(110k, 1.83–1.88)·Polyglot-Ko(30k, 1.80–1.83)와 같은 급이다. 개선 여지는 **어휘 크기**(16k에서 +8–11%)와 **학습 말뭉치를 법령·보도자료 혼합으로 바꾸는 것**(법령 +13%)과 **Llama-3식 사전분할 정규식**(+3–4%)에 있다. 형태소 사전분할(kiwi·MeCab)은 EXAONE·Polyglot-Ko가 쓰지만 KorQuAD형 추출 과제에는 일반 BPE가 더 낫다는 실험 결과가 있어 "실험"으로 둔다. | 도입(말뭉치·정규식), 실험(16k, 형태소) |
| 데이터 처리 | FineWeb-2의 `kor_Hang` 필터 임계값과 datatrove(Apache-2.0)의 한국어 단어 분리기(kiwipiepy 사용)가 공개돼 있어 그대로 옮길 수 있다. PII는 ko-pii(MIT)·Presidio KR RRN 인식기(MIT)가 체크섬 검증까지 한다. HWP 파서는 현재 쓰는 `hwp5txt`(pyhwp)가 **AGPL-3.0**이므로 rhwp(MIT)·python-hwpx(Apache-2.0)로 교체할 근거가 있다. | 도입 |
| 학습 레시피 | 현재 사전학습은 **스텝당 2,048 토큰**(batch 2×1,024), lr 3e-4, warmup 100스텝이다. 같은 크기(124M)의 llm.c 재현은 스텝당 524,288 토큰·lr 6e-4·warmup 700, SmolLM2-135M은 1,048,576 토큰·lr 3e-3·warmup 2,000이다. gradient accumulation으로 스텝당 토큰을 64k–128k로 올리고 lr를 재탐색하는 것이 가장 큰 변화다. MPS에서 bf16 autocast+torch.compile은 측정상 FP32 대비 **+79%** 처리량이었다(동시 부하 상태의 상대값). WSD 스케줄은 중간 체크포인트 비교와 디케이 구간에 고품질 데이터를 넣는 MiniCPM식 운영에 유리하다. | 도입 |
| 한국어 평가 | 91M·10억 토큰 규모에서는 KMMLU·HAE-RAE가 거의 무작위 수준이라 의미가 없다(참고: kanana 192M도 KMMLU 30.2). KoBEST(CC BY-SA 4.0)의 로그우도 선택형 5과제와 도메인별 held-out BPC(bits per character)가 사전학습 품질의 외부 점검으로 적합하다. HAE-RAE(CC BY-NC-ND, 비상업 연구 한정 동의), Ko-H5(비공개 테스트셋), LogicKor·KoMT-Bench(GPT-4 심판 필요)는 제외. | 도입(KoBEST·BPC), 제외(그 외) |
| RAG 검색 | BM25는 bm25s(MIT)+kiwipiepy(Apache-2.0)로 형태소 기반 색인을 바로 만들 수 있다. 임베딩은 bge-m3(MIT)·KURE-v1(MIT)·KoE5(MIT)·bge-reranker-v2-m3(Apache-2.0)가 라이선스 문제 없이 쓸 수 있지만, 이들은 **외부 사전학습 가중치**다. 생성 모델과 분리된 검색기에 정책을 적용할지는 결정이 필요하다. | 도입(BM25), 질문(임베딩) |
| 제약 디코딩 | outlines(Apache-2.0)·lm-format-enforcer(MIT)·xgrammar(Apache-2.0)는 정규식·JSON·CFG 제약용이며 "근거 문서의 부분 문자열만 허용"이라는 trid의 제약을 직접 표현하지 못한다. 자체 `trid/constrained.py`를 유지하되 xgrammar의 토큰 마스크 캐시 아이디어를 빌려 속도를 높이는 정도가 적절하다. | 보류(라이브러리), 도입(아이디어) |

마지막 절에 우선순위 5개 행동과 첫 단계를 적었다.

---

## 1. 토크나이저

### 1.1 현재 상태와 측정 방법

`trid/text.py`는 HF `tokenizers`의 `ByteLevel` 사전분할(GPT-2 정규식, `use_regex=True`)과 BPE 8,192 어휘를 쓰고, 위키 학습 split만으로 학습했다. PRETRAINING.md는 법령 1.68, 보도자료 1.60 글자/토큰을 보고했다.

이 조사에서는 로컬 파일럿 수집본(`data/law-ko-source` 77문서 1.51M자, `data/korea-kr-briefing-source` 137문서 0.43M자, `data/wiki-ko-full-source` 342문서 1.51M자)으로 공개 토크나이저의 글자/토큰을 같은 조건에서 측정했다. 비교용 tokenizer.json만 받았고(모델 가중치 없음), 측정 후 삭제했다.

| 토크나이저 | 어휘 | 법령 | 보도자료 | 위키 | 출처 |
|---|---|---|---|---|---|
| **trid 8k (현행)** | 8,192 | 1.673 | 1.603 | 1.886 | `runs/trid-korean-medium-base-v1/tokenizer.json` |
| kanana-1.5-2.1b (= Llama-3 tokenizer) | 128,256 | 1.542 | 1.570 | 1.617 | [HF](https://huggingface.co/kakaocorp/kanana-1.5-2.1b-base) · 논문에 Llama-3 토크나이저 채택 명시 ([arXiv 2502.18934](https://arxiv.org/abs/2502.18934)) |
| EXAONE-3.5-2.4B | 102,400 | 1.722 | 1.701 | 1.793 | [HF](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct) |
| HyperCLOVAX-SEED-Text-0.5B | 110,592 | 1.840 | 1.831 | 1.876 | [HF config](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B/raw/main/config.json) |
| Polyglot-Ko-1.3B | 30,003 | 1.803 | 1.825 | 1.830 | [HF](https://huggingface.co/EleutherAI/polyglot-ko-1.3b) |
| Qwen3-0.6B | 151,669 | 1.422 | 1.391 | 1.432 | [HF](https://huggingface.co/Qwen/Qwen3-0.6B) |
| SmolLM2-135M (영어 전용) | 49,152 | 0.488 | 0.552 | 0.535 | [HF](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) |

(측정) 결론: **어휘 8k짜리 한국어 전용 토크나이저가 10만 단위 다국어 토크나이저와 같은 압축률**이다. 한국어 모델이 자랑하는 "압축률 개선"은 다국어 어휘 안에서 한국어 비중을 늘린 효과이지, 한국어 전용 어휘를 쓰는 trid에는 해당하지 않는다. 그러므로 "큰 모델의 토크나이저를 흉내 내면 좋아진다"는 가정은 버려도 된다. 참고로 EXAONE 3.0 보고서의 토큰/단어 비교(EXAONE 2.46, Llama 3.1 3.01, Gemma 2 3.31, Qwen 2 3.29)도 같은 방향이다 ([arXiv 2408.03541 §2.2](https://arxiv.org/html/2408.03541v4)).

### 1.2 어휘 크기 (측정 + 스케일링 법칙)

위키 28.5M자로 byte-level BPE를 어휘 크기만 바꿔 학습했다(평가 문서는 학습에서 제외).

| 어휘 | 법령 | 보도자료 | 위키 |
|---|---|---|---|
| 8,192 | 1.669 | 1.602 | 1.875 |
| 16,384 | 1.806 (+8%) | 1.732 (+8%) | 2.083 (+11%) |
| 32,768 | 1.961 (+17%) | 1.875 (+17%) | 2.282 (+22%) |
| 65,536 | 2.094 | 2.028 | 2.466 |

Tao et al.의 어휘 스케일링 법칙([arXiv 2407.13623](https://arxiv.org/abs/2407.13623), NeurIPS 2024, CC BY 4.0)은 계산 예산 $C$에 대해 비어휘 파라미터 $N_{nv}=0.08\,C^{0.50}$, 어휘 파라미터 $N_v=0.20\,C^{0.42}$ ($N_v=V\cdot d$)를 적합했다. trid 91M을 10억 토큰으로 학습하면 $C\approx 6\times 91.2\text{M}\times 10^9 = 5.5\times10^{17}$이고 $N_v\approx 5.6\times10^6$, $d=768$이므로 $V_{opt}\approx 7{,}300$이다 `[INFERENCE: 영어 SlimPajama 기반 적합식을 한국어에 외삽]`. 논문의 Table 1도 3B 모델에 39K–43K를 권하므로 91M에 8k는 정상 범위다. 같은 식으로 $N_{nv}=0.08\,C^{0.5}\approx 59\text{M}$, 즉 10억 토큰 예산에서는 91M이 약간 과대 파라미터이며 데이터가 더 필요하다는 뜻이다(논문도 "LLM은 데이터에 굶주려 있다"고 강조).

판단: **16k는 실험, 32k 이상은 제외.** 16k는 글자/토큰 +8–11%(같은 문맥 1,024토큰에 더 긴 문서, 학습 토큰 수 8–11% 감소)와 임베딩 +6.3M 파라미터(tied)를 맞바꾼다. 10억 토큰 규모에서 손실이 좋아진다는 보장은 없으므로 20M 토큰 파일럿 2개(8k vs 16k)를 **BPC(bits per character)**로 비교해 결정한다. 토큰 손실은 어휘 크기에 따라 비교 불가하므로 글자 단위로 정규화해야 한다(Tao et al. §2.2, §A.5).

### 1.3 학습 말뭉치와 사전분할 정규식 (측정)

(a) 토크나이저 학습 자료를 위키 4,000문서 + 법령 563문서(평가 제외) 혼합으로 바꾸면:

| 어휘 | 법령 | 보도자료 | 위키 |
|---|---|---|---|
| 8,192 (위키만) | 1.669 | 1.602 | 1.875 |
| 8,192 (혼합) | **1.888 (+13%)** | 1.665 (+4%) | 1.852 (−1%) |
| 16,384 (혼합) | 2.058 | 1.815 | 2.054 |

(b) 현행 GPT-2 정규식은 숫자를 길이 제한 없이 묶어 BPE가 `20|26`, ` 0|2`, `12|3`처럼 자릿수를 불규칙하게 쪼갠다(측정: `2026년 3월 15일 1,500,000원 02-123-4567` → 26토큰). Llama-3 계열 정규식(`[^\r\n\p{L}\p{N}]?\p{L}+| ?\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`, 숫자는 최대 3자리)으로 바꾸면 혼합 8k 기준:

| 사전분할 | 법령 | 보도자료 | 위키 | 숫자 예시 |
|---|---|---|---|---|
| GPT-2 정규식(현행) | 1.888 | 1.665 | 1.852 | `20|26년`, ` 0|2|-|12|3` |
| Llama-3식, 숫자 ≤3자리 | **1.962** | **1.740** | **1.900** | `202|6`, ` 15`, `500`, `000` |
| Llama-3식, 숫자 1자리 | 1.927 | 1.672 | 1.849 | 모든 숫자 한 자리씩 |

Llama 3가 128K 어휘와 함께 이 정규식을 쓴다는 것은 [Llama 3 논문](https://arxiv.org/abs/2407.21783) §3.2에 있다. 한 자리 분할은 금액·전화번호 복사에서 토큰 경계를 가장 예측 가능하게 만들지만 압축률을 조금 잃는다. 어느 쪽이 RAG 값 복사 정확도에 좋은지는 SFT 후 자체 평가셋(값 유형 금액·전화·날짜)으로 비교해야 한다. HF `tokenizers`에서 `pre_tokenizers.Sequence([Split(Regex(pat), 'isolated'), ByteLevel(use_regex=False)])`로 구현하며 왕복 복원을 확인했다(측정).

### 1.4 형태소·자모 사전분할

| 항목 | 내용 | 라이선스 | trid 기대 효과 | 난이도 | 위험 | 판정 |
|---|---|---|---|---|---|---|
| [kiwipiepy](https://github.com/bab2min/kiwipiepy) / [Kiwi](https://github.com/bab2min/kiwi) | C++ 형태소 분석기. `kiwipiepy.sw_tokenizer.SwTokenizer`는 형태소를 고려하는 서브워드 토크나이저를 학습하고 HF tokenizer 호환 형식으로 저장한다 ([API 문서](https://bab2min.github.io/kiwipiepy/v0.17.1/kr/sw_tokenizer.html)). | Apache-2.0 (두 저장소 모두) | 조사·어미 경계가 깨끗해져 작은 어휘에서 의미 단위가 보존됨 | M (전처리 단계 추가, 토크나이저 형식 변경) | 분석기 버전에 토큰화가 묶임; 추론 시에도 Kiwi 필요(Raspberry Pi 목표와 충돌 가능); 추출형 QA에는 이득이 없다는 보고 | 실험 |
| [mecab-ko-dic](https://bitbucket.org/eunjeon/mecab-ko-dic) (+ [python 패키징](https://github.com/LuminosoInsight/mecab-ko-dic)) | EXAONE 3.0이 한국어 사전분할에 MeCab을 쓰고 그 위에 BBPE 102,400을 학습했다([§2.2](https://arxiv.org/html/2408.03541v4)). Polyglot-Ko도 MeCab 기반 "morpheme-aware byte-level BPE" 30,003 어휘다([arXiv 2306.02254](https://arxiv.org/html/2306.02254v2)). | 사전 Apache-2.0; MeCab 본체는 BSD/LGPL/GPL 삼중 | 위와 같음 | M | C 바이너리 의존, Kiwi보다 설치가 번거로움 | 보류(Kiwi로 대체) |
| 형태소→BPE 효과 근거 | Park et al. 2020 ([arXiv 2010.02534](https://arxiv.org/abs/2010.02534), AACL): 번역·KorNLI·KorSTS·NSMC·PAWS-X에서는 "형태소 분할 후 BPE"가 최선, **KorQuAD(추출형 QA)에서는 순수 BPE가 가장 좋았다.** | 논문 | trid 핵심 과제(근거에서 값 복사)는 KorQuAD형이므로 형태소 분할의 이득이 불확실 | — | — | 실험 설계 근거 |
| 자모(sub-character) | KR-BERT([arXiv 2008.03979](https://arxiv.org/abs/2008.03979))가 자모 수준부터 음절 수준까지 최소 단위를 조정해 작은 어휘·1/10 말뭉치로 비교 가능한 성능을 보고. | 논문 | 어휘를 더 줄일 수 있으나 byte-level BPE는 이미 미등록 글자를 UTF-8 바이트로 처리하므로 필요성 낮음 | L | 시퀀스 길이 증가, 제약 디코딩의 "글자 경계" 검사 재작성 | 제외 |

정리하면 토크나이저에서 **확실한 것은 (1) 혼합 말뭉치로 재학습, (2) Llama-3식 정규식과 숫자 분할**이고, **실험할 것은 (3) 16k 어휘, (4) Kiwi 형태소 사전분할**이다. 네 가지 모두 다음 실행 전에 20M 토큰 파일럿으로 BPC를 비교하면 하루 안에 결정할 수 있다.

---

## 2. 사전학습 데이터 처리

현재 `scripts/build_pretraining_mix.py`는 NFC 정규화 완전중복 제거 + 5단어 shingle MinHash(64해시·16밴드) + 정확 Jaccard 0.8 재검증, 200자 미만 제거, 전화번호·이메일 정규식 삭제를 한다. 아래는 그 위에 얹을 수 있는 것들이다.

| 항목 | 내용 | 라이선스 | trid 기대 효과 | 난이도 | 위험 | 판정 |
|---|---|---|---|---|---|---|
| [FineWeb-2 파이프라인](https://github.com/huggingface/fineweb-2) + [datatrove](https://github.com/huggingface/datatrove) | 언어별 필터 임계값이 공개돼 있다. [`configs/kor_Hang.yml`](https://raw.githubusercontent.com/huggingface/fineweb-2/main/configs/kor_Hang.yml): `language_score 0.842`, `dup_line_frac 0.374`, `top_n_grams (2,0.282)(3,0.228)(4,0.185)`, `dup_n_grams (5,0.201)…(10,0.127)`, `line_punct_thr 0.062`, `new_line_ratio 0.196`, `max_avg_word_length 6`, `max_non_alpha_words_ratio 0.791`, 불용어 21개(의·다·이·는·에·을·한·년·하·은·를·했·고·에서·가·있·으로·인·월·로·되). 한국어 단어 분리기는 [kiwipiepy를 쓴다](https://raw.githubusercontent.com/huggingface/datatrove/main/src/datatrove/utils/word_tokenizers.py). MinHash는 5-gram·14버킷×8해시. | datatrove Apache-2.0, fineweb-2 저장소 코드 Apache-2.0 | 법령·보도자료는 웹보다 깨끗하지만 표·목차·반복 머리글이 많다. Gopher 반복 필터(상위 n-gram 비율, 중복 줄 비율)를 한국어 임계값 그대로 적용하면 HWP 추출 잔여물(목차 반복, "그림입니다")을 수치로 걸러낼 수 있다 | S (임계값·불용어만 가져와 자체 코드에 구현) / M (datatrove 의존) | 웹용 임계값이 공문에서 과도하게 제거할 수 있음 → 제거 샘플을 반드시 눈으로 확인 | 도입(임계값 이식) |
| kiwipiepy를 MinHash·필터의 단어 분리기로 | 현재 5단어 shingle은 공백 어절 기준. 조사·어미가 붙은 어절은 같은 문장도 다른 shingle이 되어 재현율이 낮다 `[INFERENCE]`. datatrove와 같이 Kiwi 형태소를 단위로 쓰면 법령 개정판(동일 조문, 조사만 변경) 검출이 좋아진다 | Apache-2.0 | 유사중복 재현율 향상 | S | 속도(Kiwi는 C++라 충분히 빠름) | 도입 |
| [kss](https://github.com/hyunwoongko/kss) | 한국어 문장 분리기(규칙 기반, 선택적으로 형태소 분석기 사용) | BSD-3-Clause | 1,024 문맥 창에 문장 경계로 자르기, SFT 근거 문단 구성, 품질 필터의 "문장당 길이" 통계 | S | 공문의 "○", "□" 개조식 문장은 종결어미가 없어 분리 품질이 떨어짐 | 실험 |
| Kiwi `split_into_sents` | Kiwi 자체 문장 분리 API | Apache-2.0 | kss와 같은 용도, 의존성 하나로 통일 | S | 위와 같음 | 도입(kss 대신) |
| [soynlp](https://github.com/lovit/soynlp) | 비지도 단어 추출·토크나이저·정규화 | **LGPL-3.0** ([LICENSE](https://github.com/lovit/soynlp/blob/master/LICENSE)) | 반복 문자 정규화 등 일부 함수 | S | LGPL(동적 링크·미수정 사용은 가능하나 관리 부담), 2019년 이후 유지보수 적음 | 제외(Kiwi로 대체) |
| [HPLT 3.0](https://huggingface.co/datasets/HPLT/HPLT3.0) web-docs-scorer | 언어 독립 문서 품질 점수로 정렬, PII 주석 포함. 점수 코드는 [pablop16n/web-docs-scorer](https://github.com/pablop16n/web-docs-scorer) | 데이터 CC0(패키징), 스코어러 라이선스는 저장소에서 확인 필요(본 조사에서 미확인) | 웹 보충 데이터를 쓸 때만 의미 | M | 본 믹스는 공공 자료 중심이라 당장은 불필요 | 보류 |
| kanana의 한국어 edu 분류기 방식 | 공개 한국어 품질 분류기가 없어 FineWeb-Edu 방식으로 FastText 분류기를 반복 학습했고, 분류기 통과 데이터가 KMMLU +2.2, HAE-RAE +3.6 ([arXiv 2502.18934 §2.2 Table 2](https://arxiv.org/html/2502.18934)) | 방법(논문), FastText MIT | 보충용 웹 데이터 선별 | L (라벨 생성에 큰 모델 필요 → 정책상 외부 API도 안 쓰므로 자체 라벨링 필요) | 라벨 품질 | 보류 |
| [ko-pii](https://github.com/Marker-Inc-Korea/ko-pii) | 규칙·사전·형식 검증 기반 한국어 PII 탐지·가명화. 주민번호·전화·이메일·카드·이름·주소 등 33개 라벨, 코어 의존성 없음, JSONL 필드별 정책 | MIT (코드; 평가 데이터는 별도) | 현재 전화·이메일 정규식보다 넓은 범위(주민번호 체크섬, 계좌, 여권, 운전면허). 판례·재결례의 잔여 식별 정보 제거 | S | 이름·주소 탐지는 오탐이 많아 사전학습 데이터에서는 구조적 식별자(주민번호·전화·이메일·카드·계좌)만 켜는 것이 안전 | 도입 |
| [korean-pii](https://pypi.org/project/korean-pii/) | 표준 라이브러리만 쓰는 PII 엔진(주민번호·외국인번호 체크섬, 전화, 카드 Luhn, 이메일, 계좌 등) | PyPI 페이지에 라이선스 미표시 → 확인 전 사용 금지 | ko-pii와 중복 | S | 라이선스 불명 | 보류 |
| [Presidio](https://github.com/microsoft/presidio) KR RRN 인식기 | 2.2.360에 [`KrRrnRecognizer`](https://github.com/microsoft/presidio/blob/main/presidio-analyzer/presidio_analyzer/predefined_recognizers/country_specific/korea/kr_rrn_recognizer.py) 추가(2020-10 이전 발급분 체크섬 검증) | MIT | 주민번호 한 종류뿐이고 spaCy 의존성이 무거움 | M | 과도한 의존성 | 보류(ko-pii로 충분) |
| [rhwp](https://github.com/edwardkim/rhwp) | Rust+WASM HWP/HWPX 뷰어·편집기·텍스트 추출. 의존성 전부 permissive ([THIRD_PARTY_LICENSES](https://github.com/edwardkim/rhwp/blob/main/THIRD_PARTY_LICENSES.md)) | MIT | 보도자료 HWP 첨부 본문 추출 | S (CLI 교체) | 신규 프로젝트(2025–), 드문 HWP 레코드 미지원 가능 → 추출 길이·깨진 글자 비율을 pyhwp 결과와 비교 | 도입 |
| [python-hwpx](https://github.com/airmang/python-hwpx) | 순수 Python HWPX 읽기·편집·생성·검증 | Apache-2.0 ([LICENSE](https://github.com/airmang/python-hwpx/blob/main/LICENSE)) | 현재 `hwpx_text()`는 `Contents/section*.xml`을 직접 파싱한다. 표·각주·머리말 처리 규칙을 라이브러리에 맡길 수 있음 | S | 자체 파서가 이미 동작하므로 이득은 표·각주 보존 정도 | 실험 |
| [pyhwp](https://github.com/mete0r/pyhwp) (`hwp5txt`, 현재 사용) | HWP 5.0 파서 | **AGPL-3.0** | — | — | `collect_korea_briefing.py`가 subprocess로 호출한다. 별도 프로세스 호출은 일반적으로 AGPL 전염 범위 밖으로 보지만, 제품 배포 시 설명 부담이 있고 2023년 이후 유지보수 정체 | 교체 권고 |

`collect_korea_briefing.py`의 첨부 선호 순서는 HWPX > PDF(pdftotext) > HWP(hwp5txt)이므로 HWP 경로만 rhwp로 바꾸면 된다. 교체 전 같은 파일 100건에서 두 도구의 추출 길이·`\ufffd` 비율·표 셀 순서를 비교하는 것이 첫 단계다.

---

## 3. ~100M 모델 학습 레시피

### 3.1 공개 레시피 비교

| 레시피 | 규모 | 스텝당 토큰 | lr(최대) | warmup | 스케줄 | 기타 | 라이선스 | 출처 |
|---|---|---|---|---|---|---|---|---|
| **trid 현행** | 91M, 12층×768, 문맥 1,024 | **2,048** (batch 2) | 3e-4 | min(100, steps/10) | cosine → 최대의 10% | AdamW β=(0.9,0.95), wd 0.1, clip 1.0, FP32, tied embedding | — | `trid/pretraining.py`, `runs/trid-korean-medium-base-v1/training.json` |
| llm.c GPT-2 124M 재현 | 124M, 12층×768, 문맥 1,024 | 524,288 | 6e-4 | 700 스텝 | 선형 → 0 | wd 0.1(2D 가중치만), clip 1.0, 10B 토큰, bf16 | MIT | [Discussion #481](https://github.com/karpathy/llm.c/discussions/481) |
| nanoGPT `train_gpt2.py` | 124M | ~491,520 | 6e-4 | 2,000 | cosine → 6e-5 | wd 0.1, β2 0.95, clip 1.0 | MIT | [config](https://raw.githubusercontent.com/karpathy/nanoGPT/master/config/train_gpt2.py), [train.py](https://raw.githubusercontent.com/karpathy/nanoGPT/master/train.py) |
| SmolLM2-135M | 135M, **30층×576**, GQA(9Q/3KV), tied, 문맥 2,048 | 1,048,576 (dp 64 × micro 8 × 2,048) | **3e-3** | 2,000 | WSD: stable 1.6M 스텝 후 선형 디케이 400k 스텝(=20%) → 0 | wd 0.01, init std 0.0417 (=1/√576), bf16, grad clip 1.0, 2T 토큰 | 코드 Apache-2.0 | [nanotron config](https://raw.githubusercontent.com/huggingface/smollm/main/text/pretraining/smollm2/config_smollm2_135M.yaml), [논문](https://arxiv.org/abs/2502.02737) |
| MiniCPM | 1.2B/2.4B (풍동 실험은 9M–0.5B) | 2M→4M | μP 적용 시 0.01 (scale_emb 12, init_std 0.1, scale_depth 1.4) | — | **WSD**, 디케이는 지수형 $0.5^{(s-S)/T}$, "디케이 10% 스텝이면 충분" | 디케이 구간에 고품질 SFT 데이터를 섞는 것이 SFT만 하는 것보다 효과가 큼(§5) | 논문 | [arXiv 2404.06395](https://arxiv.org/html/2404.06395v2) |
| TinyLlama 1.1B | 1.1B | 2M | 4e-4 | 2,000 | cosine | 3T 토큰 | Apache-2.0 | [GitHub](https://github.com/jzhang38/TinyLlama) |
| Kanana (from scratch 8B/26.8B) | — | — | DeepSeek 하이퍼파라미터 스케일링 법칙 | — | multi-step | **z-loss 5e-6**, independent weight decay 1e-4 | 논문 | [arXiv 2502.18934 §A.3](https://arxiv.org/html/2502.18934) |
| PaLM | — | — | — | — | — | z-loss $10^{-4}\log^2 Z$ 로 softmax 정규화 상수를 0 근처로 | 논문 | [arXiv 2204.02311](https://arxiv.org/pdf/2204.02311) |
| MobileLLM 125M/350M | 서브-10억 | — | — | — | — | **deep-and-thin + embedding sharing + GQA**가 동일 파라미터에서 125M +2.7%p, 350M +4.3%p; 블록 가중치 공유로 추가 +0.7–0.8%p | 논문; 코드 저장소는 비표준 라이선스(NOASSERTION) → 코드는 쓰지 않음 | [arXiv 2402.14905](https://arxiv.org/abs/2402.14905) |
| μP ([microsoft/mup](https://github.com/microsoft/mup)) | — | — | 폭에 무관한 lr 전이 | — | — | 작은 모델에서 찾은 lr를 큰 모델로 전이 | MIT | [arXiv 2203.03466](https://arxiv.org/abs/2203.03466) |
| modded-nanogpt | 124M | — | Muon 옵티마이저 등 | — | — | GPT-2 124M 속도 경쟁 기록 | MIT | [GitHub](https://github.com/KellerJordan/modded-nanogpt) |

### 3.2 trid에 대한 함의

1. **스텝당 토큰이 256배 작다.** 2,048 토큰/스텝은 llm.c(524k)·SmolLM2(1M)의 1/256–1/512이다. 같은 10억 토큰을 48만 스텝으로 쪼개면 옵티마이저 노이즈가 크고 스텝당 Python·Metal 오버헤드가 누적된다. Mac 한 대에서 micro-batch를 키울 수는 없지만(메모리 5.5GB에서 batch 2), **gradient accumulation 32–64회로 스텝당 65k–131k 토큰**을 만드는 것은 처리량 손실 없이 가능하다 `[INFERENCE: 처리량은 micro-batch로 결정되고 accumulation은 optimizer.step 호출만 줄임]`. 구현은 `loss_for` 호출을 루프로 감싸고 `clip_grad_norm_`·`optimizer.step()`을 누적 끝에서만 호출하면 된다(S). 체크포인트 재개 상태에 누적 카운터를 포함해야 한다.
2. **lr는 배치와 함께 다시 찾아야 한다.** 비교 레시피는 0.5M 배치에 6e-4, 1M 배치에 3e-3이다. 128k 배치면 6e-4–2e-3 사이에서 20M 토큰 파일럿 3개(6e-4, 1e-3, 2e-3)를 돌려 BPC로 고른다 `[INFERENCE]`. μP를 적용하면 이 탐색을 더 작은 모델(예 20M)에서 해서 전이할 수 있지만, 모델 코드에 μP 파라미터화(출력 스케일, 폭 기준 init)를 넣어야 하므로 M 난이도다. 10억 토큰 한 번 실행이면 직접 3개 파일럿이 더 싸다.
3. **WSD 스케줄 도입.** 현행 cosine은 총 스텝을 미리 고정한다. WSD(warmup–stable–decay)는 stable 구간의 어느 체크포인트에서든 10–20% 길이의 디케이를 붙여 "완성된" 모델을 만들 수 있어(MiniCPM §4.2 "10% steps are enough", SmolLM2는 20% 선형), 데이터 수집이 길어져 토큰 수가 유동적인 trid 상황과 맞는다. `pretrain --resume`이 원래 스케줄을 고정하는 현행 설계와 호환하려면 `planned_steps` 대신 `stable_until`/`decay_steps`를 metadata에 넣는다(S). 디케이 구간에 KLUE-MRC·합성 RAG 데이터를 섞는 MiniCPM §5 방식은 SFT 2,000스텝이 기본 대화를 퇴행시키던 문제의 대안 실험이 된다(실험).
4. **z-loss는 선택.** FP32 학습에서는 불안정이 보고되지 않았다. bf16 autocast로 바꾸면 PaLM 1e-4 또는 Kanana 5e-6 계수의 z-loss를 넣어 두는 것이 안전하다(S).
5. **가중치 공유는 이미 한다.** MobileLLM과 SmolLM2-135M·HyperCLOVA X SEED 0.5B·kanana-1.5(`tie_word_embeddings: true`) 모두 tied이므로 현행 유지. deep-and-thin(30층×576)은 같은 파라미터에서 더 좋다는 MobileLLM 결과가 있으나 MPS에서 층 수 증가는 커널 호출 횟수 증가라 처리량이 떨어질 수 있다 → 보류.
6. **weight decay 적용 대상.** llm.c는 2D 가중치만 decay한다. 현행은 RMSNorm 가중치·임베딩에도 0.1을 건다(`optimizer = AdamW(model.parameters(), …)`). 파라미터 그룹을 나누는 것은 S 난이도의 정리 항목이다.

### 3.3 Mac MPS 특이사항 (측정)

프로젝트 venv의 PyTorch 2.14.1에서 확인했다.

- `torch.autocast(device_type='mps', dtype=torch.bfloat16)`과 `float16` 모두 동작했고 SDPA·Linear가 해당 dtype으로 실행됐다. (MPS bf16 autocast는 [pytorch#139386](https://github.com/pytorch/pytorch/issues/139386)에서 요청돼 이후 추가됨.)
- `torch.compile` 기본 백엔드가 MPS에서 동작했다. "Not enough SMs to use max_autotune_gemm mode" 경고는 무해하다.
- 91M 실제 모델, batch 2×1,024, 무작위 토큰 8스텝 평균(워밍업 3스텝 제외). **같은 시각에 SFT 학습 job과 수집기 2개가 돌고 있어 절대값은 낮고 상대 비교만 유효하다.**

| 모드 | tok/s | 피크 driver 메모리 |
|---|---|---|
| FP32 (현행) | 1,246 | 5.11 GB |
| bf16 autocast | 1,535 (+23%) | 5.09 GB |
| FP32 + torch.compile | 1,426 (+14%) | 5.09 GB |
| **bf16 autocast + torch.compile** | **2,230 (+79%)** | 5.09 GB |

위험: (1) bf16은 가수 7비트라 손실 수치가 FP32와 달라진다. autocast는 FP32 master weight를 유지하므로 보통 문제없지만, 250스텝 파일럿에서 FP32와 검증 손실 차이를 기록한 뒤 채택해야 한다. (2) `torch.compile`은 첫 스텝 컴파일 시간과 동적 shape(검증 batch, 생성 시 KV cache 길이 변화) 재컴파일이 있으므로 학습 forward에만 적용한다. (3) `F.scaled_dot_product_attention`은 이미 쓰고 있고 MPS에서 is_causal 경로가 지원되므로 별도 메모리 효율 attention 라이브러리는 필요 없다.

[MLX](https://github.com/ml-explore/mlx)(MIT)는 Apple Silicon 전용 프레임워크로 PyTorch MPS보다 빠를 가능성이 있으나 모델·학습 루프·safetensors 로더를 다시 써야 하고(L) 이 조사에서 속도를 측정하지 않았다 → 보류.

---

## 4. 한국어 평가

### 4.1 벤치마크 목록과 라이선스

| 벤치마크 | 형식 | 라이선스 | 91M에서 의미 | 판정 |
|---|---|---|---|---|
| [KoBEST](https://huggingface.co/datasets/skt/kobest_v1) (BoolQ·COPA·WiC·HellaSwag·SentiNeg) | 선택형, 로그우도 평가 가능 | CC BY-SA 4.0 | 2–4지선다라 무작위가 25–50%. HyperCLOVA X SEED 0.5B Base(수백억 토큰 이상)가 5-shot 0.696([모델 카드](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B)). 91M·10억 토큰이면 SentiNeg·BoolQ에서 무작위보다 조금 위, 나머지는 무작위 근처일 것 `[INFERENCE]`. 그래도 체크포인트 간 **추세**는 볼 수 있음 | 도입(보조 지표) |
| [KMMLU](https://huggingface.co/datasets/HAERAE-HUB/KMMLU) | 4지선다 35,030문항, 45과목 | CC BY-ND 4.0 (평가용 사용은 가능, 변형 재배포 불가) | kanana 192M(8B에서 증류)도 30.16 ([Table 6](https://arxiv.org/html/2502.18934)) → 무작위(25%) 근처. 91M은 측정 의미 없음 | 제외 |
| [HAE-RAE Bench 1.1](https://huggingface.co/datasets/HAERAE-HUB/HAE_RAE_BENCH_1.1) | 선택형 4,900문항 | **CC BY-NC-ND 4.0**, 게이트: "non-commercial research ONLY" 동의 필요 | 상업 제품 내부 평가라도 약관 위반 소지 | 제외 |
| [KLUE](https://huggingface.co/datasets/klue/klue) | 8과제(MRC 포함) | CC BY-SA 4.0 | 이미 MRC dev 150문항을 자체 평가셋에, train 6,000행을 SFT에 사용. YNAT·NLI는 로그우도 선택형으로 돌릴 수 있음 | 유지 |
| [KorQuAD 1.0](https://korquad.github.io/category/1.0_KOR.html) | 추출형 QA | CC BY-ND 2.0 KR | 평가 전용으로는 가능(변형 금지). 위키 기반이라 KLUE-MRC와 중복 | 보류 |
| Ko-H5 ([Open Ko-LLM Leaderboard](https://aclanthology.org/2024.acl-long.177.pdf)) | Ko-ARC·Ko-HellaSwag·Ko-MMLU·Ko-TruthfulQA·Ko-CommonGen v2 | 논문 표기: CC-BY-SA / MIT / CC-BY-SA / Apache-2.0 / Apache-2.0 | 테스트셋은 리더보드 비공개 운영; 공개 복사본은 출처 검증 필요 | 제외 |
| [LogicKor](https://github.com/instructkr/LogicKor) | 42 멀티턴, LLM-as-judge | 데이터 CC BY-SA 4.0 ([HF](https://huggingface.co/datasets/maywell/LogicKor)); 코드 저장소 라이선스 미표시; 2024-10 읽기 전용 전환 | GPT-4 심판 필요(정책상 외부 API 불가), 91M은 0점대 | 제외 |
| [KoMT-Bench](https://huggingface.co/datasets/LGAI-EXAONE/KoMT-Bench) | 80문항 멀티턴, GPT-4 심판 | LGPL-3.0 (데이터에 LGPL이라 해석 모호) | 위와 같음 | 제외 |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | 평가 프레임워크; `kobest`, `kmmlu`, `haerae`, `csatqa`, `kormedmcqa` 과제 내장 | MIT | trid는 HF 모델이 아니므로 `LM` 서브클래스(loglikelihood·generate_until)를 구현해야 함(M). KoBEST만 쓸 거면 자체 로그우도 스코어러가 더 작음(S) | 실험 |

### 4.2 사전학습 품질의 외부 점검으로 권하는 조합

1. **도메인별 held-out BPC.** `build_pretraining_mix.py`가 이미 group 해시로 train/valid/test를 나눈다. 토큰 손실 대신 $\text{BPC} = \text{loss}_{\text{token}}\times \frac{\#\text{tokens}}{\#\text{chars}\cdot\ln 2}$로 법령·보도자료·위키 각각을 기록하면 토크나이저가 바뀌어도 실행 간 비교가 된다(Tao et al. §A.5가 같은 이유로 글자 단위 지표를 씀). 난이도 S.
2. **KoBEST 5과제 로그우도.** 생성 없이 각 선택지의 조건부 로그우도만 계산하므로 250스텝마다 돌려도 싸다(각 과제 test 수백–수천 문항). 무작위 기준선과 함께 표기한다. 난이도 S.
3. **자체 310문항 RAG 평가셋**은 그대로 최종 지표. 위 둘은 사전학습 단계의 조기 경보다.

---

## 5. RAG·검색

### 5.1 키워드 검색

| 항목 | 내용 | 라이선스 | 기대 효과 | 난이도 | 위험 | 판정 |
|---|---|---|---|---|---|---|
| [bm25s](https://github.com/xhluca/bm25s) | numpy/scipy 기반 BM25, 사용자 정의 토크나이저 함수 지원 | MIT | 기관 문서 수천 건 규모에서 충분히 빠름 | S | — | 도입 |
| [rank_bm25](https://github.com/dorianbrown/rank_bm25) | 순수 Python BM25 | Apache-2.0 | bm25s보다 느림 | S | 큰 문서집합에서 느림 | 보류 |
| kiwipiepy 형태소를 BM25 토큰으로 | `Kiwi.tokenize()`의 형태소(체언·용언 어간)만 색인하면 "출장비는/출장비를/출장비가"가 한 토큰이 됨 | Apache-2.0 | 조문 번호·기관명 같은 명시적 표현 검색의 기준선 품질 | S | 신조어·기관 약어 분석 오류 → 원형 어절을 함께 색인 | 도입 |
| Lucene [Nori](https://lucene.apache.org/) 분석기 (Elasticsearch/OpenSearch) | 서버형 검색 엔진의 한국어 분석기 | Apache-2.0 | 폐쇄망 제품에서 검색 서버를 둘 때 | M | 서버 운영 | 보류(제품 단계) |

### 5.2 임베딩·재정렬 모델 — 정책 질문

| 모델 | 기반 | 라이선스 | 한국어 검색 성능(KURE 벤치, Recall@10 평균) | 비고 |
|---|---|---|---|---|
| [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | XLM-R 568M, 문맥 8,192, dense+sparse+colbert | MIT | 0.792 | 다국어 기본 |
| [nlpai-lab/KURE-v1](https://huggingface.co/nlpai-lab/KURE-v1) | bge-m3를 한국어 2M 쌍으로 미세조정 | MIT | 0.797 | 공공·법률·의료 포함 8개 한국어 검색 벤치에서 1위([모델 카드](https://huggingface.co/nlpai-lab/KURE-v1)) |
| [nlpai-lab/KoE5](https://huggingface.co/nlpai-lab/KoE5) | multilingual-e5-large 미세조정, 문맥 512 | MIT | 0.753 | 짧은 문맥 |
| [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | 크로스인코더 568M | Apache-2.0 | — | 하이브리드 검색 후 재정렬 |

이 네 가지는 라이선스상 상업 이용이 가능하고 생성 모델 가중치와 완전히 분리된다. 그러나 **README는 "기존 모델의 가중치를 사용하지 않는다", "학습·추론 중 가중치 다운로드가 없다"고 선언**한다. 검색기가 외부 가중치를 쓰면 (a) 그 문장을 "생성 모델에 한해"로 좁혀 다시 써야 하고, (b) 폐쇄망 제품에 568M 모델(FP32 2.3GB) 하나가 추가된다. 선택지:

1. 외부 임베딩 허용(정책 문구 수정) → KURE-v1 + bge-reranker-v2-m3 하이브리드. 난이도 M.
2. 외부 가중치 불허 → BM25(kiwi) 단독으로 시작하고, 필요 시 trid 사전학습 base에서 자체 bi-encoder를 contrastive 학습(KLUE-MRC 질문–문단 쌍, 합성 RAG 쌍 활용). 난이도 L, 품질 불확실.
3. 중간: 외부 임베딩을 **개발 중 상한선 측정용**(검색 실패 vs 생성 실패 분리)으로만 쓰고 제품에는 넣지 않음. 난이도 S.

**이 결정은 소유자 판단 사항으로 남긴다.** 어느 쪽이든 첫 단계는 같다: 자체 310문항 평가셋의 근거 문서를 corpus로 삼아 BM25(kiwi) vs KURE-v1의 Recall@3을 측정해 검색이 병목인지 확인하는 것이다. 공개 한국어 검색 벤치 중 공공·법률 PDF 기반인 [AutoRAGRetrieval](https://huggingface.co/datasets/yjoonjang/markers_bm)과 위키 기반 [MIRACL ko](https://huggingface.co/datasets/miracl/miracl)는 보조로 쓸 수 있다(각 데이터셋 라이선스는 사용 전 카드에서 확인).

---

## 6. 제약·구조화 디코딩

trid의 `trid/constrained.py`는 (1) `확인 불가`/`근거 충돌`/`값 [Dn]`만 허용, (2) 값은 인용한 근거 문서 안에 글자 경계로 존재하는 부분 문자열, (3) 출처 번호는 프롬프트에 있는 것만, (4) 남은 토큰 수 안에 완성 가능한 접두어만 허용(바이트 수 상한)을 토큰마다 검사한다.

| 라이브러리 | 방식 | 라이선스 | trid 제약 표현 가능? | 판정 |
|---|---|---|---|---|
| [outlines](https://github.com/dottxt-ai/outlines) | 정규식·JSON 스키마·CFG → 토큰 수준 FSM 컴파일, HF/vLLM 토크나이저 | Apache-2.0 | 근거가 프롬프트마다 바뀌므로 매 요청마다 "모든 부분 문자열의 합집합" 정규식을 컴파일해야 하고 크기가 문서 길이의 제곱으로 커짐 `[INFERENCE]`. 토큰 수 상한(4)은 FSM으로 표현 불가 | 보류 |
| [lm-format-enforcer](https://github.com/noamgat/lm-format-enforcer) | 문자 수준 파서 + 토크나이저 트라이, JSON/정규식 | MIT | 위와 같음; 다만 "문자 수준 파서 + 토큰 트라이" 구조는 trid 구현과 같은 발상 | 보류 |
| [xgrammar](https://github.com/mlc-ai/xgrammar) | CFG, 어휘를 문맥 독립/의존 토큰으로 나눠 마스크를 사전 계산(adaptive token mask cache) | Apache-2.0 | CFG로 "문서 부분 문자열"을 쓰려면 문서마다 문법 생성 필요. 반면 **토큰 마스크 캐시 아이디어**는 trid에 바로 적용 가능 | 보류(라이브러리) / 도입(아이디어) |
| [guidance](https://github.com/guidance-ai/guidance) | 프로그램형 제약, HF 모델 전제 | MIT | trid 모델 래핑 필요 | 제외 |

권고: 자체 구현을 유지한다. 속도가 문제라면 xgrammar식으로 **어휘 8,192개를 바이트열 트라이로 미리 만들고, 근거 문서의 suffix automaton(또는 글자 경계 접두어 집합)과 교차해 허용 토큰 집합을 접두어 상태별로 캐시**하면 토큰마다 8,192개를 검사하는 현재 방식(`[INFERENCE: constrained.py 설명 기준]`)보다 빠르다. 난이도 S–M. 어휘가 16k로 커져도 비용이 선형이라 문제없다.

---

## 7. 우선순위 행동 5개

1. **토크나이저 v2를 다음 10억 토큰 실행 전에 확정한다.** (1절, 난이도 S, 하루)
   첫 단계: `trid/text.py`의 `train_tokenizer`에 사전분할 옵션(Llama-3식 정규식, 숫자 1자리/3자리)을 추가하고, 토크나이저 학습 입력을 `data/korean-foundation-v3` 학습 split(법령·보도자료·위키 혼합)으로 바꾼다. 8k/16k × 숫자 1/3자리 네 조합을 만들어 valid split의 글자/토큰을 기록하고, 두 후보(8k·16k)로 20M 토큰 파일럿을 돌려 BPC로 결정한다. 측정 근거는 1.2–1.3절 표.
2. **학습 루프: gradient accumulation + WSD + bf16 autocast + torch.compile.** (3절, 난이도 M, 2–3일)
   첫 단계: `trid/pretraining.py`에 `accumulate_steps`(기본 1)와 `schedule='cosine'|'wsd'`를 추가하되 기존 체크포인트 재개 검증(`planned_steps` 고정)과 호환되게 metadata에 저장한다. 250스텝 파일럿으로 FP32 vs bf16의 검증 손실 차이와 처리량을 `reports/`에 기록한 뒤 채택한다. lr는 128k 토큰/스텝 기준 6e-4·1e-3·2e-3 파일럿 3개로 고른다.
3. **데이터 파이프라인: FineWeb-2 한국어 임계값 이식 + kiwi 형태소 shingle + ko-pii + rhwp.** (2절, 난이도 S–M, 2일)
   첫 단계: `build_pretraining_mix.py`에 Gopher 반복 필터(top n-gram, dup line frac)와 평균 단어 길이·불용어 필터를 `kor_Hang.yml` 값으로 넣고 제거 샘플 200건을 수동 검토한다. `collect_korea_briefing.py`의 `hwp5txt`를 rhwp로 바꾸기 전에 동일 HWP 100건의 추출 길이·깨진 글자 비율을 비교한다. ko-pii는 주민번호·전화·이메일·카드·계좌 라벨만 켜서 `provenance.json`의 변경 기록에 추가한다.
4. **평가: 도메인별 BPC와 KoBEST 로그우도를 체크포인트마다 기록한다.** (4절, 난이도 S, 1일)
   첫 단계: `trid/pretraining.py`의 `validate`에 글자 수를 함께 세어 BPC를 `metrics.json`에 추가하고, `scripts/evaluate_kobest.py`(CC BY-SA 4.0 데이터, 선택지 로그우도 비교)를 만들어 무작위 기준선과 함께 출력한다. KMMLU·HAE-RAE·LLM-judge 벤치는 쓰지 않는다.
5. **검색 병목 진단 후 임베딩 정책을 결정한다.** (5절, 난이도 M, 2일)
   첫 단계: bm25s + kiwipiepy로 310문항 평가셋의 근거 문서를 색인해 Recall@1/3을 측정한다. 같은 corpus에 KURE-v1(MIT)을 **개발 측정용으로만** 적용해 상한선을 본다. 두 수치를 근거로 "검색기에 외부 가중치를 허용할지"를 소유자가 결정하고, 결정에 따라 README의 "외부 가중치 미사용" 문구 범위를 명시한다.

---

## 부록 A. 측정 조건

- Mac(프로젝트 venv, PyTorch 2.14.1, MPS). 측정 시각에 `train_medium_sft_v2.py`와 수집기 2개가 동시에 실행 중이었다. 처리량 절대값은 이 때문에 README의 2,491–2,637 tok/s보다 낮다.
- 토크나이저 압축률: HF `tokenizers` `encode_batch(add_special_tokens=False)`, 글자 수는 Python `len(str)`(코드포인트). 어휘 크기 실험은 `BpeTrainer(min_frequency=2, initial_alphabet=ByteLevel.alphabet())`, 위키 8,315문서 28.5M자(평가 342문서 제외).
- 외부 토크나이저 파일은 각 HF 저장소의 `tokenizer.json`만 사용했고 측정 후 삭제했다. EXAONE·HyperCLOVA X SEED의 모델 라이선스는 비상업·별도 약관이므로 어떤 형태로도 trid에 포함하지 않는다.

## 부록 B. 1차 출처 모음

- Kiwi/kiwipiepy Apache-2.0: https://github.com/bab2min/kiwi , https://github.com/bab2min/kiwipiepy
- mecab-ko-dic Apache-2.0: https://github.com/LuminosoInsight/mecab-ko-dic
- Park et al. 2020 한국어 토큰화 전략: https://arxiv.org/abs/2010.02534
- Tao et al. 2024 어휘 스케일링: https://arxiv.org/abs/2407.13623
- EXAONE 3.0 토크나이저: https://arxiv.org/html/2408.03541v4 ; Polyglot-Ko: https://arxiv.org/html/2306.02254v2 ; Kanana: https://arxiv.org/html/2502.18934 ; Llama 3: https://arxiv.org/abs/2407.21783 ; KR-BERT: https://arxiv.org/abs/2008.03979
- FineWeb-2 파이프라인·한국어 설정: https://github.com/huggingface/fineweb-2 , https://raw.githubusercontent.com/huggingface/fineweb-2/main/configs/kor_Hang.yml ; datatrove: https://github.com/huggingface/datatrove
- kss BSD-3: https://github.com/hyunwoongko/kss ; soynlp LGPL-3.0: https://github.com/lovit/soynlp/blob/master/LICENSE
- ko-pii MIT: https://github.com/Marker-Inc-Korea/ko-pii ; Presidio MIT + KR RRN: https://github.com/microsoft/presidio
- pyhwp AGPL-3.0: https://github.com/mete0r/pyhwp ; rhwp MIT: https://github.com/edwardkim/rhwp ; python-hwpx Apache-2.0: https://github.com/airmang/python-hwpx
- llm.c 124M 레시피: https://github.com/karpathy/llm.c/discussions/481 ; nanoGPT: https://github.com/karpathy/nanoGPT ; SmolLM2 설정: https://raw.githubusercontent.com/huggingface/smollm/main/text/pretraining/smollm2/config_smollm2_135M.yaml ; MiniCPM: https://arxiv.org/html/2404.06395v2 ; TinyLlama: https://github.com/jzhang38/TinyLlama ; MobileLLM: https://arxiv.org/abs/2402.14905 ; μP: https://github.com/microsoft/mup ; PaLM z-loss: https://arxiv.org/pdf/2204.02311 ; modded-nanogpt: https://github.com/KellerJordan/modded-nanogpt ; MLX: https://github.com/ml-explore/mlx
- PyTorch MPS bf16 autocast 이슈: https://github.com/pytorch/pytorch/issues/139386
- KoBEST: https://huggingface.co/datasets/skt/kobest_v1 ; KMMLU: https://huggingface.co/datasets/HAERAE-HUB/KMMLU ; HAE-RAE 1.1: https://huggingface.co/datasets/HAERAE-HUB/HAE_RAE_BENCH_1.1 ; KLUE: https://huggingface.co/datasets/klue/klue ; KorQuAD: https://korquad.github.io/category/1.0_KOR.html ; Ko-H5 논문: https://aclanthology.org/2024.acl-long.177.pdf ; LogicKor: https://github.com/instructkr/LogicKor , https://huggingface.co/datasets/maywell/LogicKor ; KoMT-Bench: https://huggingface.co/datasets/LGAI-EXAONE/KoMT-Bench ; lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness
- bm25s MIT: https://github.com/xhluca/bm25s ; rank_bm25 Apache-2.0: https://github.com/dorianbrown/rank_bm25 ; bge-m3 MIT: https://huggingface.co/BAAI/bge-m3 ; KURE-v1 MIT: https://huggingface.co/nlpai-lab/KURE-v1 ; KoE5 MIT: https://huggingface.co/nlpai-lab/KoE5 ; bge-reranker-v2-m3 Apache-2.0: https://huggingface.co/BAAI/bge-reranker-v2-m3
- outlines Apache-2.0: https://github.com/dottxt-ai/outlines ; lm-format-enforcer MIT: https://github.com/noamgat/lm-format-enforcer ; xgrammar Apache-2.0: https://github.com/mlc-ai/xgrammar ; guidance MIT: https://github.com/guidance-ai/guidance
