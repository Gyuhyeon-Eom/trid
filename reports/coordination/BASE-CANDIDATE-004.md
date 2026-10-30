# BASE-CANDIDATE-004 → A: 소형 기반 모델 후보 3개, 추천 1개 (B, 라이선스·출처 검증만)

- **목표:** 기본 한국어 대화와 연속 지시를 통과할 2~8B instruct/chat 기반을 고른다.
  - 예전에 실패한 Qwen3-1.7B, Qwen2.5-7B, Gemma3-4B 설정은 다시 보지 않았다.
- **B가 하지 않은 것:** 가중치 다운로드 0, 학습 0, 모델 load 0. 수급과 추론은 **A만** 한다. 중복 다운로드는 하지 않는다.
- **조회 범위**
  - 공식 제작자의 Hugging Face 저장소(API·raw 파일, 특정 revision 고정)와 Google 공식 라이선스 페이지를 봤다.
  - 조회 시각은 2026-10-31 02:5x~03:1x KST다. 웹 요청은 모두 `scripts/logged_get.py`로 보냈다.
  - 조직별 최신 모델 목록(google, Qwen, kakaocorp, K-intelligence, naver-hyperclovax, LGAI-EXAONE, trillionlabs)을 조회한 뒤 2~8B instruct만 남겼다.
- **두 가지를 구분한다**
  - "최신"은 저장소 `createdAt`·`lastModified` 값일 뿐이다. 성능 순위가 아니다.
  - 벤치마크는 제작자가 직접 보고한 값이다. 어느 후보가 우수하다고 단정하지 않는다.
- **llama.cpp 호환은 두 가지로 나눈다**
  - (a) `8ca4060` 소스에 해당 구조가 있다는 근거: B가 확인했다.
  - (b) 실제 load·생성 검증: **확인하지 않았다(A 담당).**

## 후보 3개

| 항목 | **① google/gemma-4-E4B-it (추천)** | ② Qwen/Qwen3.5-4B | ③ kakaocorp/kanana-2-3b-instruct |
|---|---|---|---|
| revision | `ee0ef6023621cff504d758262d4e04895a5af4a2` (lastModified 2026-07-20, createdAt 2026-03-02) | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (lastModified 2026-03-02) | `6a5d7889964c4c590299d16e309eabab1f73f8a9` (lastModified 2026-07-29, createdAt 2026-07-24) |
| 크기 | 4.5B effective, 임베딩 포함 8B. 비전 ~150M, 오디오 ~300M 인코더 포함(카드 표) | safetensors 2개, 합계 9.32GB (bf16). 비전 포함(`Qwen3_5ForConditionalGeneration`) | safetensors 2개, 합계 7.02GB. `Qwen3ForCausalLM`, text only |
| 라이선스 | **Apache 2.0.** 카드 `license: apache-2.0`, `license_link` https://ai.google.dev/gemma/docs/gemma_4_license. 이 페이지 본문은 Apache License 2.0 전문이다(Last updated 2026-04-01 UTC, "documentation is deprecated … gemma.ai" 안내 있음) | **Apache 2.0.** 저장소 `LICENSE`, 카드 `license: apache-2.0` | **kanana-open-license** (저장소 `LICENSE`, Release Date July 17, 2025) |
| 상업 폐쇄망·재배포 | Apache 2.0 조건(라이선스 사본, NOTICE 유지, 수정 표시). Gemma 3의 Gemma Terms와는 다른 문서다 | Apache 2.0 조건 | **4.1(ii): 제3자에게 SI·온프레미스 배포 솔루션으로 제공·판매하려면 Kakao의 별도 상업 라이선스가 필요하다.** 재배포 시 약관 사본, Notice 파일, **"Powered by Kanana" 표시**, 산출물로 학습한 모델 이름에 "Kanana" 접두가 필요하다. 준거법은 대한민국 |
| template·thinking | 저장소 `chat_template.jinja` (18,569B, sha256 `0a2c8073…c5b5`). `enable_thinking`은 기본값이 **false**다. true일 때만 system에 `<\|think\|>`가 붙는다(템플릿 186~194행). 카드: "To disable thinking, remove the token" | 저장소 `chat_template.jinja`(7,756B). 카드: "**Qwen3.5 models operate in thinking mode by default**". 끄려면 `enable_thinking: False`를 넘겨야 한다 | 저장소 `chat_template.jinja`(10,725B). 카드에 thinking 모드 설명은 없다 |
| GGUF 출처 | **ggml-org/gemma-4-E4B-it-GGUF** rev `a69f018d7678a2b9f5e705f8903d6fb29b9b25dc` (2026-10-19). `.src_sha`의 PRIMARY가 `ee0ef602…`로 ①과 같다. 공식 Google QAT GGUF도 있다(아래) | 공식 Qwen·ggml-org의 4B GGUF는 **없다**. ggml-org에는 0.8B와 35B-A3B만 있다. 제3자: unsloth/Qwen3.5-4B-GGUF rev `e87f1764…`, 원본 revision 고정 정보 없음 | 공식 GGUF **없다**. 제3자: mradermacher/kanana-2-3b-instruct-GGUF rev `45a8a3e5…` |
| GGUF 파일·크기·sha256 | `gemma-4-E4B-it-Q8_0.gguf` 8,031,242,688B `34be82b17b4942d389b9b527170c4b058027abdd32531fda063d3d97dd8ce80a` | (제3자) `Qwen3.5-4B-Q8_0.gguf` 4,482,403,488B `10cc391b…61e1` | (제3자) `kanana-2-3b-instruct.Q8_0.gguf` 3,734,222,720B `259fcfee…38fb` |
| llama.cpp 8ca4060 소스 근거 | `src/llama-arch.cpp` `LLM_ARCH_GEMMA4 "gemma4"`, `src/models/gemma4.cpp`. 변환기 `conversion/gemma.py`가 `Gemma4ForConditionalGeneration`을 등록. mtmd `PROJECTOR_TYPE_GEMMA4V` | `LLM_ARCH_QWEN35 "qwen35"`, `src/models/qwen35.cpp`. `conversion/qwen3vl.py`가 `Qwen3_5ForConditionalGeneration`을 등록 | `LLM_ARCH_QWEN3 "qwen3"`, `src/models/qwen3.cpp`. `conversion/__init__.py`가 `Qwen3ForCausalLM`을 매핑 |
| 한국어 근거 | **약함.** 카드: "Out-of-the-box support for 35+ languages, pre-trained on 140+ languages". 한국어를 명시하지 않는다. 다국어 지표 MMMLU E4B 76.6%(제작자 보고, 한국어 개별 값 없음) | **약함.** 카드: "201 languages and dialects". 한국어 명시 없음 | **강함(제작자 보고).** Korean Tasks 표(KMMLU, KoSimpleQA, HAE-RAE). "Kanana-2 tokenizer: Improves Korean tokenization efficiency by over 30%" |
| context | 128K (카드) | 262,144 (config `max_position_embeddings`) | 32,768 (카드·config) |

## 추천 1개: ① google/gemma-4-E4B-it

- **이유 (확인된 사실만)**
  - **라이선스가 명확하다.** Apache 2.0이고 공식 Google 페이지가 근거다. 폐쇄망 상업 배포나 재배포에 별도 상업 라이선스가 필요 없다(③과 다른 점).
  - **GGUF 출처를 검증할 수 있다.** llama.cpp 관리 조직(ggml-org)이 변환했다.
    - `.src_sha`로 원본 revision이 고정돼 있다.
    - `convert.log`에 `convert_hf_to_gguf.py --outtype bf16`으로 변환한 기록과 `Gemma4ForConditionalGeneration` 구조가 남아 있다.
    - Google도 QAT q4_0 GGUF를 직접 배포한다.
  - **thinking 기본값이 꺼져 있다.** 템플릿 기본값이 false여서, 아무것도 넘기지 않아도 비사고 모드다.
  - **4B급이다.** 4.5B effective다.
- **약점과 미확인**
  - **한국어 근거가 약하다.** 카드에 한국어 명시가 없다. 한국어 대화 통과 여부는 A의 실행 검증으로만 판단한다.
  - 메모리: E4B는 임베딩 포함 8B다. Q8_0 파일 8.03GB(텍스트 모델)이고, 이미지가 필요하면 mmproj 0.99GB가 따로 있다. 자원 판단은 A가 한다.
  - **8ca4060 load·생성 미검증.** ggml-org GGUF(10-19 수정)는 `8ca4060`(10-28)보다 먼저 만들어졌다. 소스에는 해당 구조가 있지만 실제 load는 A가 검증해야 한다.
  - `convert.log`에 `Duplicated key name 'gemma4.context_length' … overwriting` 같은 경고 9줄이 있다. 기록만 하고 의미는 해석하지 않았다.
  - Gemma 4에는 E2B·E4B의 assistant, mtp 변형과 QAT 변형이 있다. 위 hash는 기본 `-it` 원본을 변환한 파일이다.
- **A 인계 정보 (A가 단독으로 수급)**

  | 항목 | 값 |
  |---|---|
  | 원본 model id | `google/gemma-4-E4B-it` @ `ee0ef6023621cff504d758262d4e04895a5af4a2` (gated: false) |
  | 원본 가중치 (참고) | `model.safetensors` 15,992,595,884B `cfbd3d2f1cd71bd471c37fe2bf8546d5028d41e5736f64e1ca6c6b8893125503`. `tokenizer.json` 32,169,626B `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f` |
  | 권장 GGUF | `ggml-org/gemma-4-E4B-it-GGUF` @ `a69f018d7678a2b9f5e705f8903d6fb29b9b25dc` / `gemma-4-E4B-it-Q8_0.gguf` / 8,031,242,688B / sha256 `34be82b17b4942d389b9b527170c4b058027abdd32531fda063d3d97dd8ce80a` |
  | 같은 저장소의 다른 파일 | `gemma-4-E4B-it-Q4_0.gguf` 4,590,807,392B `a555b900214b477d8880e7832e0b8925e139b0159640036b09fe472b6f2097f2`. `gemma-4-E4B-it-BF16.gguf` 15,053,097,408B `f0669d20ec5bd5a1a0b38707e25e441f8bc1f52d8a0ef2f0168f67fd4664f5de` |
  | 공식 Google GGUF (대안, QAT) | `google/gemma-4-E4B-it-qat-q4_0-gguf` @ `4b4a2c1d584be7264f87aac328a1bc739ce81b6c` / `gemma-4-E4B_q4_0-it.gguf` 5,154,941,280B `676c35070db6dbe52f93e9c864ee0fba4eddea94b9c875d9cb10daff453fbaee`. QAT 가중치라 `-it` 원본과는 다른 체크포인트다 |
  | template | GGUF에 내장된 template은 변환 때 원본 저장소에서 복사된 것이다(`convert.log` "Setting chat_template"). 원본 `chat_template.jinja` sha256은 `0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5`. 내장 template과 이 파일이 같은지는 A가 load 뒤 확인한다 |
  | thinking | 기본 off. `enable_thinking`을 넘기지 않거나 false로 넘긴다. system 메시지를 쓰면 template이 system 블록을 연다(190행). `<\|think\|>`는 true일 때만 붙는다 |
  | 라이선스 동봉 | 배포한다면 Apache 2.0 사본과 원본 저장소의 고지(있다면)를 함께 넣는다. 저장소에 별도 NOTICE 파일은 없다(파일 목록 기준) |

## 고르지 않은 이유

- **② Qwen3.5-4B**
  - 라이선스는 Apache 2.0으로 명확하다.
  - 하지만 공식 GGUF나 ggml-org 4B GGUF가 없다. 제3자 GGUF는 원본 revision 고정 정보가 없다.
  - 기본이 thinking 모드라 꺼야 한다.
  - 한국어 명시가 없다.
- **③ kanana-2-3b-instruct**
  - 한국어 근거는 가장 강하다.
  - 하지만 **온프레미스·SI 방식의 제3자 제공에 별도 상업 라이선스가 필요하다**(4.1(ii)). TRID의 폐쇄망 공공기관 배포 경로와 정면으로 겹친다.
  - 공식 GGUF가 없다.
  - Kakao가 상업 라이선스를 준다면 다시 검토할 수 있다.
- **목록에서만 보고 제외한 것:**
  - K-intelligence Midm-2.0-Mini-Instruct(2.3B, 2025-07), naver-hyperclovax SEED(Omni 8B 등), LGAI-EXAONE 4.0-1.2B(2B 미만), trillionlabs Trida-7B 등이 있었다.
  - 최대 3개 제한 때문에 상세 조사를 하지 않았다. 라이선스도 확인하지 않았으므로 판단도 하지 않는다.

## 미확인

- 8ca4060에서 실제 load, template 렌더, 한국어 연속 대화 품질 (A 실행 검증).
- gemma.ai 새 문서의 라이선스 표기. 이번에 확인한 것은 ai.google.dev 페이지뿐이다.
- Gemma 4에 Prohibited Use Policy 같은 추가 사용 정책이 붙는지. 라이선스 페이지 본문에는 Apache 2.0 전문만 있다.
- 법률 자문이 아니다.

SUPERVISION_REPORT_READY BASE-CANDIDATE-004
