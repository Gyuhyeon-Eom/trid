# 외부 가중치 없이 trid 배포 모델을 작고 빠르게 만드는 방법 (폐쇄망 CPU)

작성일: 2026-10-28. 대상: `trid` 91M(`runs/trid-korean-medium-base-v1`, 12층·768·12 head·FFN 2,048·문맥 1,024·자체 8,192 byte-level BPE, RoPE·SwiGLU·RMSNorm·임베딩 공유, GQA·QK-norm 옵션)을 **폐쇄망 공공기관 CPU**에서 돌릴 때의 모델 크기·추론 속도. 과제는 근거 문단에서 값을 그대로 복사해 `값 [Dn]`으로 답하거나 보류하는 것이다.

범위: 이 문서는 **크기·속도**만 다룬다. 시스템 구조·추출 vs 생성·검색 통합·충실성 학습은 [closed-network-architecture-survey.md](closed-network-architecture-survey.md)에, 토크나이저·학습 레시피·데이터 처리는 [korean-oss-opportunities.md](korean-oss-opportunities.md)에 있다. 겹치는 항목(MLA·슬라이딩 창·MobileLLM·MoE)은 그 문서의 "과제 적합성" 판단을 반복하지 않고 **파라미터·메모리·실측 속도** 관점만 보탠다.

표기 규칙
- 라이선스는 2026-10-28에 GitHub API·LICENSE 파일·논문 페이지에서 직접 확인했다. "상업적 이용"은 코드·도구를 제품 파이프라인에 쓰는 경우 기준이다. 외부 **가중치**는 어떤 항목에서도 쓰지 않는다.
- **(측정)**은 이 조사에서 Mac 한 대의 CPU로 직접 잰 값이다. 측정 중 91M 사전학습(MPS)과 수집기 2개가 동시에 돌고 있었으므로 절대값은 낮고 분산이 크다(부록 A). 상대 비교와 자릿수 판단에만 쓴다.
- 판정: **적용**(배포 경로에 넣을 것), **실험**(소규모 비교 후 결정), **보류**(가치 있으나 지금은 아님), **부적합**(측정·근거상 이득 없음).
- 난이도 S/M/L과 손댈 파일을 항목마다 적는다. 추정은 `[INFERENCE]`.

---

## 0. 요약

| 영역 | 핵심 발견 | 판정 |
|---|---|---|
| 현재 병목 | (측정) 평가셋 310문항의 프롬프트는 중앙값 162토큰, p90 816토큰, 정답은 중앙값 9토큰이다. 지연의 대부분은 **prefill**이고 생성은 10토큰 미만이다. 91M FP32 PyTorch CPU 4스레드: prefill 4,000–4,500 tok/s, 생성 101–106 tok/s. p90 질문 한 건이 0.3초 안팎이므로 **속도는 이미 과제 요건을 만족**하고, 남는 문제는 크기(365MB)·의존성(PyTorch)·제약 디코딩 비용이다. | — |
| GGUF 내보내기 | (측정) trid의 RoPE는 짝·홀 쌍 회전이라 ggml의 `NORM` 모드와 같고, 묶인 임베딩·RMSNorm·SwiGLU·GQA 모두 llama.cpp `llama` 아키텍처에 **텐서 이름 매핑만으로** 들어간다. 변환 스크립트 80줄로 F32 GGUF를 만들었고 CPU 탐욕 생성 24토큰이 PyTorch와 **글자 단위로 같았다**. C++ 수정 없음. | **적용** |
| 양자화 수준 | (측정) llama.cpp Q8_0: 92.5MiB, 한국어 PPL 27.57→27.61(+0.1%), 탐욕 출력 동일, 생성 316–423 tok/s(FP32 PyTorch의 3–4배). Q4_K_M: 53.8MiB, PPL 동률이지만 같은 프롬프트에서 출처 번호가 `[D4]`→`[D5]`로 **바뀌었다**. torchao RTN int4(g128)는 마지막 토큰 KL 0.156 nat, 생성 33 tok/s(arm 커널 없음). | **적용(Q8_0)**, 실험(Q4_K_M+imatrix), 부적합(PyTorch int4) |
| PyTorch 경로 양자화 | (측정) `torch.ao.quantization.quantize_dynamic`은 이 빌드에서 엔진 지정(`qnnpack`) 없이는 실패하고, 지정하면 생성 +19%·**prefill −58%**. torchao Int8DynAct도 같은 양상(생성 +23%, prefill −58%). Int8WeightOnly는 생성이 오히려 −22%. bf16 CPU는 prefill 13배 느림. torch.compile(CPU)은 prefill 동률, 생성 −10%. → PyTorch 경로는 FP32 eager가 최선. | 부적합(PyTorch 양자화) |
| 파라미터 효율 구조 | (측정) 91,245,312 중 임베딩 6.29M(6.9%), attention 28.3M(31%), FFN 56.6M(62%). GQA kv_heads=4면 −9.4M(−10%)로 MobileLLM이 125M에서 손실 없다고 보고한 범위. deep-thin(24층×576, GQA 9/3)은 89.7M으로 같은 예산. 층 공유·MLA·슬라이딩 창·NoPE·병렬 블록·MoE는 문맥 1,024·단일 요청·CPU에서 이득이 없거나 품질 위험. | 실험(GQA, deep-thin), 부적합(나머지) |
| 가지치기·증류 | 외부 교사 없이 **자기 91M을 교사**로 Minitron식 폭 가지치기(FFN 2,048→1,024·head 12→8 → 53.5M)와 KL 증류를 할 수 있다. Minitron은 1.8B 토큰(원 학습의 0.02%)으로 순위가 안정됐고 94B로 완성했다. 10억 토큰 예산에서 비어휘 최적 크기가 약 59M(OSS 조사 §1.2)이므로 방향은 맞지만, 현재 지연이 이미 충분해 **Pi급 대상이 확정될 때만** 한다. | 실험(2순위) |
| RAG 지연 | (측정) 고정 접두 `RAG_PREFIX`는 41토큰(중앙값 프롬프트의 25%, p90의 5%)이라 접두 캐시 이득이 작다. KV 캐시는 문맥 1,024에서 f16 36MiB로 병목이 아니고 q8_0 KV는 생성이 오히려 −8%. 생성이 9토큰이라 추측 디코딩은 부적합. 반면 `constrained.py`의 `choose()`는 토큰당 중앙값 3.1ms·p90 13.4ms로 **llama.cpp Q8_0의 토큰당 2.4ms보다 크다** → xgrammar식 마스크 캐시가 가장 싼 개선. | 적용(마스크 캐시), 부적합(KV 양자화·추측 디코딩·접두 캐시) |
| 한국어 레버 | (측정) 평가 프롬프트 1.786글자/토큰, 1,024토큰≈1,830글자. 바이트 폴백 토큰이 전체의 3.0%. 자모(NFD) 분해는 현 토크나이저에서 토큰 7.9배. 16k 어휘는 +8–11% 글자/토큰(OSS 조사 §1.2)에 임베딩 +6.3M(+7%)·Q8_0 +6.6MiB·lm_head 연산 +7%로 속도는 상쇄되고 문맥만 늘어난다. | 실험(16k, BPC로 결정), 부적합(자모) |

배포 형태 제안(§8): **91M · Q8_0 GGUF · llama.cpp(MIT) CPU 4스레드 · KV f16 · 문맥 1,024**, 제약 디코딩은 llama-cpp-python logits processor로 `EvidenceConstraint`를 그대로 사용. 예상 RSS 300MB 미만, 이 Mac 기준 p90 질문당 0.4초. PyTorch FP32 경로(1GB, 0.3초)는 테스트·개발용으로 유지.

---

## 1. 측정: 현행 91M의 구성과 CPU 속도

### 1.1 파라미터 분해 (측정)

`runs/trid-korean-medium-base-v1/checkpoints/step-00033250`을 `/tmp`로 복사해 `Trid.parameter_count()`와 모듈별 `numel()`로 셌다.

| 구성 요소 | 파라미터 | 비중 |
|---|---:|---:|
| 임베딩(입·출력 공유, 8,192×768) | 6,291,456 | 6.9% |
| attention(층당 qkv 1.77M + out 0.59M) ×12 | 28,311,552 | 31.0% |
| FFN(층당 gate·up·down 3×768×2,048) ×12 | 56,623,104 | 62.1% |
| RMSNorm | 19,200 | 0.0% |
| **합계** | **91,245,312** | FP32 365.0MB |

어휘 크기만 바꿨을 때(dim 768 고정, 계산):

| 어휘 | 임베딩 | 전체 | 임베딩 비중 | Q8_0 크기 `[INFERENCE: 92.5MiB×비율]` |
|---|---:|---:|---:|---:|
| 8,192 | 6.29M | 91.2M | 6.9% | 92.5MiB(측정) |
| 16,384 | 12.58M | 97.5M | 12.9% | ≈99MiB |
| 32,768 | 25.17M | 110.1M | 22.9% | ≈112MiB |

MobileLLM이 "32k 어휘·512차원에서 임베딩이 125M의 20% 이상"이라며 공유를 권한 상황([arXiv 2402.14905 §2.2.3](https://arxiv.org/html/2402.14905))과 달리 trid는 8k 어휘에 이미 공유라 임베딩 비중이 7%다. 어휘를 32k로 늘려도 23%에 그치지만, OSS 조사 §1.2의 어휘 스케일링 법칙은 10억 토큰 예산에 $V_{opt}\approx7{,}300$을 주므로 16k까지만 실험 대상이다.

### 1.2 KV 캐시 (계산)

토큰당 원소 수 $2\,n_{kv}\,d_h\,L$에 문맥 1,024를 곱했다([DeepSeek-V2 Table 1](https://arxiv.org/html/2405.04434v5) 공식).

| attention | 토큰당 원소 | 1,024토큰 FP32 | f16 | q8_0 |
|---|---:|---:|---:|---:|
| MHA(현행, kv 12) | 18,432 | 75.5MB | 37.7MB | 18.9MB |
| GQA kv_heads=4 | 6,144 | 25.2MB | 12.6MB | 6.3MB |
| GQA kv_heads=3 (MobileLLM 9/3 비율) | 4,608 | 18.9MB | 9.4MB | 4.7MB |
| MQA kv 1 | 1,536 | 6.3MB | 3.1MB | 1.6MB |
| MLA $d_c=4d_h=256,\ d_h^R=32$ | 3,456 | 14.2MB | 7.1MB | 3.5MB |

단일 요청·문맥 1,024에서 KV 최대치는 가중치(Q8_0 92.5MiB)와 같은 자릿수이고, 생성이 9토큰이라 KV를 읽는 횟수도 9번이다. KV 축소는 **동시 요청 수를 늘릴 때만** 의미가 있다.

### 1.3 PyTorch CPU: FP32 대 양자화 (측정)

조건: Apple M5(성능 4·효율 6코어), 메모리 32GB, PyTorch 2.14.1(프로젝트 venv), `torch.set_num_threads(4)`, `inference_mode`, KV 캐시 사용. 프롬프트는 `data/korean-rag-eval-v1/test.jsonl`의 프롬프트를 이어 붙여 `<|bos|>` 포함 **정확히 300토큰**(504글자). prefill 1회 + 32토큰 탐욕 생성, 예열 2회 뒤 5회 중앙값. torchao 0.18.0과 gguf 0.19.0은 `/tmp/extra-site`에 따로 설치해 venv를 바꾸지 않았다.

| 설정 | prefill tok/s | 생성 tok/s | state_dict 크기 | 마지막 토큰 KL(nat, FP32 대비) | 탐욕 32토큰 |
|---|---:|---:|---:|---:|---|
| FP32 eager (`Trid` 원본) | 4,472 | 105.9 | 365.0MB | 0 | 기준 |
| FP32 eager (lm_head를 nn.Linear로 분리한 래퍼) | 4,008 | 101.2 | 390.2MB | 0 | 동일 |
| `torch.ao.quantization.quantize_dynamic` int8 (qnnpack, Linear만; lm_head FP32) | 1,880 | 126.3 | 110.6MB | — | 3번째 토큰부터 다름 |
| 같은 방법, lm_head 포함 | 1,723 | 127.5 | 117.0MB | 0.018 | 3번째 토큰부터 다름 |
| torchao `Int8WeightOnlyConfig` | 3,568 | 79.0 | 117.1MB | 0.0008 | 동일 |
| torchao `Int8DynamicActivationInt8WeightConfig` | 1,675 | 124.9 | 117.1MB | 0.0022 | 3번째 토큰부터 다름 |
| torchao `IntxWeightOnlyConfig` int4 g128 (RTN) | 2,774 | 33.5 | 120.1MB(비패킹) | 0.156 | 3번째 토큰부터 다름 |
| torchao `Int8DynamicActivationIntxWeightConfig` int4 g128 | 2,415 | 31.3 | 120.1MB | 0.161 | 3번째 토큰부터 다름 |
| torchao `Int4WeightOnlyConfig` | 실패(`Requires mslk >= 1.0.0`) | | | | |
| `torch.compile(dynamic=True)` CPU inductor, FP32 (컴파일 10.7초) | 4,068 | 90.8 | — | 0 | 동일 |
| bf16 eager CPU | 311 | 101.6 | 195.1MB | 0.00008 | 동일 |

읽는 법:
- `quantize_dynamic`은 이 PyTorch 빌드에서 `torch.backends.quantized.engine`이 `none`이라 그대로 호출하면 `NoQEngine` 오류가 나고, `qnnpack`으로 지정해야 돈다. API 자체가 2.10에서 제거 예고(deprecation 경고)라 신규 코드에 넣을 것은 아니다.
- int8 **동적** 양자화는 활성값을 토큰마다 양자화하므로 prefill(300토큰 행렬곱)에서 FP32 Accelerate BLAS보다 2.4배 느리고, 생성(1토큰)에서만 20% 빠르다. trid 지연은 prefill이 지배하므로 순손실이다.
- int4는 torchao의 arm 전용 커널(KleidiAI, [torchao 문서](https://docs.pytorch.org/ao/stable/workflows/inference.html) "experimental, ARM CPU")이 pip 휠에 없어서 역양자화 경로로 떨어져 생성이 3배 느리다. 품질도 RTN g128 기준 KL 0.16으로 크다(§3).
- 결론: **PyTorch 경로에서는 FP32 eager를 그대로 쓴다.** 양자화 이득은 llama.cpp에서만 난다(§1.4).

### 1.4 llama.cpp: GGUF 내보내기와 속도 (측정)

**변환.** `gguf` 패키지(llama.cpp 저장소의 gguf-py, MIT)로 trid 체크포인트를 `general.architecture = llama`로 썼다. 매핑:

| trid 텐서 (`trid/model.py`) | GGUF 이름 | 비고 |
|---|---|---|
| `embedding.weight` [8192,768] | `token_embd.weight` | `output.weight`를 쓰지 않으면 llama.cpp가 `token_embd`를 출력에 재사용(묶인 임베딩) |
| `norm.weight` | `output_norm.weight` | |
| `blocks.i.attn_norm.weight`, `ffn_norm.weight` | `blk.i.attn_norm.weight`, `blk.i.ffn_norm.weight` | RMSNorm, `attention.layer_norm_rms_epsilon = 1e-5`(`model.py:46`) |
| `blocks.i.attn.qkv.weight` [dim+2·kv_dim, 768] | `blk.i.attn_q/k/v.weight` | `split((dim, kv_dim, kv_dim))`로 나눔. GQA는 `attention.head_count_kv` |
| `blocks.i.attn.output.weight` | `blk.i.attn_output.weight` | |
| `blocks.i.gate/up/down.weight` | `blk.i.ffn_gate/up/down.weight` | SwiGLU = llama FFN |
| RoPE | `rope.dimension_count = 64`, `rope.freq_base = 10000` | **치환 불필요** (아래) |
| 토크나이저 | `tokenizer.ggml.model = gpt2`, `pre = gpt-2`, tokens·types·merges, bos 1·eos 4·pad 0 | `tokenizer.json`의 `ByteLevel(use_regex=True)` = GPT-2 정규식 |

RoPE: `model.py:60-61`은 `x[..., 0::2], x[..., 1::2]` 즉 **인접 쌍 회전**이고, llama.cpp는 `LLM_ARCH_LLAMA`에 대해 "normal RoPE, operating on pairs of consecutive head values"(`LLAMA_ROPE_TYPE_NORM`, [src/llama-model.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model.cpp))를 쓴다. HF Llama는 rotate-half(NeoX식)라 `convert_hf_to_gguf.py`가 q·k를 치환하지만 trid는 치환 없이 그대로 들어간다. QK-norm을 켠 구성은 `qwen3` 아키텍처(`attn_q_norm`, `attn_k_norm`, head 차원 RMSNorm → RoPE 순서가 trid와 같고 묶인 임베딩 허용)로 보낸다. 단 qwen3는 NEOX RoPE라 q·k 투영 행과 q_norm·k_norm 가중치를 head 안에서 `(0,2,4,…,1,3,5,…)`로 치환한다(`scripts/export_gguf.py`). (측정, 2026-10-28) 무작위 작은 qk_norm·GQA 모델을 teacher forcing으로 48위치 비교해 argmax 불일치 0, top-5 logprob 최대 절대차 1.3e-4였다 (`reports/gguf-export-parity.json` `qk_norm_qwen3_tiny_check`).

**검증.** 같은 300토큰 프롬프트를 (a) PyTorch FP32, (b) ollama 0.31.1(내장 llama.cpp, `num_gpu 0`, 4스레드), (c) 직접 빌드한 llama.cpp(commit 8ca4060, 2026-10-28, `GGML_METAL=OFF`, Accelerate BLAS)로 돌렸다. F32·Q8_0의 탐욕 24토큰이 PyTorch와 **같았다**(`…며칠간 진행한다.\n[D4] [복무 안내] …`). "prompt eval count: 300"으로 토크나이저 매핑도 확인됐다.

**속도** (`llama-bench -t 4 -r 10`, 동시 부하 상태라 ±10–25%):

| 형식 | 크기 | pp300 tok/s | pp1000 tok/s | tg32 tok/s |
|---|---:|---:|---:|---:|
| F32 | 348.1MiB | 2,311 ±482 | 1,901 ±239 | 113 ±13 |
| F16 | 174.1MiB | 1,862 ±471 | — | 191 ±51 |
| **Q8_0** | **92.5MiB** | 3,006 ±267 | 2,500 ±240 | **423 ±69** (별도 실행 316 ±63) |
| Q4_K_M | 53.8MiB | 2,244 ±167 | 1,702 ±363 | 452 ±101 (별도 실행 515 ±165) |
| Q4_0 | 50.6MiB | 2,035 ±256 | — | 305 ±122 |
| Q8_0 + `-fa 1 -ctk q8_0 -ctv q8_0` | 92.5MiB | — | — | 388 ±59 |

참고로 ollama F32: prompt eval 2,942 tok/s, eval 124 tok/s. 커뮤니티 측정으로는 SmolLM2-135M F16이 Apple M4에서 241 tok/s([MetriLLM](https://metrillm.dev/models/smollm2-135m), 조건 불명)이고 MobileLLM-125M은 iPhone 13 MPS에서 토큰당 15.6ms(64 tok/s, [arXiv 2402.14905 Table 7](https://arxiv.org/html/2402.14905))다. trid Q8_0의 423 tok/s는 그 사이에서 자릿수가 맞다.

**품질** (`llama-perplexity -c 1024 --chunks 12`, 평가셋 프롬프트 120건 25,304글자; 학습에 쓰지 않은 공문·KLUE 문단):

| 형식 | PPL | FP32 대비 | 같은 프롬프트 탐욕 24토큰 |
|---|---:|---:|---|
| F32 | 27.57 ±1.11 | — | `…진행한다.\|[D4] [복무 안내]…` |
| Q8_0 | 27.61 ±1.11 | +0.1% | 동일 |
| Q4_K_M | 27.47 ±1.10 | −0.4% (잡음 범위) | `…진행.\|[D5] [복무 안내]…` — **출처 번호가 바뀜** |
| Q4_0 | 27.78 ±1.11 | +0.8% | `…진행.\|[D4]…` |

PPL은 12k 토큰이라 ±1.1의 표준오차가 있지만 같은 텍스트의 짝 비교다. 핵심은 Q4가 PPL로는 보이지 않는 **상위 1 토큰 뒤집힘**을 일으키고, 그것이 trid에서 가장 치명적인 출처 번호라는 점이다. 310문항 전수 평가 전에는 Q4를 쓰지 않는다.

### 1.5 프롬프트 길이·제약 디코딩 비용 (측정)

- `data/korean-rag-eval-v1/test.jsonl` 310건 프롬프트: 중앙값 162토큰, 평균 366, p90 816, 최대 956. 글자/토큰 1.786. 정답 토큰 중앙값 9.
- `trid/prompts.py`의 `RAG_PREFIX`는 **41토큰**.
- `EvidenceConstraint`(평가셋 40건, 무작위 점수, 토큰 320회): 생성자 0.15ms, `choose()` 중앙값 **3.13ms**, p90 **13.4ms**. 무작위 점수는 상위 후보가 거부되는 횟수가 많아 비관적 상한이다 `[INFERENCE: 실제 logits에서는 상위 1이 허용 토큰인 경우가 많아 더 짧음]`. 그래도 llama.cpp Q8_0의 토큰당 2.4ms(423 tok/s)보다 길다.

질문 한 건의 지연 분해(이 Mac, p90 816토큰 + 9토큰 생성):

| 경로 | prefill | 생성 9토큰 | 제약 9토큰 | 합계 |
|---|---:|---:|---:|---:|
| PyTorch FP32 4스레드 | 816/4,000 = 0.20s | 9×9.4ms = 0.08s | 9×3ms = 0.03s | ≈0.31s |
| llama.cpp Q8_0 4스레드 | 816/2,500 = 0.33s | 9×2.4ms = 0.02s | 0.03s | ≈0.38s |

x86 폐쇄망 서버는 측정하지 못했다. llama.cpp는 AVX2/AVX-512 VNNI int8 커널이 있어 Q8_0의 상대 이득은 유지될 것이고, 절대값은 코어당 Apple M5보다 느릴 것이다 `[INFERENCE]`. 2–4배 느려도 질문당 1.5초 안이다.

---

## 2. 파라미터 효율 구조 (~100M)

| 항목 | 내용·출처·라이선스 | 핵심 수치 | 91M·Mac 한 대 적용 | 난이도 | 손댈 파일 |
|---|---|---|---|---|---|
| deep-and-thin | MobileLLM([ICML 2024, arXiv 2402.14905](https://arxiv.org/abs/2402.14905); 코드 [facebookresearch/MobileLLM](https://github.com/facebookresearch/MobileLLM) **FAIR NC**(비상업) — 코드는 못 쓰지만 방법은 논문 공개) | 125M에서 12층→30·42층이 유의하게 우수; 125M 구성 30층×576·FFN 1,536·head 9/kv 3([config](https://raw.githubusercontent.com/facebookresearch/MobileLLM/main/configs/125M/config.json)); 0.25T 토큰으로 탐색 | **실험**. 같은 91M 예산의 trid 변형은 24층×576·FFN 1,536·head 9/kv 3 = 비임베딩 84.96M + 임베딩 4.72M = **89.7M**(계산). 증거는 1T 토큰·영어 상식 과제라 10억 토큰·한국어 복사 과제에 외삽. MPS에서 층 2배는 커널 호출 2배라 처리량 하락 우려(ClosedNet 조사 §4도 같은 지적). CPU 생성은 메모리 바이트가 같아 비슷하고 prefill FLOPs도 같음 `[INFERENCE]` | S | `config` JSON, `trid/model.py`(변경 없음), 20M 토큰 파일럿 |
| 즉시 블록 가중치 공유 | MobileLLM-LS, 같은 논문 §2.3 | 125M +0.7%p, 350M +0.8%p, 크기 동일; iPhone MPS에서 실행 시간 +2.6%(메모리 바운드 환경) | **보류**. Q8_0 92.5MiB를 2.4ms에 읽는 현재 속도는 약 40GB/s 수준으로 M5 대역폭에 못 미치는 **오버헤드·연산 바운드**라 층 2배는 지연도 거의 2배 `[INFERENCE: 1.4절 수치에서 유도]`. 크기 상한이 걸릴 때만 가치 있음. llama.cpp `llama` 아키텍처는 공유 층을 표현 못 해 새 아키텍처 필요 | M | `trid/model.py` `Trid.forward`, GGUF 변환기 |
| 임베딩 공유 | Press & Wolf 2016; MobileLLM Table 1 | 30층·512 모델에서 −16M(−11.8%)에 −0.2점, 절약분으로 층을 늘리면 +0.4점 | **적용(이미 적용)**. trid는 6.29M(6.9%) 절약 중 | — | — |
| GQA | MobileLLM §2.2.4; SmolLM3([블로그](https://huggingface.co/blog/smollm3)) 3B·100B 토큰 ablation | 125M에서 kv 16→4 동등, 350M −0.2점에 크기 −10%; SmolLM3 GQA 4그룹이 MHA와 동등 | **실험(1순위 구조 변경)**. kv_heads=4면 attention −9.4M(전체 −10.3%), KV 캐시 1/3. 옵션이 이미 있어 config만 바꾸면 됨. 다음 10억 토큰 실행 전 20M 파일럿에서 BPC 비교 | S | `config` JSON (`kv_heads: 4`), `tests/` |
| MLA | DeepSeek-V2([arXiv 2405.04434](https://arxiv.org/abs/2405.04434)), 코드 MIT | KV를 GQA 2.25그룹 수준으로 압축하면서 MHA보다 강함(236B MoE) | **부적합**. 1.2절: 문맥 1,024·단일 요청에서 KV 최대 75MB(FP32)로 병목 아님. 분리 RoPE·가중치 흡수 구현 M, llama.cpp는 `deepseek2` 아키텍처 매핑 필요 | M | `trid/model.py` `Attention` 전면 |
| 슬라이딩 창 + 전역 5:1 | Gemma 3([arXiv 2503.19786 §2, §5.2](https://arxiv.org/html/2503.19786v1)) | 창 1,024·5:1에서 PPL 영향 최소, 32k 문맥에서 KV 오버헤드 60%→15% 미만 | **부적합(메모리 목적)**. trid 문맥이 1,024라 창 1,024는 전역과 동일하고, 창을 줄이면 질문(끝)이 앞 문단을 못 봄. 품질 목적의 귀납 편향 실험은 ClosedNet 조사 §4에 있음 | S | `trid/model.py` `Attention.forward` 마스크 |
| NoPE·부분 RoPE | SmolLM3: 4층마다 RoPE 제거("RoPE to NoRoPE and Back Again", [arXiv 2501.18795](https://arxiv.org/abs/2501.18795)) | 3B·100B 토큰에서 단문맥 유지, 장문맥 개선 | **부적합**. 장문맥 확장용이고 trid는 1,024 고정. llama.cpp `llama` 매핑도 깨짐(`smollm3` 아키텍처 필요) | S | `trid/model.py` |
| 병렬 attention+FFN | GPT-J식, PaLM([arXiv 2204.02311 §2](https://arxiv.org/html/2204.02311v4)) | 대규모에서 학습 15% 빠름; **8B에서 소폭 품질 저하**, 62B에서 없음 | **부적합**. 91M은 8B보다 훨씬 작아 저하 쪽. MPS 처리량 이득은 행렬곱 융합 전제라 작은 모델에서는 미미 `[INFERENCE]`. GGUF 매핑도 `gptneox`/`falcon` 계열로 바뀜 | S | `trid/model.py` `Block.forward` |
| GLU 변형 | Shazeer([arXiv 2002.05202](https://arxiv.org/abs/2002.05202)); ReLU 희소성(ReLU Strikes Back, [arXiv 2310.04564](https://arxiv.org/abs/2310.04564)) | SwiGLU/GEGLU 최상; MobileLLM 125M ReLU-FFN→SwiGLU 42.6→43.9; ReLU 활성 희소성으로 추론 연산 최대 3배 절감 | **적용(이미 SwiGLU)**. ReLU² 희소 FFN은 llama.cpp에 희소 커널이 없어 CPU 이득이 실현되지 않음 → 보류 | — | — |
| 소형 MoE·가중치 공유 MoE | OLMoE 등(ClosedNet 조사 §4 참조) | — | **부적합**. 10억 토큰 예산의 비어휘 최적 크기가 ≈59M(OSS 조사 §1.2)이라 trid는 **파라미터가 아니라 데이터가 부족**하다. 총 파라미터를 늘리는 MoE는 반대 방향. CPU 배포에서 RAM은 남지만 MPS 학습에서 작은 전문가 행렬곱은 비효율 | L | — |

---

## 3. CPU 양자화

### 3.1 소형 모델이 더 많이 깎이는 근거

| 출처 | 수치 |
|---|---|
| GPTQ([ICLR 2023, arXiv 2210.17323 Table 3](https://arxiv.org/html/2210.17323v2), CC BY 4.0) | WikiText2 PPL, OPT-125M: FP16 27.65 → RTN 4bit **37.28(+35%)**, GPTQ 4bit 31.12(+12.5%), GPTQ 3bit 53.85. OPT-175B: 8.34 → RTN 10.54(+26%), GPTQ 8.37(**+0.4%**). 논문 본문: "larger models generally appear easier to quantize" |
| Scaling Laws for Precision([ICLR 2025, arXiv 2411.04330](https://arxiv.org/abs/2411.04330)) | 사후 양자화 손실은 **학습 토큰이 많을수록 커진다**(최대 1.7B·26B 토큰 적합). trid는 토큰/파라미터 ≈ 11(10억 토큰)로 낮은 편이라 아직 유리한 구간 `[INFERENCE]` |
| MobileLLM §3.4 | 125M/350M W8A8 per-token min-max PTQ 손실 0.5점 미만 |
| torchao 문서 | Llama-3.1-8B int8 weight-only PPL 7.33→7.35, int8/int8 7.45 |
| (측정, 1.3–1.4절) | trid 91M: Q8_0 PPL +0.1%·탐욕 동일; int8 동적 KL 0.002–0.018, 3번째 토큰부터 분기; RTN int4 g128 KL 0.156; Q4_K_M PPL 동률이나 출처 번호 뒤집힘; Q4_0 PPL +0.8% |

결론: **8비트는 91M에서도 사실상 무손실**이고, 4비트는 형식(블록 크기·k-quant 스케일)에 따라 PPL은 지키지만 상위 1 토큰이 흔들린다. 4비트가 필요한 이유(RAM)가 trid 배포 대상에는 없다(92.5MiB vs 53.8MiB).

### 3.2 라이브러리별 지원 범위

| 라이브러리 | 라이선스 | 방식 | 사용자 정의 `nn.Module`(trid) 지원 | 91M 적용 | 난이도 | 손댈 파일 |
|---|---|---|---|---|---|---|
| llama.cpp `llama-quantize`([ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)) | MIT | Q8_0·Q4_K_M 등 블록 RTN, 선택적 imatrix(중요도 행렬) | GGUF만 있으면 됨(1.4절에서 검증) | **적용(Q8_0)**, 실험(Q4_K_M + imatrix) | S | `scripts/export_gguf.py`(신규) |
| torchao([pytorch/ao](https://github.com/pytorch/ao)) | BSD-3 | `quantize_()`가 모든 `nn.Linear` 교체; QAT API 포함 | **됨**(측정). 단 trid의 묶인 lm_head는 `F.linear(hidden, embedding.weight)`라 Linear가 아니어서 래퍼 필요 | 부적합(CPU 속도, 1.3절). QAT는 int4 필요 시에만 | S | `trid/runtime.py` |
| `torch.ao.quantization.quantize_dynamic` | PyTorch(BSD) | 레거시 eager int8 동적 | 됨, `engine='qnnpack'` 필수 | 부적합(prefill −58%, deprecated) | S | — |
| GPTQModel([ModelCloud/GPTQModel](https://github.com/ModelCloud/GPTQModel)) | Apache-2.0 | GPTQ(헤시안 기반 PTQ) | HF `transformers` 모델 등록 전제 → trid HF 래퍼 필요 | 보류. 4비트가 필요해지면 llama.cpp imatrix와 비교 | M | HF 래퍼(신규) |
| llm-awq([mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq)) | MIT | 활성값 인지 스케일링 | HF 모델 전제; 알고리즘은 작아 재구현 가능 | 보류 | M | — |
| hqq([mobiusml/hqq](https://github.com/mobiusml/hqq)) | Apache-2.0 | 반복 최적화 PTQ, `HQQLinear`로 모듈 교체 | 됨(모듈 단위) | 보류. CPU 커널 없음 | S | — |
| SmoothQuant([mit-han-lab/smoothquant](https://github.com/mit-han-lab/smoothquant)) | MIT | W8A8 활성 이상값 완화 | 알고리즘만; HF 전제 | 부적합. W8 weight-only(Q8_0)면 활성 양자화가 필요 없음 | — | — |
| Gemma 3 QAT 레시피([arXiv 2503.19786 §2.3](https://arxiv.org/html/2503.19786v1)) | 방법 | 비양자화 체크포인트의 확률을 목표로 약 5,000스텝 QAT | torchao QAT로 재현 가능 | 보류(int4 결정 후) | M | `trid/training.py` |

---

## 4. 가지치기·증류 (외부 교사 없이)

| 항목 | 내용·출처·라이선스 | 핵심 수치 | 91M 적용 | 난이도 | 손댈 파일 |
|---|---|---|---|---|---|
| Minitron식 폭 가지치기 + 자기 증류 | [NeurIPS 2024, arXiv 2407.14679](https://arxiv.org/abs/2407.14679). 저장소 [NVlabs/Minitron](https://github.com/NVlabs/Minitron)은 README뿐(LICENSE 없음); 구현은 NeMo(Apache-2.0). 방법은 논문 | 15B→8B/4B를 원 학습의 1/40 토큰(94B)으로; 후보 순위는 **1.8B 토큰**(0.02%) 재학습에서 안정; **폭(head·FFN·임베딩 채널) 가지치기가 깊이보다 우수(≤15B)**; 손실은 logits KLD만이 최선(Table 15: 2.117→2.107); 중요도는 활성값 (L2, mean) 1회 | **실험(2순위)**. 교사 = 자기 91M(정책 위반 없음). 학생 예: head 12→8·FFN 2,048→1,024(dim 768 유지) → 47.2M + 임베딩 6.3M = **53.5M**(계산), OSS 조사 §1.2의 1B-토큰 최적 비어휘 59M과 일치. 재학습 100M 토큰 ≈ 2.2k tok/s 기준 13시간 + 교사 forward. 이득은 크기 −41%·CPU 생성 약 1.7배 `[INFERENCE]`인데 현 지연이 이미 충분해 Pi급 대상이 확정될 때 실행 | M | `scripts/prune_distill.py`(신규), `trid/training.py`(KL 손실), `trid/model.py`(head 수 축소 시 `Config`) |
| Sheared-LLaMA | [ICLR 2024, arXiv 2310.06694](https://arxiv.org/abs/2310.06694), [princeton-nlp/LLM-Shearing](https://github.com/princeton-nlp/LLM-Shearing) MIT | 7B→1.3B/2.7B를 **3% 계산**(0.4B 가지치기 + 50B 계속 학습); 도메인 손실로 배치 구성을 바꾸는 dynamic batch loading | 보류. 마스크 학습(Lagrangian) 구현이 Minitron보다 무겁고 91M에서는 이득 근거 없음. dynamic batch loading 아이디어는 법령·보도자료·위키 혼합비 조절에 쓸 수 있음(데이터 쪽) | L | `scripts/build_pretraining_mix.py` |
| 체크포인트 평균 | LAWA([COLM 2024, arXiv 2306.03241](https://arxiv.org/abs/2306.03241)); trid는 이미 v4 200·800스텝 평균을 썼음(`DIALOGUE.md`, `MODEL_CARD.md`) | nanoGPT 125M·335M·770M에서 간격 둔 체크포인트 평균이 EMA·SWA보다 수렴·일반화 우수, 높은 lr에서 이득 큼 | **적용**. WSD stable 구간 마지막 k개(예: 250스텝 간격 4개) 평균을 배포 후보로 만들고 BPC·310문항으로 비교. 학습 비용 0 | S | `scripts/average_checkpoints.py`(신규), `trid/pretraining.py`(체크포인트 보존 수) |
| 같은 크기 자기 증류(EMA 교사) | Born-again류; 소형 LM 증거 없음 | — | 보류. Minitron도 "교사는 더 큰 원본"일 때 효과를 보고(§4.3 "15B 교사가 8B 교사보다 우수"). 크기를 줄이지 않는 자기 증류는 근거 부족 | M | — |
| 외부 교사 증류 | MobileLLM §3.5: LLaMA-2 7B 교사 KD가 125M/350M에서 **학습 2.6–3.2배 느리고 정확도 동등 이하** | — | 부적합(정책상 금지이기도 함). 작은 모델은 교사가 있어도 이득이 작다는 반증 | — | — |

---

## 5. 런타임

| 런타임 | 라이선스 | trid에 필요한 것 | 측정·근거 | 판정 | 난이도 | 손댈 파일 |
|---|---|---|---|---|---|---|
| llama.cpp(GGUF `llama` 아키텍처) | MIT | 변환 스크립트(1.4절 표, ~80줄). **새 아키텍처 추가 불필요**([HOWTO-add-model.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/development/HOWTO-add-model.md)의 2·3단계 생략). 제약 디코딩은 llama-cpp-python(MIT) `logits_processor`에서 `EvidenceConstraint.choose`를 호출하거나, 토큰마다 `eval`을 부르는 Python 루프 | (측정) 탐욕 출력 동일, Q8_0 생성 3–4배, 크기 1/4, 단일 바이너리 배포 | **적용** | S(변환)+M(제약 통합) | `scripts/export_gguf.py`, `trid/runtime.py`(백엔드 옵션), `scripts/evaluate_rag_eval.py --backend` |
| PyTorch FP32 eager | BSD | 없음(현행) | (측정) prefill 4,000–4,500 tok/s로 llama.cpp보다 빠름(Accelerate). 생성 101–106 tok/s. RSS ≈ 1GB `[INFERENCE: 가중치 365MB + torch 런타임]` | 적용(개발·테스트 경로 유지) | — | — |
| torch.compile CPU inductor | BSD | `dynamic=True` 필요(캐시 길이 가변), 첫 호출 10.7초 | (측정) prefill 동률, 생성 −10% | 부적합 | — | — |
| ONNX Runtime([microsoft/onnxruntime](https://github.com/microsoft/onnxruntime)) | MIT | `torch.onnx.export`(dynamo)로 KV 캐시 입·출력을 동적 축으로 내보내고, `quantize_dynamic`(MatMulInteger int8) | 미측정. 사용자 정의 모듈이라도 ONNX 연산자만 쓰면 됨 | 보류. GGUF 경로가 이미 검증돼 두 번째 포맷을 유지할 이유가 약함. Windows 폐쇄망에서 Python 없이 C#/C++로 붙여야 할 때 재검토 | M | `scripts/export_onnx.py`(신규) |
| OpenVINO([openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino)) | Apache-2.0 | `openvino.convert_model(torch_model)` + NNCF 가중치 압축 | 미측정. Intel CPU 특화 | 보류(위와 같음) | M | — |
| ExecuTorch([pytorch/executorch](https://github.com/pytorch/executorch)) | BSD | export + XNNPACK | MobileLLM 측정 환경. 모바일 특화라 서버 CPU에는 llama.cpp가 단순 | 부적합 | L | — |

---

## 6. RAG 메모리·지연

| 항목 | 내용·출처 | 핵심 수치 | 91M 적용 | 난이도 | 손댈 파일 |
|---|---|---|---|---|---|
| KV 캐시 양자화 | llama.cpp `-ctk/-ctv q8_0`(`-fa 1` 필요); KIVI([jy-yuan/KIVI](https://github.com/jy-yuan/KIVI) MIT, 2비트) | (측정) Q8_0 생성 423→388 tok/s(−8%), KV f16 36MiB→18MiB | **부적합**. 문맥 1,024·단일 요청에서는 절약이 18MiB뿐 | — | — |
| 공유 접두 KV 캐시 | llama-server `cache_prompt`(자동 접두 재사용); PyTorch는 `[bos]+RAG_PREFIX` 41토큰의 KV를 미리 계산 | (측정) 접두 41토큰 = 중앙값 프롬프트의 25%, p90의 5%. 절약 ≤ 10ms | 보류. 구현 S이지만 이득이 p90에서 3%. `RAG_PREFIX`가 `\n`으로 끝나 GPT-2 정규식이 접두 경계에서 토큰을 자르므로 경계는 안정 `[INFERENCE: 정규식 기준, 테스트 필요]` | S | `trid/runtime.py` |
| 문단 단위 KV 캐시 | 근거 문단마다 KV를 저장해 재사용 | — | 부적합. 인과 attention에서 문단 앞의 내용(앞 문단·접두)이 바뀌면 KV가 무효. 문단 순서가 질문마다 바뀜 | — | — |
| 추측 디코딩(자기 추측·LayerSkip) | [arXiv 2404.16710](https://arxiv.org/abs/2404.16710), 코드 [facebookresearch/LayerSkip](https://github.com/facebookresearch/LayerSkip) **CC BY-NC 4.0**(비상업) | 요약 2.16배, 코드 1.82배(생성 길이 수백 토큰) | **부적합**. 정답 중앙값 9토큰, 생성이 지연의 10% 미만(1.5절). 17M 세대를 draft로 쓰는 고전 추측 디코딩도 같은 이유 | — | — |
| 제약 디코딩 마스크 캐시 | XGrammar([MLSys 2025, arXiv 2411.15100](https://arxiv.org/abs/2411.15100), [mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar) Apache-2.0) | 어휘를 문맥 독립/의존 토큰으로 나눠 독립 토큰(보통 99% 이상) 마스크를 사전 계산, 최대 100배 | **적용(아이디어)**. (측정) `choose()` 3.1ms/13.4ms가 Q8_0 토큰당 2.4ms를 넘음. 요청마다 근거 문단 바이트열에 대한 suffix automaton(또는 글자 경계 접두 집합)을 만들고, 8,192 토큰 바이트열 트라이와 교차해 **접두 상태별 허용 토큰 비트마스크**를 캐시. `scores` argsort 뒤 비트마스크로 걸러 첫 허용 토큰을 고르면 토큰마다 8,192개 검사가 사라짐. 출력은 바이트 단위로 동일해야 함(테스트로 고정) | S–M | `trid/constrained.py`, `tests/test_constrained.py` |
| 생성 상한 | 정답 9토큰, `max_tokens=64`(`evaluate_rag_eval.py`) | — | 적용. 상한을 32로 줄여도 완성 가능성 검사(`cost`)가 보장하므로 지연 상한만 내려감 | S | `scripts/evaluate_rag_eval.py` |

---

## 7. 한국어 크기 레버

| 항목 | 측정·근거 | 영향 | 판정 |
|---|---|---|---|
| 글자/토큰 | (측정) 평가 프롬프트 1.786; OSS 조사 1.60–1.89 | 문맥 1,024 ≈ 1,830글자(접두·질문 제외 약 1,700글자의 근거). 토큰 수가 prefill 시간과 정비례 | — |
| 16k 어휘 | OSS 조사 §1.2: +8–11% 글자/토큰 | prefill −8–10%, 같은 문맥에 근거 +8–11%; 임베딩 +6.3M(+7%), Q8_0 +6.6MiB, lm_head 행렬곱 768×16k = 12.6M MAC/토큰(층 전체 85M MAC의 7%) → **속도는 상쇄, 문맥만 이득** | 실험(BPC 파일럿은 OSS 조사 행동 1에 이미 포함) |
| 바이트 폴백 | (측정) 평가 프롬프트 113,311토큰 중 불완전 UTF-8 바이트 토큰 3,364개(**3.0%**), 단일 바이트 토큰 12.0%; 어휘 8,192 중 완전한 문자열 7,519개 | 폴백은 희귀 한자·기호에서 발생. `constrained.py`가 256바이트 토큰을 요구하므로 폴백 제거 불가 | 적용(현행 유지) |
| 자모(NFD) 분해 | (측정) 같은 텍스트 NFD는 코드포인트 1.92배, 현 토크나이저 토큰 **7.94배** | 자모 전용 BPE를 새로 학습해도 코드포인트가 2배라 문맥·속도 손해. 한국어 전용 8k BPE가 이미 10만 어휘 다국어와 같은 압축(OSS 조사 §1.1) | 부적합 |
| 숫자·단위 토큰화 | OSS 조사 §1.3(Llama-3식 정규식 +3–4%) | 값 복사 과제에서 숫자 토큰 경계가 안정하면 제약 디코딩의 허용 토큰 수가 줄어 마스크 캐시도 작아짐 `[INFERENCE]` | 실험(OSS 조사 행동 1과 함께) |

---

## 8. 배포 형태 제안과 검증 실험

### 8.1 제안

| 항목 | 값 | 근거 |
|---|---|---|
| 모델 | 현행 91M(12×768) — 다음 10억 토큰 실행에서 `kv_heads=4`, 필요 시 24×576 변형은 파일럿(§2) 결과로 결정 | 속도는 이미 충분(1.5절); 크기·품질 결정은 BPC 파일럿 |
| 어휘 | 8k(16k는 BPC 파일럿 뒤) | §7 |
| 가중치 형식 | **GGUF Q8_0**, 92.5MiB | PPL +0.1%, 탐욕 동일(1.4절) |
| 런타임 | llama.cpp CPU 빌드(`GGML_METAL=OFF`, BLAS 선택), 4스레드, 문맥 1,024, KV f16, flash-attn 끔 | 1.4절 |
| 제약 디코딩 | `EvidenceConstraint`를 마스크 캐시 버전(§6)으로 바꾼 뒤 llama-cpp-python `logits_processor`로 호출 | 1.5절 |
| 예상 RAM | 가중치 92.5MiB + KV 36MiB + 연산 버퍼·토크나이저 수십 MB → **RSS 300MB 미만** `[INFERENCE: 버퍼 크기 미측정]` | 1.2·1.4절 |
| 예상 지연(이 Mac) | 중앙값 162토큰 ≈ 0.10s, p90 816토큰 ≈ 0.4s, 최대 956 ≈ 0.45s | 1.5절 |
| 대체 경로 | PyTorch FP32 eager(현행): RSS ≈ 1GB, p90 0.31s. 테스트 108개와 평가 스크립트가 이 경로라 유지 | 1.3절 |
| 하지 않을 것 | PyTorch int8/int4·bf16·compile, KV 양자화, 추측 디코딩, MLA, 슬라이딩 창, NoPE, 병렬 블록, MoE, 외부 교사 증류 | 각 절 측정·근거 |

### 8.2 검증 실험 (`data/korean-rag-eval-v1`, 310문항, `scripts/evaluate_rag_eval.py` strict/value/source/abstained)

| # | 실험 | 통과 기준 | 비용 |
|---|---|---|---|
| E1 | GGUF F32 vs PyTorch FP32: 프롬프트 50건 마지막 logits 최대 절대차, 310문항 제한 디코딩 결과 | 310건 답·출처 바이트 동일; logits 차 1e-3 이하 `[INFERENCE: BLAS 누적 순서 차]` | CPU 10분 |
| E2 | Q8_0 vs F32 | strict·source 차 ≤1문항 | CPU 10분 |
| E3 | Q4_K_M(imatrix 없음/있음, 보정 자료 = `korean-foundation-v3` valid 1M 토큰) vs F32 | 출처 뒤집힘 건수 보고; strict 차 0이 아니면 Q4 채택 안 함 | CPU 30분 |
| E4 | 지연: 310문항 `total_seconds`·`first_token_seconds` p50/p90, PyTorch FP32 vs llama.cpp Q8_0, 제약 디코딩 시간 분리 기록 | §8.1 예상치 ±50% | CPU 20분 |
| E5 | 마스크 캐시 `constrained.py`: `choose()` 토큰당 ms 전후, 310문항 출력 바이트 동일 | 동일 출력, p90 < 1ms | CPU 10분 |
| E6 | 다음 사전학습 전 20M 토큰 파일럿 3개: 12×768 MHA / 12×768 kv4 / 24×576 kv3 — 도메인별 BPC(`scripts/evaluate_bpc.py`)와 MPS tok/s | BPC 차 0.01 이내면 작은 쪽 채택 | MPS 3×3시간 |
| E7 | (조건부) 53.5M 학생 자기 증류 100M 토큰 → BPC·310문항 vs 91M | strict −2문항 이내면 Pi급 배포 후보 | MPS 15–20시간 |

---

## 9. 우선순위 행동 목록

1. **GGUF 내보내기 스크립트 + Q8_0 + 동일성 테스트** (완료, 2026-10-28). `scripts/export_gguf.py`(1.4절 매핑, `--quant f32|f16|q8_0`, `--check`), `tests/test_export_gguf.py`, `scripts/gguf_parity.py` → `reports/gguf-export-parity.json`: 91M 체크포인트(/tmp 스냅샷, 30,500 step)를 F32·Q8_0으로 내보내 llama-server(CPU, `temperature=0`, `return_tokens`)로 한국어 프롬프트 3건(평가셋 A-001 포함) 32토큰 greedy를 돌린 결과 **F32·Q8_0 모두 토큰 id 96/96이 PyTorch와 같았다**. 크기 F32 348.4MiB / Q8_0 92.9MiB, 서버 RSS 389 / 248MiB, 생성 63–131 / 487 tok/s(4스레드, 사전학습 동시 부하). 남은 일: `runs/<run>/trid.q8_0.gguf`와 sha256을 `training.json`에 기록, 310문항 E1·E2.
2. **`constrained.py` 마스크 캐시** (S–M, 1일). 접두 상태별 허용 토큰 비트마스크, E5로 바이트 동일성 고정. llama.cpp 전환과 무관하게 PyTorch 경로에서도 토큰당 3→<1ms.
3. **llama.cpp 백엔드 생성 경로** (M, 1–2일). `trid/runtime.py`에 `backend="torch"|"llamacpp"`, llama-cpp-python(MIT) logits processor로 2번의 제약기 호출, `evaluate_rag_eval.py --backend`로 E4. 폐쇄망 설치물은 llama.cpp 바이너리 + GGUF + Python 패키지 3개.
4. **다음 10억 토큰 실행 구조 파일럿 E6** (S, MPS 9시간). `kv_heads=4`는 config 한 줄. 24×576은 config만으로 되지만 MPS 처리량을 먼저 본다.
5. **WSD stable 구간 체크포인트 평균** (S, 반나절). `scripts/average_checkpoints.py`, 평균본을 E2와 같은 방식으로 310문항 비교.
6. **16k 어휘 파일럿**은 OSS 조사 행동 1에 이미 있음 — 이 문서의 크기·속도 수치(§7)를 결정 기준에 추가.
7. **Q4_K_M + imatrix** (S)는 RAM 256MB 미만 대상이 나타날 때만, E3 통과 조건.
8. **53.5M 자기 증류** (M–L)는 Raspberry Pi급 대상이 확정될 때만, E7.

---

## 부록 A. 측정 조건과 재현

- 기기: Apple M5(성능 4·효율 6코어), 메모리 32GiB, macOS 25.2. 측정 내내 `runs/trid-korean-medium-base-v1` 사전학습(MPS)과 수집기 2개가 실행 중이었다. 절대값은 README·`KOREAN_MODEL_OPTIMIZATION.md`의 한가한 상태 측정과 직접 비교하지 않는다.
- 체크포인트: `checkpoints/step-00033250`(valid loss 3.353)을 `/tmp/trid-ckpt`로 1회 복사해 사용. 실행 디렉터리는 다시 읽지 않았다.
- PyTorch: 프로젝트 venv 2.14.1, `torch.set_num_threads(4)`, `torch.backends.quantized.engine="qnnpack"`(동적 양자화 때만), torchao 0.18.0·gguf 0.19.0·cmake·ninja는 `uv pip install --target /tmp/extra-site`로 venv 밖에 설치.
- llama.cpp: `git clone --depth 1` commit 8ca4060(2026-10-28), `cmake -DGGML_METAL=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release`, `llama-quantize`·`llama-bench -t 4 -p 300 -p 1000 -n 32 -r 10`·`llama-perplexity -c 1024 --chunks 12 -t 4`·`llama-completion --temp 0 -n 24`. ollama 0.31.1은 `Modelfile`에 `num_gpu 0`, `num_thread 4`, `temperature 0`.
- 프롬프트: 평가셋 프롬프트를 이어 붙여 `<|bos|>` 포함 300토큰으로 자름. PPL 텍스트는 평가셋 프롬프트 120건(25,304글자). 둘 다 학습 자료가 아니다.
- 제약 디코딩 측정: 평가셋 40건, `torch.randn` 점수, `max_tokens=64`, 320회.
- 변환·측정 스크립트(`/tmp/trid-bench/bench_cpu.py`, `bench_torchao.py`, `to_gguf.py`)는 저장소에 넣지 않았다. `to_gguf.py`의 매핑(1.4절 표)은 `scripts/export_gguf.py`로 옮겼고, 동일성 측정은 `scripts/gguf_parity.py`가 재현한다(llama.cpp는 `-DGGML_METAL=OFF -DLLAMA_USE_PREBUILT_UI=OFF`로 빌드한 `llama-server`).

## 부록 B. 1차 출처

- MobileLLM https://arxiv.org/abs/2402.14905 (코드 FAIR NC) ; MobileLLM-125M config https://raw.githubusercontent.com/facebookresearch/MobileLLM/main/configs/125M/config.json
- Gemma 3 https://arxiv.org/abs/2503.19786 ; DeepSeek-V2(MLA) https://arxiv.org/abs/2405.04434 ; SmolLM3 https://huggingface.co/blog/smollm3 ; PaLM https://arxiv.org/abs/2204.02311 ; GLU variants https://arxiv.org/abs/2002.05202 ; ReLU Strikes Back https://arxiv.org/abs/2310.04564
- GPTQ https://arxiv.org/abs/2210.17323 ; Scaling Laws for Precision https://arxiv.org/abs/2411.04330 ; torchao https://github.com/pytorch/ao (BSD-3) 및 https://docs.pytorch.org/ao/stable/workflows/inference.html ; GPTQModel https://github.com/ModelCloud/GPTQModel (Apache-2.0) ; llm-awq https://github.com/mit-han-lab/llm-awq (MIT) ; hqq https://github.com/mobiusml/hqq (Apache-2.0) ; SmoothQuant https://github.com/mit-han-lab/smoothquant (MIT)
- Minitron https://arxiv.org/abs/2407.14679 ; Sheared-LLaMA https://arxiv.org/abs/2310.06694 (https://github.com/princeton-nlp/LLM-Shearing MIT) ; LAWA https://arxiv.org/abs/2306.03241
- llama.cpp https://github.com/ggml-org/llama.cpp (MIT), HOWTO-add-model https://github.com/ggml-org/llama.cpp/blob/master/docs/development/HOWTO-add-model.md ; ONNX Runtime https://github.com/microsoft/onnxruntime (MIT) ; OpenVINO https://github.com/openvinotoolkit/openvino (Apache-2.0) ; ExecuTorch https://github.com/pytorch/executorch (BSD)
- LayerSkip https://arxiv.org/abs/2404.16710 (코드 CC BY-NC 4.0) ; KIVI https://github.com/jy-yuan/KIVI (MIT) ; XGrammar https://arxiv.org/abs/2411.15100 (https://github.com/mlc-ai/xgrammar Apache-2.0)
- 커뮤니티 속도 참고 https://metrillm.dev/models/smollm2-135m
