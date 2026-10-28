# ~100M 모델의 토큰당·초당 품질을 올리는 학습 기법 조사 (trid 91M, Mac 1대)

작성일: 2026-10-28. 대상: `trid` 91M(12층 × 768, SwiGLU 2,048, 문맥 1,024, 자체 8k byte-level BPE, 임베딩 공유), 다음 약 10억 토큰 사전학습을 Mac 1대(MPS, bf16 autocast + `torch.compile` 측정 2.2k tok/s)로 돌리는 경우. 시스템 구조(폐쇄망 배포·검색)는 `reports/closed-network-architecture-survey.md`가 다루므로 여기서는 **학습 루프와 모델 코드 안에서 바꿀 수 있는 것**만 다룬다. 토크나이저·데이터 필터·평가는 `reports/korean-oss-opportunities.md`에 있다.

표기
- 라이선스는 GitHub API/LICENSE 파일에서 2026-10-28에 직접 확인한 값.
- **(측정)**: 이 조사에서 프로젝트 venv(PyTorch 2.14.1, MPS)로 직접 잰 값. 측정 시각에 91M 사전학습·SFT·수집기 2개가 같이 돌고 있었으므로 절대값은 낮고 상대 비교만 유효하다.
- **(계산)**: 논문의 공식에 trid 숫자를 넣어 계산한 값.
- 판정: **적용**(다음 실행에 넣는다) · **실험**(1일 ablation 후 결정) · **보류**(가치 있으나 지금 아님) · **부적합**(MPS·규모·정책상 불가).
- 난이도: S(반나절, 한 함수) · M(1–2일, 모델/체크포인트 형식 변경) · L(3일+).
- 추정은 `[INFERENCE]`.

---

## 0. 요약

| 영역 | 결론 | 판정 |
|---|---|---|
| 스텝당 토큰·lr·warmup | 현행 2,048 토큰/스텝·warmup 100스텝은 모든 공식(DeepSeek·Porian·Zhang·Power Lines)이 주는 최적 배치 18만–39만 토큰의 1/100이고, warmup도 20만 토큰으로 공개 레시피(토큰 수 ≈ 파라미터 수, 즉 약 9천만 토큰)의 1/400이다(배치를 262k로 올려도 100스텝 = 2,600만 토큰, 1/3.5). **스텝당 262,144 토큰(accumulate 128), lr 2e-3 근처, warmup ≈ 90M 토큰(≈ 350스텝), β₂ 0.98–0.999**가 네 공식과 두 벤치마크의 교집합이다. | 적용 |
| 스케줄 | WSD(상수 + 10–20% 선형/1-sqrt 디케이)가 cosine과 같거나 낫고(Hägele 2024 124M–1B), 최종 lr은 0이 10%보다 낫다(Semenov 2025 Takeaway 8). 이미 `schedule="wsd"`로 구현돼 있음. | 적용 |
| 옵티마이저 | 91M·10억 토큰에서는 행렬 기반(Muon·SOAP)이 잘 튜닝된 AdamW보다 1.3–1.4× 토큰 효율(Wen et al. 2025, 130M). 단 배치가 작을수록(≤131k 토큰) AdEMAMix·MARS 같은 분산 감소형이 Muon과 비슷하거나 앞선다(Semenov et al. 2025, 124M). `torch.optim.Muon`이 PyTorch 2.9+에 들어 있고 MPS에서 돈다(측정). | 실험(Muon → AdEMAMix 순) |
| modded-nanogpt 기법 | MIT. 기록 92개 중 MPS에서 못 쓰는 것은 FlexAttention(역전파 미지원, 측정)과 FP8(dtype 미정의, 측정) 둘뿐. QK-norm·zero-init·logit softcap·value embedding·U-net skip·BOS 정렬·batch size 스케줄·EMA는 모두 순수 PyTorch라 옮길 수 있다. 가장 싼 것은 BOS 정렬 샘플링(기록 26, −1.1% 시간)과 zero-init(기록 5 묶음). | 적용(BOS·zero-init), 실험(softcap·VE·U-net) |
| 깊이 vs 폭 | 동일 파라미터에서 MobileLLM 125M(30층×512~576)이 12층×768보다 +0.9–1.5pt(Table 11); 한국어 소형 모델도 모두 깊다(EXAONE 4.0 1.2B 30층, HyperCLOVA X SEED 0.5B 24층, kanana nano 32층). trid 24층×576(89.7M, 측정)이 1일 ablation 후보. | 실험 |
| 데이터 효율 | 10억 토큰은 수집 추정치(법령 11억 + 보도 5.3억 + 위키 3.1억)의 1 epoch 미만이므로 반복은 필요 없고, 모자라면 4 epoch까지는 손실 차이가 없다(Muennighoff 2023). 혼합 비율 최적화(DoReMi·RegMix)는 3개 도메인뿐이라 17M 모델로 직접 격자 탐색이 더 싸다. SFT는 현재 패딩이므로 길이 정렬 배치가 바로 효과. | 적용(길이 정렬), 보류(RegMix) |
| μP | 10억 토큰 1회 실행이면 60M 토큰 lr 파일럿 3개(합계 1일)가 μP 구현(M)보다 싸다. 폭·깊이를 함께 바꿀 계획이 생기면 CompleteP(MIT 참조 구현)로 전환. | 보류 |

§9에 다음 실행 설정과 1일 ablation 3개, §10에 우선순위 행동 목록이 있다.

---

## 1. modded-nanogpt 속도 기록과 MPS 적용 가능성

저장소 [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) (MIT, LICENSE 확인). 과제: 8×H100으로 FineWeb 검증 손실 3.28 도달 시간(llm.c GPT-2 124M 재현 45분 → 현재 0.665분, 토큰 10B → 3.3억 미만). 각 기록은 **동일 손실 목표를 더 적은 시간/토큰에 도달**한 것이므로 "손실 이득"은 토큰 효율로 읽어야 한다. 아래는 요청된 기법별 기록(README world-record 표, 2026-10-28 조회).

| # | 기록 시간 | 기법 | 이전 대비 | MPS 가능? (측정 근거) | trid 적용 | 난이도 | 손대는 파일 |
|---|---|---|---|---|---|---|---|
| 2 | 45 → 31.4분 | lr 튜닝 + RoPE | −30% | 가능 | 이미 RoPE. lr은 §3 | — | — |
| 3–4 | 31.4 → 22.3분 | **Muon** (+개선) | −29% | 가능. `torch.optim.Muon` 60개 행렬 1스텝 441ms vs AdamW 150ms (측정, 부하 중). 262k 토큰/스텝(≈120초)이면 0.3% 오버헤드 | 실험 (§2) | M | `trid/pretraining.py` 205행 옵티마이저 생성, `save_checkpoint`/`restore_checkpoint`(상태 키가 `exp_avg*`로 고정됨), `trid/model.py` `Attention.qkv` 분리 |
| 5 | 22.3 → 15.2분 | 임베딩 패딩(50257→50304), **ReLU²**, **projection zero-init**, **QK-norm** | −32% (묶음) | 가능 | QK-norm은 `Config.qk_norm` 옵션으로 이미 있음(기본 off). zero-init은 `Trid.__init__` 148–150행의 `attn.output`·`down` 초기화를 0으로 바꾸면 됨. 어휘 8,192는 이미 128의 배수. ReLU²는 SwiGLU 대체이므로 보류 | 적용(QK-norm on, zero-init) | S | `trid/model.py` |
| 8 | 12.0 → 10.8분 | **untied embedding/head** | −10% | 가능 | +6.3M 파라미터(측정: 91.25M → 97.54M). 이후 기록 51에서 **다시 tied**로 돌아갔고 기록 53은 "학습 2/3 지점에서 untie". MobileLLM·kanana(Table 17: tied 44.88 vs untied 44.45)는 tied가 같거나 낫다 | 보류(기록 53 방식은 2차 실험) | M(체크포인트 키 변경) | `trid/model.py` 169행 `F.linear(hidden, self.embedding.weight)` |
| 9 | 10.8 → 8.2분 | **value embedding(초기형)·embedding skip, Muon momentum warmup(0.85→0.95), logit softcap tanh(30)** | −24% (묶음) | 가능 | softcap: `logits = c·tanh(logits/c)` 한 줄. 제한 디코딩(`trid/constrained.py`)은 마스크만 쓰므로 단조 변환에 영향 없음 | 실험(softcap 15) | S | `trid/model.py` forward, `trid/training.py` `loss_for`(동일 경로) |
| 10 | 8.2 → 7.8분 | bf16 activations | −5% | 가능 (이미 `precision="bf16"`) | 적용됨 | — | — |
| 11 | 7.8 → 7.2분 | **U-net 패턴 skip** (인코더 블록 출력을 디코더 블록 입력에 학습 가중치로 더함) + lr 2배 | −8% | 가능 | 12층이면 앞 6층 → 뒤 6층 skip, 가중치 6개 추가. 체크포인트에 새 텐서 | 실험 | M | `trid/model.py` `Trid.forward` 161–164행 |
| 12 | 7.2 → 5.03분 | 1024 dense causal → **64K 문맥 FlexAttention(문서 마스크 + 슬라이딩 윈도)** | −30% | **불가.** `torch.compile(flex_attention)` 역전파가 MPS에서 `NotImplementedError: FlexAttention does not support backward on MPS` (측정). 대체인 SDPA bool 마스크는 `is_causal`보다 5배 느림(9.9 vs 2.0 ms, 2×12×1024×64, 측정) | 부적합 | — | — |
| 13 | 5.03 → 4.66분 | **attention window warmup** (윈도를 학습 중 키움) | −7% | 윈도 자체가 불가. 유사한 "max_seq_len 스케줄"(기록 72)·시퀀스 길이 warmup(Li et al. 2021, §5)은 가능 | 실험(§5) | S | `trid/pretraining.py` `sample_batch` |
| 14–17 | 4.66 → 3.57분 | **value embeddings** (어휘×d 테이블을 attention V에 더함; 기록 15 U-net 배치, 16 분할, 17 희소화·attn 층 1개 제거) | −23% (누적) | 가능 | 8,192×768 테이블 1개 = 6.3M 파라미터. 기록 17처럼 일부 층에만 공유하면 2개(12.6M). 91M에 14% 추가 | 실험(2차) | M | `trid/model.py` `Attention.forward` 92행 v에 가산, `Trid.__init__` |
| 18 | 3.57 → 3.4분 | softcap 30 → 15 | −5% | 가능 | 위 9번과 함께 15로 | 실험 | S | `trid/model.py` |
| 19 | 3.4 → 3.142분 | **FP8 head**, offset logits, lr 디케이 끝값 0.1 | −8% | **FP8 불가.** `tensor.to(torch.float8_e4m3fn)`이 MPS에서 `Undefined type Float8_e4m3fn` (측정). lr 끝값은 §4 | 부적합(FP8) | — | — |
| 20–21 | 2.99 → 2.93분 | QKV 병합, long-short attention, attention scale, Adam ε 하향, 배치 축소 | −2% | 부분 | trid는 이미 QKV 병합(`qkv`). Muon 쓸 때는 **Q·K·V를 따로 직교화**해야 하므로(Muon 블로그 "Empirical considerations") 분리 필요 | — | — | — |
| 26 | 2.896 → 2.863분 | **학습 배치 시작을 EOS/BOS에 정렬**, 디케이 비율 0.45 | −1.1% | 가능 | 현행 `sample_batch`는 임의 오프셋. `<\|bos\|>` 위치 배열을 만들어 거기서만 시작 | 적용 | S | `trid/pretraining.py` 52–58행 |
| 37 | 2.495 → 2.483분 | bf16 cross-entropy | −0.5% | autocast가 CE를 fp32로 올림 | 보류 | — | — |
| 38, 41, 43 | → 2.284분 | Polar Express(NS 대체), NorMuon([arXiv 2510.05491](https://arxiv.org/abs/2510.05491)), Cautious WD | 각 −0.3~−1.3% | 가능 | Muon 채택 후 2차 | 보류 | S | — |
| 46 | 2.248 → 2.203분 | **batch size 스케줄**(작게 시작해 키움) | −2% | 가능 | `accumulate`를 스텝 함수로. 임계 배치가 데이터·손실에 따라 커진다는 Power Lines(§3)와 일치 | 실험 | S | `trid/pretraining.py` 200행 `step_tokens`, `tokens_seen` 집계 |
| 53 | 2.037 → 1.988분 | multi-token prediction, 2/3 지점 untie | −2% | 가능 | 복사 과제라 MTP 효과 불확실 `[INFERENCE]` | 보류 | M | — |
| 92 | 1.126 → 0.665분 | sampled softmax, 전체 FP8 MLP, 84.6M행 해시 n-gram 임베딩(65B 파라미터), ANVIL 옵티마이저, **tail EMA**, CUDA graph | −41% | EMA·해시 임베딩만 가능 | tail EMA는 Hägele SWA와 같은 효과(§4). 해시 n-gram 임베딩은 CPU 메모리상 부적합 | 실험(EMA) | S | `trid/pretraining.py` 학습 루프 |

요약: **MPS에서 못 쓰는 것은 FlexAttention(기록 12·13·29)과 FP8(기록 19·84·89·90)뿐**이고, 나머지는 모델 코드 수준이다. 단 속도 기록은 524k 토큰 배치·8 GPU·3.28 손실이라는 짧은 학습(≈3억 토큰, 124M)에서 잰 것이므로, trid의 10억 토큰(11.8 토큰/파라미터)보다 더 "짧은 학습" 쪽이다. Wen et al.(§2)에 따르면 이 영역에서는 Muon의 이득이 가장 크다.

---

## 2. AdamW 이후의 옵티마이저 — 소형 모델 증거

두 개의 독립 벤치마크가 2025-09에 동시에 나왔고, 결론이 **배치 크기**에 따라 갈린다.

- **Wen, Hall, Ma, Liang, "Fantastic Pretraining Optimizers and Where to Find Them"** ([arXiv 2509.02046](https://arxiv.org/abs/2509.02046), CC BY 4.0; 코드 [marin-community/marin](https://github.com/marin-community/marin/tree/kaiyue/optimizers)). Llama 130M(32층×512)–1.2B, 배치 ≥0.4M 토큰, 11개 옵티마이저를 좌표 하강으로 튜닝. **130M에서 행렬 기반(Muon·SOAP·Kron·Scion)이 AdamW 대비 1.3–1.4×, 스칼라 기반(NAdamW·Mars·Cautious·Lion·Adam-mini)은 ≤1.2×.** 1.2B에서는 1.1×로 줄어든다. 1–4× Chinchilla에서는 Muon 1위, 8× 이상에서는 SOAP·Kron. "AdamW baseline의 lr 하나만 튜닝해도 GPT-3 레시피 대비 2× 빨라진다"(Fig. 1). 중간 체크포인트 비교는 디케이 중 순위가 뒤집히므로 **학습 끝에서 비교**해야 한다.
- **Semenov, Pagliardini, Jaggi, "Benchmarking Optimizers for LLM Pretraining"** ([arXiv 2509.01440](https://arxiv.org/abs/2509.01440), CC BY-SA 4.0). Llama 124M(tied, SwiGLU, RMSNorm, RoPE)–720M, **배치 32×512 = 16k 토큰("small")과 256×512 = 131k 토큰("large")**, 1B–16.8B 토큰. Takeaway 1: **AdEMAMix가 두 배치 모두에서 최상위**, 부호 기반(Lion·Signum)·MARS는 큰 배치에서만 좋음, Sophia는 작은 배치에서 발산, SOAP는 두 설정에서 일관. Takeaway 6: 스케줄은 대부분 cosine이 최선이나 **Muon은 WSD가 최선**. Takeaway 8: 최종 lr을 최대의 10% 아래로 더 내리면 유의하게 좋아진다. Takeaway 9: z-loss는 124M–583M에서 손실도 스파이크도 개선하지 않았다. 124M "large" 설정의 튜닝 결과(Table 7 외): AdamW lr 1e-3·warmup 2,000·wd 0.1·clip 0.5·**β = (0.8, 0.999)**; Muon(hidden) 1e-2 + AdamW 1e-3; AdEMAMix 1e-3; Lion 1e-3·wd 0.1(small batch에서는 wd 0.5); SOAP 2e-3. Wen et al.는 두 논문의 차이를 "배치 0.1M/0.02M vs ≥0.4M"으로 설명한다(§2 "Comparison with concurrent work").
- **modded-nanogpt 2024-10-29 옵티마이저 비교** ([records/track_1_short/2024-10-29_Optimizers](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2024-10-29_Optimizers), MIT). 124M·524k 배치·WSD·wd 0: Muon(lr 0.02, momentum 0.95) > SOAP ≈ DistributedShampoo > Adam(lr 1.8e-3) 순의 토큰 효율; Sophia·Lion·AdamWScheduleFree는 Adam 대비 이득 없음, AdEMAMix는 "Adam보다 조금 나음".

trid의 다음 실행은 배치 131k–262k 토큰이므로 Semenov의 "large"와 Wen의 하한 사이다. 따라서 **Muon과 AdEMAMix 둘 다 후보**이고, 1일 ablation으로 정해야 한다.

| 옵티마이저 | 구현 (라이선스) | 소형 모델 핵심 수치 | trid 판정 · 이유 | 난이도 | 손대는 파일 |
|---|---|---|---|---|---|
| **Muon** | [`torch.optim.Muon`](https://docs.pytorch.org/docs/main/generated/torch.optim.Muon.html) (PyTorch ≥2.9, BSD-3; venv 2.14.1에 있음, 측정) · [KellerJordan/Muon](https://github.com/KellerJordan/Muon) (MIT) · [MoonshotAI/Moonlight](https://github.com/MoonshotAI/Moonlight) (MIT, wd + RMS 매칭 `adjust_lr_fn="match_rms_adamw"`) | Wen: 130M 1.3–1.4×(1–4× Chinchilla에서 1위). Semenov: D-Muon(전 파라미터 wd) ≫ Muon(wd 없음), 모든 설정에서 일관. Moonlight([arXiv 2502.16982](https://arxiv.org/abs/2502.16982)): "wd 추가 + 업데이트 스케일 조정"으로 대규모 안정, ~2× (자체 baseline 기준). speedrun: 1.35× (기록 3) | **실험 1순위.** MPS 동작·오버헤드 확인(측정). 2D hidden 가중치(qkv·output·gate·up·down = 60개)에 Muon lr ≈ 0.02(speedrun)·0.01(Semenov), 임베딩·RMSNorm은 AdamW. `qkv`는 Q/K/V 분리 직교화가 더 좋다는 결과(Muon 블로그, Boza)가 있어 `Attention.qkv`를 세 `nn.Linear`로 나누거나 Muon 전용 view가 필요. 두 옵티마이저 상태를 `optimizer.safetensors`에 저장하도록 `save_checkpoint`(68–73행)·`restore_checkpoint`(91–95행, 키 `exp_avg*` 고정)를 일반화 | M | `trid/pretraining.py`, `trid/model.py` |
| **AdEMAMix** | [apple/ml-ademamix](https://github.com/apple/ml-ademamix) (MIT) 단일 파일; [arXiv 2409.03137](https://arxiv.org/abs/2409.03137) | 1.3B·101B 토큰 ≈ AdamW 197B 토큰(+95%). Semenov 124M: 16k·131k 배치 모두 1위, 720M·1M 배치도 1위. speedrun: "Adam보다 조금 나음" | **실험 2순위.** 느린 EMA(β₃=0.9999, α=8, warmup 스케줄)가 긴 학습일수록 유리하므로 10억 토큰 7,600스텝(131k 기준)에서 β₃ warmup = 전체 스텝으로. AdamW 인터페이스와 같아 S | S | `trid/pretraining.py` 205행 |
| **SOAP** | [nikhilvyas/SOAP](https://github.com/nikhilvyas/SOAP) (MIT); [arXiv 2409.11321](https://arxiv.org/abs/2409.11321) | 360M/660M 큰 배치에서 AdamW 대비 반복 −40%·벽시계 −35%. Wen: 130M 1.3–1.4×, 8× Chinchilla 이상에서 1위. Semenov: 배치 무관하게 일관, 단 **벽시계가 가장 느림**(Fig. 18) | 실험 3순위. 고유분해(`torch.linalg.eigh`)가 MPS에서 CPU로 떨어질 가능성 `[INFERENCE]`; 768²·2048² 행렬 10스텝마다면 감당 가능. Muon이 더 싸고 비슷하므로 Muon이 실패할 때만 | M | `trid/pretraining.py` |
| Distributed Shampoo | [facebookresearch/optimizers](https://github.com/facebookresearch/optimizers) (BSD) | speedrun 비교에서 Muon·SOAP와 비슷한 토큰 효율, 더 큰 오버헤드. DDP 전제 설계 | 부적합(복잡도; Muon이 "accumulation 없는 Shampoo"와 동치, Bernstein & Newhouse) | L | — |
| **Schedule-Free AdamW** | [facebookresearch/schedule_free](https://github.com/facebookresearch/schedule_free) (Apache-2.0); [arXiv 2405.15682](https://arxiv.org/abs/2405.15682) | AlgoPerf 2024 self-tuning 우승. Hägele 2024 §4.2(210M): β 설정에 민감하고 "둘 다 cooldown 스케줄에 비슷하거나 뒤짐". Semenov Takeaway 5: 긴 warmup·큰 lr 필요. speedrun: 이득 없음 | 보류. "총 스텝을 미리 정하지 않아도 된다"는 장점은 WSD가 이미 준다 | S | — |
| Lion | [lucidrains/lion-pytorch](https://github.com/lucidrains/lion-pytorch) (MIT); [arXiv 2302.06675](https://arxiv.org/abs/2302.06675) | LM에서 "Adam과 비슷하거나 조금 나음", 이득은 배치가 클수록. Wen: AdamW 대비 ≤5% 느림, 모델이 커질수록 벌어짐, wd 0.6 필요. Semenov: 큰 배치 필요, lr 2e-3에서 발산 | 부적합(91M에서 메모리 절약 의미 없음, 토큰 효율 열세) | S | — |
| MARS | [AGI-Arena/MARS](https://github.com/AGI-Arena/MARS) (Apache-2.0); [arXiv 2411.10438](https://arxiv.org/abs/2411.10438) | GPT-2 small/medium/large에서 AdamW "큰 폭" 개선(자체 baseline). Wen: 1.1–1.2×. Semenov: 131k 이상 배치에서만 좋고, 720M·1M 배치에서 AdEMAMix와 공동 1위, lr에 둔감 | 보류(AdEMAMix와 같은 계열, 배치 의존성 더 큼) | S | — |
| NorMuon / Polar Express / Cautious WD | modded-nanogpt 기록 38·41·43 (MIT) | 각 −0.3~−1.3% 시간 | Muon 채택 후 2차 | S | — |

가중치 감쇠 적용 대상: 현행 `AdamW(model.parameters(), weight_decay=.1)`은 RMSNorm 가중치·임베딩에도 0.1을 건다. OLMo 2([arXiv 2501.00656](https://arxiv.org/abs/2501.00656) §3.4.2)는 임베딩을 제외, llm.c는 2D만, Moonlight·D-Muon은 전 파라미터. Power Lines(§3)는 wd가 TPP의 함수라고 본다. → 파라미터 그룹을 나누는 것은 어느 옵티마이저를 쓰든 필요한 S 작업(`trid/pretraining.py` 205행).

---

## 3. 배치 크기·학습률·warmup — 91M @ 10억 토큰 계산

네 논문의 공식에 trid 숫자(비임베딩 84.95M, 전체 91.25M, D = 10⁹)를 넣었다(계산). 단위 변환은 각 논문의 시퀀스 길이를 따랐다.

| 출처 (라이선스) | 공식 | 91M·1B 계산 | 비고 |
|---|---|---|---|
| **DeepSeek LLM** [arXiv 2401.02954](https://arxiv.org/abs/2401.02954) §3.1 | $B_{opt}=0.2920\,C^{0.3271}$, $\eta_{opt}=0.3118\,C^{-0.1250}$ (C = M·D, M = 72nd² + 12ndl) | C = 6.2×10¹⁷ → **B = 193k 토큰, η = 1.9e-3** | kanana가 이 법칙과 multi-step 스케줄을 그대로 씀(§7) |
| **Porian et al.** [arXiv 2406.19146](https://arxiv.org/abs/2406.19146) Fig. 3·Table 4 | $BS = 160\,(N/108\mathrm{M})^{2/3}$ seq(2,048), $LR = 0.0047\,(N/108\mathrm{M})^{-1/3}$, **β₂ 0.99**(배치 ≤128 seq), 독립 wd 1e-4, z-loss 1e-4, warmup = min(N 토큰, 20%) | **B = 280–293k 토큰, lr 5e-3, β₂ 0.99, warmup ≈ 85M 토큰** | "임계 배치 아래는 전부 괜찮다"는 통념과 달리 **최적 배치 아래에서 손실이 나빠진다**고 보고. 작은 배치에서는 β₂ 0.95가 해롭다 |
| **Zhang et al.** [arXiv 2410.21676](https://arxiv.org/abs/2410.21676) Table 7 | $B^*=93.20\,N^{0.47}$ (N: M, 512-토큰 seq) / $B^*=22.91\,D^{0.47}$ | **385k 토큰(N식) / 302k 토큰(D식, D를 M 단위로 읽음)** | CBS는 모델 크기보다 데이터 양에 따라 커짐; D식은 논문 표기 단위가 모호해 `[INFERENCE: 85M·1.7B 토큰에서 log₂B*=9.54와 맞으려면 D는 M 단위]` |
| **Power Lines (Cerebras)** [arXiv 2505.13738](https://arxiv.org/abs/2505.13738) Fig. 1·Table 5 | $B_{opt}=0.0306\,D^{0.383}$ seq(2,048), $B_{crit}=0.0471\,D_{min}^{0.462}$; AdamW 시간척도 $\tau_{opt}=1.084\,TPP^{-0.527}$, $\lambda = B/(\eta D \tau)$ | **B_opt = 175k 토큰, B_crit = 1.39M 토큰**; TPP 11.8 → τ 0.30 → **λ_opt ≈ 0.44 (B 131k, η 1e-3) / 0.22 (η 2e-3)** | wd 0.1은 이 짧은 학습에서 과소(τ=1.3). μP 전제. Semenov Takeaway 3은 "0.5 이상이 짧은 학습에서 효과, 긴 학습은 0.1이 안전" |
| MiniCPM [arXiv 2404.06395](https://arxiv.org/abs/2404.06395) §3 | $bs = 1.21\times10^9 / L^{6.24}$ (C4 손실 기준) | 토크나이저가 달라 손실 비교 불가 → 생략 | μP(scale_emb 12, init 0.1, scale_depth 1.4, lr 0.01), WSD 디케이 10% |
| 공개 124M–135M 레시피 | llm.c 524k·6e-4 / SmolLM2-135M 1M·3e-3 / Cerebras-GPT-111M 246k·6e-4(SP) 6e-3(μP) / Pythia-160M 2M·6e-4 / MobileLLM-125M 2.1M·2e-3 / Semenov 124M 131k·1e-3 | — | lr은 파라미터화·wd·β₂에 따라 2–5배 흔들리므로 **배치는 공식으로, lr은 파일럿으로** |

결론(적용):
- **스텝당 262,144 토큰(micro 2 × accumulate 128)**: 네 공식의 B_opt(175k–390k) 안에 있고 B_crit(1.39M)보다 한참 아래라 토큰 효율 손실이 없다. 10억 토큰 = 3,815스텝. 131k(accumulate 64)도 허용 범위 하한이지만 Porian의 "최적 아래에서 손실 악화" 때문에 262k를 권한다. 처리량은 micro-batch로 정해지므로 accumulate는 공짜다(PRETRAINING.md 테스트 확인).
- **lr 파일럿 {1e-3, 2e-3, 4e-3}** (DeepSeek 1.9e-3, Porian 5e-3, Semenov 1e-3을 포괄). β₂ 0.98(Wen)–0.99(Porian). 현행 (0.9, 0.95)는 Porian이 작은 배치에서 해롭다고 한 값.
- **warmup ≈ 90M 토큰 ≈ 350스텝** (Porian "N 토큰", Cerebras-GPT 375M 토큰, Semenov 2,000스텝×131k = 262M). 현행 `min(100, steps//10)`(`lr_ratio` 34행)은 262k 배치에서 26M 토큰이라 3.5배 짧다 → warmup을 토큰 수로 지정하는 인자 추가(S, `trid/pretraining.py` `lr_ratio`).
- **wd**: 0.1 기본, 0.3을 선택 변수로(Power Lines 0.2–0.4, Semenov 0.5). 임베딩·RMSNorm 제외.

## 4. 스케줄: WSD vs cosine, 최종 lr, 가중치 평균

| 출처 (라이선스) | 핵심 수치 | trid | 난이도 | 파일 |
|---|---|---|---|---|
| **Hägele et al. 2024** [arXiv 2405.18392](https://arxiv.org/abs/2405.18392) (CC BY-SA; 코드 [epfml/schedules-and-scaling](https://github.com/epfml/schedules-and-scaling) MIT) | 124M–1B·8B: 상수 lr + 20% 선형 cooldown = 잘 튜닝된 cosine과 "거의 완벽히 일치", lr 민감도는 더 낮음. **cooldown 10–20%에서 cosine을 넘어서고**, 긴 학습에서는 절대 스텝만 충분하면 더 짧아도 됨(200k 스텝 중 10k). **(1−√) 형태**가 선형보다 일관되게 낫고 길수록 차이 커짐. cosine 끝값은 10%보다 0에 가까울수록 좋음(Fig. 22). **SWA**는 디케이 없이도 중간 체크포인트 품질을 올리지만 cooldown에는 못 미침 | `schedule="wsd"`는 선형 → 0이므로 이미 부합. (1−√) 옵션 추가는 S. **EMA/SWA를 stable 구간 평가용으로 유지**하면 lr 파일럿·ablation의 중간 비교가 안정됨(speedrun 기록 92 "tail EMA"와 같은 발상) | S | `trid/pretraining.py` `lr_ratio` 39–40행, 학습 루프 |
| MiniCPM §4.2 | WSD 디케이 10%면 충분(지수형), 디케이 구간에 고품질/SFT 데이터 혼입이 SFT만보다 효과적 | 이미 `decay_fraction` 인자. 10억 토큰이면 10%=100M 토큰(≈380스텝)은 Hägele의 "절대 스텝 충분" 조건에 걸릴 수 있으므로 **20%** 권장 | — | — |
| Semenov Takeaway 6·8 | Muon은 WSD가 최선, 다른 것은 cosine; 최종 lr 0.01·γ_max가 0.1·γ_max보다 유의하게 좋음 | cosine 경로(38행)의 바닥 0.1을 0.01 또는 0으로 | S | `lr_ratio` |
| DeepSeek multi-step (80%에서 31.6%, 90%에서 10%) | cosine과 최종 성능 동일, 첫 구간 재사용 가능 | WSD와 같은 목적 → 추가 불필요 | — | — |

## 5. 데이터 효율

| 기법 | 출처 (라이선스) | 핵심 수치 | trid 판정 | 난이도 | 파일 |
|---|---|---|---|---|---|
| **다중 epoch** | Muennighoff et al. 2023 [arXiv 2305.16264](https://arxiv.org/abs/2305.16264) (코드 [huggingface/datablations](https://github.com/huggingface/datablations) Apache-2.0) | 고정 계산량에서 **4 epoch까지는 유일 데이터와 손실 차이 무시 가능**, 그 이상은 추가 계산의 가치가 0으로 감소. TinyLlama(3T = 약 3 epoch, Apache-2.0)가 실전 예 | 수집 추정 19억 토큰(법령 11억·보도 5.3억·위키 3.1억) 중 10억이면 1 epoch 미만. 수집이 늦으면 **있는 만큼 2–4회 반복해도 됨** → 10억 토큰 일정을 수집 완료에 묶지 않아도 된다. 단 `sample_batch`는 임의 구간 샘플링이라 epoch 개념이 없음(아래 BOS 정렬과 함께 문서 단위 순회로 바꾸면 반복 횟수가 명시됨) | 적용(정책) | S | `trid/pretraining.py` `sample_batch` |
| **데이터 혼합 비율** | DoReMi [arXiv 2305.10429](https://arxiv.org/abs/2305.10429) (280M proxy → 8B, 2.6× 빠름, +6.5pt); RegMix [arXiv 2407.01492](https://arxiv.org/abs/2407.01492) ([sail-sg/regmix](https://github.com/sail-sg/regmix) MIT; 1M-파라미터 모델 512개 × 1B 토큰 회귀, DoReMi 비용의 10%) | 둘 다 도메인 수십 개를 전제. trid는 **3개 도메인**(법령·보도·위키) | 보류. 3개면 회귀 모델이 필요 없고, 17M 모델(이전 세대 구성)로 비율 격자 {법령 0.3–0.6} × {보도 0.2–0.4} 6–9개를 각 20M 토큰씩 돌려 도메인별 held-out BPC로 고르는 것이 RegMix와 같은 절차의 축소판. 목표 분포(평가셋 근거 문서의 도메인 비율)를 손실 가중치로 쓰는 것이 RegMix의 교훈 | S | `scripts/build_pretraining_mix.py` 비율 인자, `scripts/evaluate_bpc.py` |
| **시퀀스 길이 warmup / shortest-first** | Li, Zhang, He 2021 [arXiv 2108.06084](https://arxiv.org/abs/2108.06084) (DeepSpeed, Apache-2.0) | GPT-2/GPT-3 125M 재현에서 긴 시퀀스가 초기 gradient 분산 폭주의 원인; 길이 warmup으로 8× 배치·4–40× lr에서도 안정, 같은 성능에 토큰 2.2×·시간 3.7× 절감(불안정한 baseline 대비). speedrun 기록 72 "max_seq_len 스케줄" −1.3% | 실험(2차). trid는 문맥 1,024·RoPE라 구현은 `sample_batch`의 `length`를 스텝 함수로 두면 끝. 이득은 주로 초기 안정성이므로 lr 4e-3 변형에서 발산할 때 켠다 | S | `trid/pretraining.py` |
| **BOS 정렬·문서 단위 샘플링** | speedrun 기록 26 (MIT) | −1.1% 시간 | 적용. `corpus.py` 190행이 이미 `<\|bos\|>…<\|eos\|>`로 쓰므로 BOS 오프셋 배열만 만들면 됨 | S | `trid/pretraining.py` 52–58행 |
| **Packing vs padding (SFT)** | Krell et al. 2021 [arXiv 2107.02027](https://arxiv.org/abs/2107.02027) (패딩이 토큰의 최대 50–89%, 패킹 2× 속도, 마스크로 수학적 동치); Ding et al. 2024 "Fewer Truncations" [arXiv 2404.10830](https://arxiv.org/abs/2404.10830) (Best-fit packing: 독해 +4.7%, 문맥 따르기 +16.8%, 폐쇄 도메인 환각 −58.3%) | `trid/training.py` 18–26행 `batch()`는 **배치 내 최대 길이로 우측 패딩**. RAG 예제는 근거 문단 길이가 제각각이라 패딩 비율이 큼 `[INFERENCE]` | 적용(1단계: **길이 정렬 버킷 배치** — 패딩만 줄이고 attention은 그대로). 2단계 cross-doc 마스크 패킹은 MPS에서 bool 마스크 SDPA가 5× 느려(측정) 보류 | S | `trid/training.py` `batch`, 샘플러 |
| **손실 마스킹** | Shi et al. 2024 [arXiv 2405.14394](https://arxiv.org/abs/2405.14394): 지시문에도 손실을 주면(Instruction Modelling) **긴 지시·짧은 출력**·적은 예제에서 과적합이 줄어 개선. Huerta-Enochian & Ko 2024 [arXiv 2401.13586](https://arxiv.org/abs/2401.13586): 짧은 답 데이터에서 prompt loss weight **0.01–0.5**가 0이나 1보다 낫고(음의 2차 관계), 긴 생성은 ~1.0 | trid SFT는 정확히 "긴 근거 + 짧은 답(값 [Dn])" 형태이고 `loss_for`는 프롬프트를 −100으로 완전 마스킹. **PLW 0.1 정도의 부분 가중**이 두 논문의 예측과 맞음. 근거 문단 토큰에 작은 손실을 주면 사전학습 분포(공문)도 함께 유지됨 `[INFERENCE]` | 실험(SFT 단계, 1일 미만) | S | `trid/training.py` `loss_for` 28–46행, `trid/text.py` 라벨 생성 |

## 6. μP / depth-μP — 작은 파일럿에서 lr 전이

| 항목 | 출처 (라이선스) | 핵심 수치 | trid |
|---|---|---|---|
| μTransfer | Yang et al. 2022 [arXiv 2203.03466](https://arxiv.org/abs/2203.03466); [microsoft/mup](https://github.com/microsoft/mup) (MIT) | 13M 모델에서 튜닝 → BERT-large 350M을 넘김; 40M → GPT-3 6.7B가 13B 공개 수치를 넘김(튜닝 비용 7%). Cerebras-GPT([arXiv 2304.03208](https://arxiv.org/abs/2304.03208)): 40M에서 튜닝한 lr 6e-3을 111M–2.7B에 그대로, SP 대비 Pile 손실 −0.43%·다운스트림 +1.7% | 폭 전이용. trid는 폭을 바꿀 계획이 없고 lr 후보 3개를 직접 돌리는 비용(각 60M 토큰, §9)이 μP 구현(출력 스케일, 폭 기준 init, 임베딩 lr 분리: M)보다 싸다 |
| Depth-μP | Yang, Yu, Zhu, Hayou 2023 [arXiv 2310.02244](https://arxiv.org/abs/2310.02244) | 블록 1층 resnet에서 깊이 전이 성립, **"블록이 깊은 현대 transformer에서는 모든 무한 깊이 극한에 근본적 한계"** | 깊이 ablation(12 vs 24층)에 lr 전이를 기대하면 안 됨 → 깊이 변형도 lr을 따로 본다 |
| CompleteP | Dey et al. 2025 [arXiv 2505.01618](https://arxiv.org/abs/2505.01618); [EleutherAI/nanoGPT-mup (completep)](https://github.com/EleutherAI/nanoGPT-mup/tree/completep) (MIT) | 깊이·폭 동시 전이 + non-lazy 학습, 이전 SOTA 대비 계산 효율 +12–34%, 폭/깊이 비율을 넓게 쓸 수 있음 | 보류. §9 ablation에서 깊이 변형이 이기고 이후 세대(2026 하반기)에서 폭·깊이를 함께 탐색하게 되면 그때 nanoGPT-mup 참조 구현을 옮김(M) |
| MiniCPM μP | scale_emb 12, init std 0.1, scale_depth 1.4, lr 0.01 | 9M–0.5B 풍동 실험에서 lr 0.01이 폭·깊이 무관 | 위와 같음 |

판정: **보류**. 이번 세대는 파일럿 3개로 lr을 정하고, 깊이 ablation은 lr 그대로 두되 발산 시에만 재조정한다.

## 7. 소형 LM 참조 레시피

출처는 각 행에. "—"는 1차 출처에 없음. 토큰은 사전학습 총량.

| 모델 (라이선스) | 층 × d | ffn | 헤드 Q/KV (head_dim) | 어휘 | 문맥 | 배치(토큰) | lr(최대) / 스케줄 | warmup | 토큰 | 기타 | 출처 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **trid 현행** | 12 × 768 | 2,048 SwiGLU | 12/12 (64) | 8,192 | 1,024 | 2,048 | 3e-4 cosine→10% | min(100, 10%) | ~0.07B (위키) | tied, AdamW (0.9,0.95) wd 0.1 전부, FP32 | `runs/trid-korean-medium-base-v1/training.json` |
| SmolLM2-135M (Apache-2.0 코드) | 30 × 576 | 1,536 | 9/3 (64) | 49,152 | 2,048 | 1,048,576 | 3e-3 WSD(20% 선형→0) | 2,000 | 2T | tied, wd 0.01, init 1/√d | [config](https://raw.githubusercontent.com/huggingface/smollm/main/text/pretraining/smollm2/config_smollm2_135M.yaml) |
| SmolLM2-360M | 32 × 960 | 2,560 | 15/5 (64) | 49,152 | 2,048 | 1,572,864 | 3e-3 WSD(520k/2.6M 스텝 = 20%) | 2,000 | 4T | tied, wd 0.01, init 0.03227 | [config](https://raw.githubusercontent.com/huggingface/smollm/main/text/pretraining/smollm2/config_smollm2_360M.yaml) |
| MobileLLM-125M (코드 FAIR Noncommercial Research License → 코드 미사용, 논문만) | 30 × 576 | 1,536 | 9/3 (64) | 32k | 2k | 2,097,152 (32 GPU × 32) | 2e-3 cosine | — | 1T (480k 스텝) | tied, GQA, wd 0.1; **LS-125M**: 인접 블록 공유로 2×30층, +0.7pt | [arXiv 2402.14905](https://arxiv.org/abs/2402.14905) Table 9, §3.1 |
| MobileLLM-350M | 32 × 960 | 2,560 | 15/5 (64) | 32k | 2k | 동일 | 동일 | — | 1T | Table 11: 125M급에서 12층×768 avg 43.9 vs 30층×512 44.8 vs 24층×576 44.8 | 동일 |
| TinyLlama-1.1B (Apache-2.0) | 22 × 2,048 | 5,632 | 32/4 | 32k | 2,048 | 2,097,152 | 4e-4 cosine | 2,000 | 3T (≈3 epoch of 950B) | FA2, 24k tok/s/A100 | [README](https://github.com/jzhang38/TinyLlama) |
| OLMo-2-1B (Apache-2.0) | 16 × 2,048 | 8,192 | 16/16 | 100,352 | 4,096 | — (7B: 1024×4096) | — (7B: 3e-4 cosine→10%) | 2,000 (7B) | 4T + 50B mid | QK-norm, reordered norm, z-loss, 임베딩 wd 제외, untied | [config](https://huggingface.co/allenai/OLMo-2-0425-1B/raw/main/config.json), [논문](https://arxiv.org/abs/2501.00656) Table 3 (1B 하이퍼파라미터는 미공개) |
| Pythia-160M (Apache-2.0) | 12 × 768 | 3,072 | 12 (64) | 50,304 | 2,048 | 2,097,152 | 6e-4 cosine | — | 300B | untied, parallel residual, rotary 25%; **비임베딩 85.06M = trid 84.95M과 같은 크기** | [config](https://huggingface.co/EleutherAI/pythia-160m/raw/main/config.json), [arXiv 2304.01373](https://arxiv.org/abs/2304.01373) |
| Cerebras-GPT-111M (Apache-2.0) | 10 × 768 | 3,072 | 12 (64) | 50,257 | 2,048 | 246k | 6e-4 (SP) / **6e-3 (μP)** 선형→10% | 375M 토큰 | 2.2B (20 TPP) | 선형 디케이가 cosine보다 좋았음 | [arXiv 2304.03208](https://arxiv.org/abs/2304.03208) Table 1 |
| Qwen3-0.6B (Apache-2.0) | 28 × 1,024 | 3,072 | 16/8 (128) | 151,936 | 40,960 | — | — | — | 36T (계열) | tied, QK-norm | [config](https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json) |
| EXAONE-4.0-1.2B (EXAONE AI Model License, 비상업) | 30 × 2,048 | 4,096 (2×d) | 32/8 (64) | 102,400 | 65,536 | — | — | — | 12T | tied, QK-Reorder-LN, 4K에서 학습 후 확장 | [config](https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B/raw/main/config.json), [arXiv 2507.11407](https://arxiv.org/abs/2507.11407) Table 1–2 |
| HyperCLOVAX-SEED-Text-0.5B (자체 라이선스) | 24 × 1,024 | 4,096 | 16/8 (128) | 110,592 | 8,192 (4K 지원) | — | — | — | — (4,358 A100h) | tied, `resid_pdrop 0.2`, "고성능 사전학습 모델"을 써서 학습 → 증류/초기화 `[INFERENCE]` | [config](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B/raw/main/config.json), [card](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B) |
| kanana-nano-2.1B (Apache-2.0 가중치; 미사용) | 32 × 1,792 | 8,064 | 24/8 (128) | 128,256 (Llama 3) | 8,192 | 512 × 8,192 (증류) | 1.2e-4 cosine (증류) | 100 | 300B (증류) | 8B에서 pruning + KL 증류 | [config](https://huggingface.co/kakaocorp/kanana-nano-2.1b-base/raw/main/config.json), [arXiv 2502.18934](https://arxiv.org/abs/2502.18934) A.4 |
| kanana 192M | — | — | — | 128,256 | — | 동일 | 동일 | — | 증류 | KMMLU 30.16(무작위 25) = 192M의 상한 참고치 | 동일 Table 6 |

읽기:
1. **깊이.** 100–400M급 공개 모델은 전부 24–32층이고 d/층 비율이 trid(768/12 = 64)의 1/2–1/3이다. MobileLLM Table 11은 같은 파라미터에서 12층×768(43.9) < 24층×576(44.8) = 30층×512(44.8)을 보여 준다. trid로 같은 파라미터를 맞춘 **24층×576, ffn 1,536, 9Q/3KV = 89.7M**(측정)이 §9 ablation 후보. 층이 늘면 MPS 커널 호출이 2배라 처리량이 떨어질 수 있으니 ablation 첫 100스텝에서 tok/s를 함께 기록한다. CPU 추론 지연은 FLOPs가 같으면 비슷하다 `[INFERENCE]`.
2. **warmup.** 공개 레시피는 2,000스텝(SmolLM2·TinyLlama·OLMo·Semenov) 또는 토큰 기준 N–375M(Porian·Cerebras). trid 100스텝은 예외적으로 짧다.
3. **wd.** SmolLM2 0.01, MobileLLM 0.1, Porian/kanana 독립 wd 1e-4(= η·λ 기준 0.1–0.5), Lion 0.5–0.6. 0.1이 중앙값.
4. **ffn 비율.** EXAONE 4.0 1.2B는 ffn = 2×d(30층 확보용), SmolLM2/MobileLLM은 2.67×d. trid 2,048/768 = 2.67×.

## 8. 한국어 소형 LM이 보고한 것 (토크나이저·깊이)

| 모델 | 공개된 학습 기록 | 토크나이저·어휘 | 깊이-폭 | trid에 주는 것 |
|---|---|---|---|---|
| **kanana** (Kakao, [arXiv 2502.18934](https://arxiv.org/abs/2502.18934)) | from-scratch 8B/26.8B: Llama 3 아키텍처·토크나이저(128k), **DeepSeek 하이퍼파라미터 법칙 + multi-step 스케줄**, 독립 wd 1e-4, z-loss 5e-6 (A.3). 소형(2.1B→192M)은 전부 **8B 교사로 pruning + KL 증류**(A.4: lr 1.2e-4, 배치 512×8,192, warmup 100) | Llama 3 128k 그대로. 한국어 edu 분류기로 KMMLU +2.2 | pruning ablation Table 16(≈1B 비임베딩): hidden 1,536·24헤드(35.10) > 1,280·24헤드(34.27) > 1,536·8헤드(24.39) — **헤드 수를 너무 줄이면 급락**. Table 17: tied(44.88) ≥ untied(44.45) | 92M에 헤드 12개는 유지(9Q/3KV 깊이 변형은 MobileLLM 비율). tied 유지. 증류 경로는 외부 교사 불가로 부적합 |
| **HyperCLOVA X SEED 0.5B** ([card](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B)) / **THINK** ([arXiv 2506.22403](https://arxiv.org/abs/2506.22403)) | SEED: 3단계(사전학습 "고품질 데이터 + 고성능 사전학습 모델 활용" → RFT → SFT), 4,358 A100h. THINK: **Peri-LN + μP**, 3단계 문맥 커리큘럼 | 110,592 BBPE | 0.5B가 24층×1,024(d/층 43), resid_dropout 0.2 | μP를 쓰는 한국어 사례. 소형 모델에 dropout 0.2를 켠 점은 작은 데이터 반복 학습 시 참고(실험 가치 낮음) |
| **EXAONE 3.0/3.5/4.0** (LG, [arXiv 2408.03541](https://arxiv.org/abs/2408.03541), [arXiv 2507.11407](https://arxiv.org/abs/2507.11407)) | 4.0 1.2B: 12T 토큰, 8.65×10²² FLOPs, 4K에서 학습 후 32K→64K 확장, QK-Reorder-LN(깊은 층 무용화 방지) | MeCab 사전분할 + BBPE 102,400 (한·영 반반) | 1.2B가 30층×2,048·ffn 2×d, tied | 깊이 우선 + ffn 축소로 층 수 확보. QK-norm 계열 정규화를 소형에도 적용 |
| **Polyglot-Ko** (EleutherAI, [arXiv 2306.02254](https://arxiv.org/abs/2306.02254), Apache-2.0) | GPT-NeoX, 1.3B: 배치 1,024×2,048 = 2M 토큰, 213B 토큰, **10만 스텝 근처에서 손실 급락 후 생성 붕괴**(1.3B·3.8B 모두) → 붕괴 전 체크포인트 선택 | 형태소 인지 BBPE 30,003 (MeCab) | 1.3B 24층×2,048 | 긴 학습의 불안정 사례. z-loss·QK-norm이 없던 설정. trid 규모에서는 FP32 master + clip 1.0로 보고된 문제 없음 |
| **KoGPT** (Kakao Brain, [HF](https://huggingface.co/kakaobrain/kogpt)) | 6B만 공개, 학습 로그 없음 | 64,512 | 28층×4,096 | 참고 가치 낮음 |

공통점: **공개된 한국어 소형 모델 중 from-scratch 학습 로그가 있는 것은 없다**(kanana 소형은 증류, SEED는 "고성능 모델 활용", EXAONE은 수치 미공개). 토크나이저는 모두 10만 단위 다국어 어휘이고 trid 8k 전용 어휘와 압축률이 같다는 것은 이전 보고서 §1.1에서 측정했다. 깊이는 예외 없이 24층 이상이다.

---

## 9. 다음 사전학습 실행 제안 (91M, ~10억 토큰, Mac 1대)

시간 예산(계산): 2.2k tok/s → **1일 = 190M 토큰**, 10억 토큰 = 5.3일(검증·체크포인트 오버헤드 제외). 전체 일정 ≈ lr 파일럿 1일 + ablation 3일 + 본 실행 5.3일 ≈ **10일**.

### 9.1 본 실행 설정 (control)

| 항목 | 값 | 근거 | 코드 변경 |
|---|---|---|---|
| 모델 | 12 × 768, SwiGLU 2,048, 12 헤드, tied, **`qk_norm: true`**, `attn.output`·`down` **zero-init** | QK-norm: speedrun 기록 5, OLMo 2 §3.3.2, Qwen3·EXAONE 4.0 채택. zero-init: 기록 5 (muP-like) | `config.json` 1줄; `trid/model.py` 148–150행 (S) |
| 토크나이저 | 이전 보고서 §1 (혼합 말뭉치, Llama-3 정규식, 8k) | — | — |
| 스텝당 토큰 | **262,144** (batch 2 × accumulate 128) | DeepSeek 193k·Porian 290k·Zhang 300–385k·Power Lines 175k(B_opt)/1.39M(B_crit) 안 (§3) | 인자만 |
| 샘플링 | **BOS 정렬** 문서 시작 + 문서 순회(epoch 명시) | speedrun 기록 26 (−1.1%); Muennighoff 4 epoch 무해 | `sample_batch` (S) |
| 옵티마이저 | AdamW **β = (0.9, 0.98)**, ε 1e-8, **wd 0.1을 2D hidden 가중치에만** (임베딩·RMSNorm 제외) | β₂: Porian(0.99, 배치 ≤262k), Wen(0.98), Semenov 124M 최적 0.999. wd 제외: OLMo 2, llm.c | 205행 파라미터 그룹 (S) |
| lr | **파일럿 {1e-3, 2e-3, 4e-3} × 60M 토큰(≈ 230스텝, 각 7.6시간) 후 선택**; 사전 예상 2e-3 | DeepSeek 1.9e-3, Semenov 1e-3(131k), Porian 5e-3, SmolLM2 3e-3(1M) | — |
| warmup | **90M 토큰(≈ 350스텝)** | Porian min(N, 20%), Cerebras 375M, Semenov 2,000스텝 | `lr_ratio` 34행 → `warmup_tokens` 인자 (S) |
| 스케줄 | **WSD, 디케이 20%, 선형→0** (옵션: 1−√) | Hägele Takeaway 1–3·Fig. 22; Semenov Takeaway 8; SmolLM2 20% | 이미 구현; (1−√) 추가 S |
| 정밀도 | bf16 autocast + `torch.compile`, z-loss 0 | 측정 +79%; Semenov Takeaway 9(z-loss 무효). 스파이크 발생 시 5e-6(kanana) | 이미 구현 |
| clip | 1.0 (Semenov 124M 최적 0.5도 후보) | — | — |
| 평가 | 250스텝마다 도메인별 BPC + stable 구간 **EMA(0.999) 가중치**로도 평가 | Hägele SWA Takeaway 5; speedrun 92 | 학습 루프 (S) |
| 체크포인트 | 옵티마이저 상태 키를 일반화(`exp_avg*` 고정 제거) | Muon 변형 재개를 위해 | `save/restore_checkpoint` (S) |

예상 처리량 영향: accumulate·BOS 정렬·zero-init·QK-norm은 FLOPs 변화 없음(QK-norm은 head_dim RMSNorm 2개/층, <1%). 10억 토큰 = 3,815 스텝 ≈ 5.3일.

### 9.2 1일 ablation 3개 (각 ≈ 170M 학습 토큰 = 650스텝 @262k; 2.2k tok/s로 21.5시간 + 검증·컴파일 여유)

공통: control과 같은 데이터 순서(seed 고정), **마지막 20%(130M 토큰 이후) 디케이**를 각 변형에 똑같이 적용하고 **디케이 끝의 BPC로만** 비교한다(Wen: 디케이 전 비교는 순위가 뒤집힘). 170M 토큰은 2 TPP로 본 실행(11.8 TPP)보다 짧지만, Wen의 "Muon은 낮은 데이터 비율에서 가장 유리, 8× Chinchilla 이상에서 SOAP/Kron" 결과상 10억 토큰(<1× Chinchilla)의 순위를 바꿀 가능성은 낮다 `[INFERENCE]`. 각 변형이 control보다 BPC 0.01 이상 좋으면 채택(Semenov의 유의 임계 3e-3 손실 ≈ BPC 0.004의 2배).

| # | 변형 | 바꾸는 것 | 기대 | 판단 기준 | 손대는 파일 |
|---|---|---|---|---|---|
| A | **Muon(hidden) + AdamW(embed·norm)** | `torch.optim.Muon(lr=0.02→파일럿 lr×10 `[INFERENCE: Semenov 비율 1e-2:1e-3]`, momentum 0.95, nesterov, wd 0.1, adjust_lr_fn="match_rms_adamw")` 60개 행렬; `qkv` → q/k/v 분리; 첫 300스텝 momentum 0.85→0.95(기록 9) | 토큰 효율 1.2–1.4× (Wen 130M; Semenov large-batch에서는 AdEMAMix ≥ D-Muon) | BPC −0.02 이상이면 채택. 미달이면 B 대신 AdEMAMix(S)로 재실행 | `trid/pretraining.py`, `trid/model.py` Attention |
| B | **깊이: 24 × 576, ffn 1,536, 9Q/3KV (89.7M)** | `config.json`만. lr은 control과 동일(depth-μP 전이 불성립이므로 발산 시 ½) | MobileLLM Table 11 +0.9pt(≈ 손실 −0.02~−0.03 `[INFERENCE]`); 한국어 소형 전부 24층+ | BPC와 **tok/s** 둘 다 기록. BPC 이득이 처리량 손실(층 2배 → 커널 호출 2배)을 넘어야 채택: 같은 벽시계에서의 BPC로 비교 | `config.json`, (GQA는 `kv_heads` 옵션 있음) |
| C | **speedrun-lite 아키텍처 묶음**: logit softcap 15 + U-net skip(6→6) + 공유 value embedding 1개(앞 3층·뒤 3층) | 기록 9·11·14·18의 순수 PyTorch 부분 | 기록 기준 누적 −20~30% 시간이나 trid 규모·과제에서는 미검증 | 묶음으로 이기면 다음 세대에서 요소 분해. +6.3M 파라미터(VE)는 CPU 메모리 허용 | `trid/model.py` (M), 체크포인트 키 추가 |

변형이 서로 독립이므로 A·B·C가 각각 이기면 조합해 본 실행에 넣되, 조합은 ablation 없이 적용한다(시간 예산). SFT 단계의 길이 정렬 배치(§5)와 PLW 0.1은 사전학습과 무관하므로 별도 반나절 실험.

---

## 10. 우선순위 행동 목록

1. **배치·warmup·β₂·wd 그룹 수정 — 적용, S, 반나절.** `trid/pretraining.py`: `warmup_tokens` 인자(`lr_ratio`), 파라미터 그룹(2D hidden만 wd), β₂ 0.98, cosine 바닥 0.01. 근거: §3 네 공식 모두 B 175k–390k·warmup ≈ N 토큰; Semenov Takeaway 8. 이것만으로 현행 레시피(2k 토큰/스텝·warmup 100)와의 차이가 다른 모든 항목보다 크다 `[INFERENCE: Wen Fig. 1 "lr 하나 튜닝 = 2×"에 준함]`.
2. **BOS 정렬·문서 순회 샘플링 + EMA 평가 — 적용, S, 반나절.** `sample_batch`, 학습 루프. 근거: 기록 26, Hägele SWA.
3. **lr 파일럿 3개(60M 토큰씩, 1일)** → 2e-3 근처 결정. 근거: §3 표.
4. **QK-norm on + zero-init — 적용, S.** `config.json`, `model.py` 148–150행. 근거: 기록 5, OLMo 2.
5. **Ablation A: Muon — 실험, M, 1일 + 구현 1일.** 체크포인트 상태 키 일반화 포함. 실패 시 AdEMAMix(S). 근거: Wen·Semenov·speedrun 3자 모두 행렬 기반 또는 AdEMAMix가 AdamW를 이김.
6. **Ablation B: 24×576 깊이 — 실험, S(설정만), 1일.** 근거: MobileLLM Table 11, 한국어 소형 모델 전부 24층+.
7. **Ablation C: softcap·U-net·value embedding 묶음 — 실험, M, 1일 + 구현 1일.** 근거: 기록 9·11·14·18.
8. **SFT: 길이 정렬 배치 + prompt loss weight 0.1 — 실험, S, 반나절.** `trid/training.py` `batch`·`loss_for`. 근거: Krell(패딩 50–89%), Huerta-Enochian(PLW 0.01–0.5), Shi(긴 지시·짧은 출력).
9. **부적합으로 닫는 것:** FlexAttention·슬라이딩 윈도·FP8(MPS 미지원, 측정), Shampoo(복잡도), Lion(토큰 효율 열세), Schedule-Free(WSD에 뒤짐), DoReMi/RegMix(도메인 3개), μP·CompleteP(이번 세대 폭 고정), kanana식 증류(외부 교사 불가).

---

## 부록 A. 측정 기록 (2026-10-28, 프로젝트 venv PyTorch 2.14.1, MPS, 동시 부하: 91M 사전학습·SFT·수집기 2개)

- `torch.optim.Muon` 존재 확인(`'Muon' in dir(torch.optim)` → True). 기본값 lr 1e-3, wd 0.1, momentum 0.95, nesterov True, ns_steps 5, `adjust_lr_fn ∈ {None(original), match_rms_adamw, spectral_unclamped}`; 2D가 아닌 파라미터는 `ValueError`.
- Muon 1스텝(12층 91M과 같은 60개 행렬: 2304×768, 768², 2048×768 ×2, 768×2048): **441 ms**; 같은 텐서 AdamW **150 ms**. 262k 토큰/스텝(≈119 s)에서 각각 0.37%·0.13%.
- `torch.compile(flex_attention)` + `create_block_mask`(슬라이딩 256) 역전파: `NotImplementedError: FlexAttention does not support backward on MPS`. 순전파 eager는 전체 점수 행렬을 만드는 비융합 경로.
- `torch.randn(64,64,device="mps").to(torch.float8_e4m3fn)`: `RuntimeError: Undefined type Float8_e4m3fn`.
- SDPA 순전파(2×12×1024×64): `is_causal=True` 2.0 ms vs bool 슬라이딩 마스크 9.9 ms.
- 파라미터 수(`Trid.parameter_count`): 현행 91.25M(비임베딩 84.95M); untied +6.29M; 24×576·ffn 1,536·9Q/3KV 89.68M; 30×576 110.92M; 16×768 119.56M; 어휘 16,384 97.54M.
- 시간 예산: 2,200 tok/s × 86,400 s = 190M 토큰/일; 10억 토큰 5.3일. 2,000 tok/s면 173M/일·5.8일.

## 부록 B. 1차 출처

- modded-nanogpt (MIT): https://github.com/KellerJordan/modded-nanogpt ; Muon 블로그: https://kellerjordan.github.io/posts/muon/ ; KellerJordan/Muon (MIT): https://github.com/KellerJordan/Muon ; 옵티마이저 비교 기록: https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2024-10-29_Optimizers ; torch.optim.Muon: https://docs.pytorch.org/docs/main/generated/torch.optim.Muon.html ; Moonlight (MIT): https://github.com/MoonshotAI/Moonlight , https://arxiv.org/abs/2502.16982 ; NorMuon: https://arxiv.org/abs/2510.05491
- 옵티마이저 벤치마크: Wen et al. https://arxiv.org/abs/2509.02046 ; Semenov et al. https://arxiv.org/abs/2509.01440
- SOAP (MIT): https://github.com/nikhilvyas/SOAP , https://arxiv.org/abs/2409.11321 ; Distributed Shampoo (BSD): https://github.com/facebookresearch/optimizers ; Schedule-Free (Apache-2.0): https://github.com/facebookresearch/schedule_free , https://arxiv.org/abs/2405.15682 ; Lion (MIT impl): https://github.com/lucidrains/lion-pytorch , https://arxiv.org/abs/2302.06675 ; AdEMAMix (MIT): https://github.com/apple/ml-ademamix , https://arxiv.org/abs/2409.03137 ; MARS (Apache-2.0): https://github.com/AGI-Arena/MARS , https://arxiv.org/abs/2411.10438
- 배치·lr 법칙: DeepSeek LLM https://arxiv.org/abs/2401.02954 ; Porian et al. https://arxiv.org/abs/2406.19146 ; Zhang et al. https://arxiv.org/abs/2410.21676 ; Power Lines https://arxiv.org/abs/2505.13738 ; MiniCPM https://arxiv.org/abs/2404.06395
- 스케줄: Hägele et al. https://arxiv.org/abs/2405.18392 , 코드(MIT) https://github.com/epfml/schedules-and-scaling
- 데이터: Muennighoff et al. https://arxiv.org/abs/2305.16264 (datablations Apache-2.0 https://github.com/huggingface/datablations) ; DoReMi https://arxiv.org/abs/2305.10429 ; RegMix (MIT) https://github.com/sail-sg/regmix , https://arxiv.org/abs/2407.01492 ; 시퀀스 길이 warmup https://arxiv.org/abs/2108.06084 ; 패킹 https://arxiv.org/abs/2107.02027 , Best-fit packing https://arxiv.org/abs/2404.10830 ; 손실 마스킹 https://arxiv.org/abs/2405.14394 , https://arxiv.org/abs/2401.13586
- μP: https://arxiv.org/abs/2203.03466 , https://github.com/microsoft/mup (MIT) ; Depth-μP https://arxiv.org/abs/2310.02244 ; CompleteP https://arxiv.org/abs/2505.01618 , https://github.com/EleutherAI/nanoGPT-mup/tree/completep (MIT)
- 레시피: SmolLM2 configs https://github.com/huggingface/smollm/tree/main/text/pretraining/smollm2 ; MobileLLM https://arxiv.org/abs/2402.14905 ; TinyLlama https://github.com/jzhang38/TinyLlama ; OLMo 2 https://arxiv.org/abs/2501.00656 , https://huggingface.co/allenai/OLMo-2-0425-1B ; Pythia https://arxiv.org/abs/2304.01373 , https://huggingface.co/EleutherAI/pythia-160m ; Cerebras-GPT https://arxiv.org/abs/2304.03208 ; Qwen3-0.6B https://huggingface.co/Qwen/Qwen3-0.6B ; EXAONE 4.0 https://arxiv.org/abs/2507.11407 , https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B ; HyperCLOVA X SEED https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B , THINK https://arxiv.org/abs/2506.22403 ; kanana https://arxiv.org/abs/2502.18934 , https://huggingface.co/kakaocorp/kanana-nano-2.1b-base ; Polyglot-Ko https://arxiv.org/abs/2306.02254 ; KoGPT https://huggingface.co/kakaobrain/kogpt
