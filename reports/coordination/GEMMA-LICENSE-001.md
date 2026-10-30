# GEMMA-LICENSE-001 → A: Gemma 3(4B) 사용·배포 조건 확인 (B, 법률 검토 아님)

- **목적:** A는 gemma3:4b를 비용 없는 내부 대화 비교에 쓰려 하고, 배포할 계획은 없다. 나중에 오프라인 공공기관 상업 배포나 파생 모델 배포를 검토할 때 지켜야 할 조건을 공식 문서 기준으로 정리했다.
- **이 문서는 법률 자문이 아니다.** 적합성을 보장하지 않는다. 미확인 사항은 마지막 절에 따로 적었다.
- **Gemma 3는 Apache 2.0이 아니다.** 아래의 **Gemma Terms of Use**를 적용받는다.
- B는 모델을 다운로드·설치·학습하지 않았다.
  - 웹 요청 8건은 모두 `scripts/logged_get.py`로 보냈다(로그 `logs/requests-*.jsonl`).
  - ai.google.dev robots.txt 1건과 공식 문서 3건, huggingface.co robots.txt 1건과 메타데이터 API 1건, ollama.com robots.txt 1건과 라이브러리 페이지 1건이다.

## 확인한 공식 문서 (확인 시각 2026-10-31 02:31~02:33 KST)

| 문서 | URL | 문서 기재 날짜 |
|---|---|---|
| Gemma Terms of Use | https://ai.google.dev/gemma/terms | Last modified: April 1, 2026 (페이지 Last updated 2026-04-01 UTC) |
| Gemma Prohibited Use Policy | https://ai.google.dev/gemma/prohibited_use_policy | Last modified: February 21, 2024 |
| Gemma 3 model card | https://ai.google.dev/gemma/docs/core/model_card_3 | Last updated 2025-08-14 UTC. "Terms of Use: Terms" 링크 |
| (보조) HF 메타데이터 | https://huggingface.co/api/models/google/gemma-3-4b-it | `license: gemma`, `gated: manual`, "review and agree to Google's usage license" |
| (보조) Ollama 라이브러리 | https://ollama.com/library/gemma3:4b | license 층이 "Gemma Terms of Use, Last modified: February 21, 2024"(8.4kB) |

- 약관 머리말: "The terms below apply to Gemma models listed in the Appendix". Appendix에 **Gemma 3**가 있다.
- 같은 머리말에 "For Gemma 4 terms, see the Gemma 4 license"라고 따로 안내한다. **Gemma 4 라이선스는 이번에 보지 않았다.** Gemma 3와 섞어 쓰지 않는다.

## 1. 무엇이 Gemma이고 무엇이 파생 모델인가 (약관 1.1)

- **(c) Gemma:** "the set of machine learning language models, trained model weights and parameters identified in the Appendix, regardless of the source that you obtained it from"
  - Ollama 등 어디서 받았든 같은 약관이 적용된다.
- **(e) Model Derivatives:** "(i) modifications to Gemma, (ii) works based on Gemma, or (iii) any other machine learning model which is created by transfer of patterns of the weights, parameters, operations, or Output of Gemma … including distillation methods … or methods based on the generation of synthetic data Outputs by Gemma for training that model. For clarity, Outputs are not deemed Model Derivatives."
- **TRID에 주는 함의 (중요):**
  - Gemma를 미세조정하거나 병합한 모델은 파생 모델이다.
  - **Gemma 출력으로 학습 데이터를 만들어 다른 모델(예: Qwen 기반 TRID)이 비슷하게 동작하도록 학습시키면, 그 모델도 파생 모델 정의에 들어갈 수 있다.**
  - 내부 대화 비교를 넘어 Gemma 출력을 학습 데이터로 쓰는 일은 별도 결정 없이 하지 않아야 한다.
  - 출력 자체는 파생 모델이 아니다. 3.3에 따라 Google은 출력에 권리를 주장하지 않고, 출력의 책임은 사용자에게 있다.
- **(b) Distribution:** "any transmission, publication, or other sharing of Gemma or Model Derivatives to a third party". API나 웹으로 제공하는 Hosted Service도 포함한다.
  - [추정] 기관에 오프라인으로 설치해 넘기는 것도 제3자에게 공유하는 것이므로 Distribution에 해당한다.

## 2. 배포·재배포할 때 지켜야 할 것 (약관 3.1, 원문 요지)

| 요건 | 원문 근거 |
|---|---|
| 사용 제한(3.2)을 배포·사용 계약(라이선스, 이용약관 등)에 **강제 가능한 조항**으로 넣고, 받는 사람에게 사용 제한 대상이라는 사실을 알린다 | "You must include the use restrictions referenced in Section 3.2 as an enforceable provision in any agreement … and you must provide notice to subsequent users" |
| 모든 제3자 수령자에게 **약관 사본**을 준다 | "You must provide all third party recipients of Gemma or Model Derivatives a copy of this Agreement." |
| 수정한 파일에는 수정했다는 **눈에 띄는 고지**를 단다 | "You must cause any modified files to carry prominent notices stating that you modified the files." |
| Hosted Service가 아닌 배포에는 **"Notice" 텍스트 파일**을 함께 넣는다. 정해진 문구를 그대로 쓴다 | "Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms" |
| 자기 수정분에 별도 조건을 붙일 수는 있지만, 약관과 충돌하면 안 된다 | "Any additional or different terms and conditions you impose must not conflict with the terms of this Agreement." |

- **상업 이용:** 약관에서 상업 이용을 금지하는 조항은 찾지 못했다. 2.2는 "only in accordance with the terms of this Agreement"라는 조건 아래 사용·복제·수정·배포를 허용한다.
  - 이것을 "상업 이용 허용 보장"으로 바꿔 쓰지 않는다. 판단은 법률 검토 몫이다.

## 3. 금지 용도 (약관 3.2 + Prohibited Use Policy, 참조로 편입)

- 약관 3.2는 Prohibited Use Policy에 적힌 용도와 법령 위반 용도를 금지한다.
- 또한 "Google reserves the right to restrict (remotely or otherwise) usage"라는 원격 제한 유보 조항이 있다.
- **공공기관 업무 보조에서 특히 관련 있는 항목** (Prohibited Use Policy 원문):
  - "Making automated decisions in domains that affect material or individual rights or well-being (e.g., finance, legal, employment, healthcare, housing, insurance, and social welfare)"
    - 권리·급여에 영향을 주는 자동 결정에 쓰면 안 된다. 사람이 검토하는 보조 도구로만 써야 한다.
  - "Misleading claims of expertise or capability made particularly in sensitive areas (e.g. health, finance, government services, or legal)"
    - 정부 서비스·법률 분야에서 전문성·정확성을 과장해 표시하면 안 된다.
  - "Engagement in the illegal or unlicensed practice of any vocation or profession including … legal, medical, accounting, or financial professional practices"
  - "Generating, gathering, processing, or inferring sensitive personal or private information about individuals without obtaining all rights, authorizations, and consents required by applicable laws"
  - 권리 침해 콘텐츠, 사칭, 출처 오인(생성물을 사람이 쓴 것처럼 표시해 속이는 것) 금지.
- 이 제한을 배포 계약에 넣어 최종 사용자(기관)에게도 걸어야 한다(2절 첫 줄).

## 4. 그 밖의 조항

- **4.2 상표:** Google 상표·로고 사용권은 없다. 보증·제휴를 암시하면 안 된다.
- **4.3·4.4:** 보증 부인과 책임 제한. 사용·배포의 적합성은 사용자가 판단한다.
- **4.5 해지:** 위반하면 Google이 해지할 수 있다. 해지되면 보유한 Gemma와 **파생 모델의 모든 사본을 삭제하고** 사용·배포를 멈춰야 한다.
- **4.6 준거법·관할:** 미국 캘리포니아주 법, Santa Clara County 법원 전속 관할.
- **판본 차이:** Ollama가 함께 배포하는 약관 사본은 2024-02-21판이고, 현재 공식 판은 2026-04-01판이다. 문장 단위로 비교한 차이는 다음과 같다.
  - Gemma 정의: "identified at ai.google.dev/gemma" → "identified in the Appendix".
  - 4.1: "you must make reasonable efforts to use the latest version of Gemma" 문장이 삭제됐다.
  - 3절(배포 요건)과 3.2(사용 제한)는 같다.
  - 어느 판이 특정 사본에 적용되는지는 미확인이다. 배포한다면 확인 당일의 공식 판을 동봉하고 확인일을 기록하는 쪽이 보수적이다.

## 5. 구성요소별 라이선스 구분

| 구성요소 | 적용 조건 | 근거 |
|---|---|---|
| 가중치·파라미터 (text model) | Gemma Terms of Use | 약관 1.1(c) |
| 비전 projector (Ollama gemma3:4b의 clip 층, 851MB) | Gemma Terms of Use [추정] | Gemma 3 모델의 일부(멀티모달)로 배포된다. 약관이 구성요소별로 따로 정하지는 않는다 |
| tokenizer (`tokenizer.model`·`tokenizer.json`, GGUF 안에 내장된 vocab) | **명시 조항 없음.** [추정] Gemma Terms 아래 배포되는 자료로 취급 | 약관 정의는 "models, trained model weights and parameters"만 적는다. HF 공식 저장소는 tokenizer 파일을 가중치와 같은 gated 저장소(`license: gemma`)에 둔다. Apache로 표기하지 않는다 |
| chat template | Google HF 저장소의 `chat_template.json`은 같은 `license: gemma` 저장소에 있다. Ollama template 층(358B)은 Ollama가 작성한 Go 템플릿이며 **라이선스 미표기(미확인)** | HF 파일 목록, Ollama 페이지 |
| 실행 소프트웨어 | Ollama: MIT (로컬 `/opt/homebrew/Cellar/ollama/0.31.1/LICENSE`). llama.cpp: MIT (`/tmp/llama.cpp/LICENSE`). 이것은 런타임 라이선스이고, 모델 가중치의 조건을 바꾸지 않는다 | 로컬 LICENSE 파일 |
| 모델 출력 | 파생 모델이 아님. 권리 주장 없음. 책임은 사용자 | 약관 1.1(e), 3.3 |

## 6. 로컬 상태 (B 관찰, 변경 없음)

- `ollama list`에는 **gemma3:4b가 없다.** gemma3:12b만 있다.
  - "기존 설치"라는 전제와 다르다. A가 무엇을 기준으로 할지 확인이 필요하다.
- A 경로에 `runs/baseline-compare-003/gemma3-4b-it-q4km-ollama-sha256-f06e91e1.gguf` 파일이 있다.
  - 크기 2,499,012,288 바이트, 수정 시각 2026-10-31 02:32 KST.
  - 텍스트 모델 GGUF뿐이다. projector는 없다.
  - GGUF 메타데이터에 `general.license`는 없다.
  - 같은 폴더에 약관 사본이나 Notice 파일이 없다.
  - Ollama blob 폴더에서 같은 digest(f06e91e1)를 찾지 못했다. 출처 경로는 A가 기록해야 한다.
- 내부 비교에만 쓰면 3.1의 배포 요건은 적용되지 않는다. [추정] 제3자에게 넘기지 않기 때문이다. 3.2의 금지 용도는 내부 사용에도 적용된다.

## 7. 미확인 / 법률 검토가 필요한 것

- 기관에 오프라인으로 설치해 납품하는 것이 Distribution에 해당하는지에 대한 공식 해석. [추정]으로는 해당한다.
- 국내 공공기관 계약에서 캘리포니아 준거법·관할 조항, 원격 사용 제한 유보, 해지 시 삭제 의무를 받아들일 수 있는지.
- 공공 조달, 국가 보안 기준, 수출 통제 같은 국내외 규제와의 관계.
- tokenizer·template에 별도 조건이 있는지(약관에 명시 조항 없음).
- 특정 사본에 적용되는 약관 판본(2024-02-21판 사본 vs 2026-04-01 공식 판).
- Gemma 출력으로 TRID 학습 데이터를 만들었을 때 파생 모델 해당 여부의 경계(정의 1.1(e)(iii)의 "to cause that model to perform similarly to Gemma").
- Gemma 4 라이선스(보지 않음).

SUPERVISION_REPORT_READY GEMMA-LICENSE-001
