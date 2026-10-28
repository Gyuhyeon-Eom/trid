# 차기 사전학습 코퍼스(~1B 토큰) 한국어 소스 조사

작성일: 2026-10-27. 대상: `trid` 91M 모델의 다음 사전학습(현재 코퍼스 `data/korean-foundation-v2/`, 한국어 위키백과 단일 소스 약 80M 토큰). 목표는 공문·규정·공고·보도자료·법령 RAG 응답에 가까운 도메인으로 약 10억 토큰을 모으되, **상업적 이용·학습·학습된 모델 배포**가 허용되는 소스만 본 믹스에 넣는 것이다.

표기 규칙
- 토큰 수는 별도 표기가 없으면 **이 프로젝트 토크나이저**(`runs/trid-korean-medium-base-v1/tokenizer.json`, 8192 BPE) 기준이다. 외부 데이터셋 카드의 토큰 수는 각자 다른 토크나이저(공백 분리, Gemma 3 등)를 쓰므로 그대로 비교할 수 없다.
- "(추정)" 표시는 1차 출처 숫자가 아니라 파일럿 샘플 외삽이나 본 조사에서 계산한 값이다.
- 같은 날 동료 에이전트가 수행한 파일럿 수집 결과(`data/law-ko-source/`, `data/korea-kr-briefing-source/`, `data/wiki-ko-full-source/`의 `pilot-stats.json`·`provenance.json`)를 1차 관측으로 인용한다.

---

## 1. 요약

| 우선순위 | 소스 | 텍스트 종류 | 가용 규모(프로젝트 토크나이저) | 라이선스 | 상업·학습·모델배포 | 접근 | 도메인 근접성 | 판정 |
|---|---|---|---|---|---|---|---|---|
| 1 | 법제처 국가법령정보센터 Open API (법령·행정규칙·자치법규) | 법령 전문, 훈령·예규·고시, 조례·규칙 | 법령 61M / 행정규칙 171M / 자치법규 361M (추정, 파일럿 외삽) | 저작권법 제7조 비보호 저작물 + 법제처 "영리 목적 이용 포함 자유 활용" | 가능(출처표시) | Open API, 1 req/s 자율 제한, 전체 수집 약 6일 | **최상** | **본 믹스** |
| 1 | 법제처 판례·헌재결정례·법령해석례·행정심판재결례 | 판결문(비실명), 결정문, 해석례, 재결서 | 판례 317M / 헌재 69M / 해석례 18M / 재결례 110M (추정) | 저작권법 제7조 제3호 | 가능 | 위와 동일 API | 상(규정 해석 문체) | **본 믹스** |
| 1 | 정책브리핑(korea.kr) 보도자료 | 부처·기관 보도자료 본문(HWP/HWPX/PDF 첨부 포함) | 약 535M (추정, 2015년 이후 296,480건 외삽) | 공공누리 제1유형(텍스트 한정, 건별 표시 확인) | 가능(출처표시) | 웹 크롤, 첨부 파싱, 약 13일 | **최상** | **본 믹스** |
| 2 | 한국어 위키백과 전체 덤프 | 백과사전 | 307.7M (파일럿 측정, 379,068문서) | CC BY-SA 4.0 | 가능(SA·출처표시) | HF/덤프 벌크, 이미 로컬 보유 | 중 | **본 믹스**(일반 지식·어휘) |
| 2 | 국회회의록 | 본회의·위원회 속기록 | 미측정; 역대 회의록 50,523건 | 공공누리 제1유형(국회사무처) | 가능(출처표시) | 공공데이터포털 API(LINK형, 활용신청) / 회의록시스템 | 중상(구어체 국정 담화) | 본 믹스 후보(본문 벌크 경로 미확인) |
| 3 | 공공데이터포털(data.go.kr) 텍스트성 데이터셋 | 공공서비스 안내, FAQ, 통계 메타 등 | 소규모, 데이터셋별 상이 | 데이터셋별 공공누리 유형 표시(제1유형 또는 "제한 없음") | 데이터셋별 확인 | 파일 다운로드 또는 API 활용신청 | 상(일부) | 소규모 보강 |
| 4 | 한국어 위키뉴스 | 뉴스 | 0.9M (본 조사 측정: 851문서, 1.47M자) | CC BY 4.0 | 가능 | 덤프 1.3MB | 중 | 무시 가능 |
| 4 | 한국어 위키문헌 | 고전·법령 원문·공문서 사본 등 | 86.8M 단어(사이트 통계) → 약 150M 토큰(추정) | CC BY-SA 4.0 + 개별 문서 PD/제7조 혼재 | 가능 | 덤프 148MB | 중(고문·한자 비중 큼) | 선별 사용 |
| 플래그 | FineWeb-2 `kor_Hang` | 웹 크롤(Common Crawl) | 48.6B 단어 / 60.9M 문서 / 213GB | ODC-By(패키징) + Common Crawl ToU; 원문 저작권은 각 사이트 | 법적 회색지대(한국 저작권법에 TDM 면책 없음) | HF parquet | 하(잡다한 웹) | **플래그** |
| 플래그 | HPLT 2.0/3.0 `kor_Hang` | 웹 크롤(Internet Archive + CC) | v2 cleaned 3.89e7 문서 / 1.97e10 단어 / 8.92e10 자; v3 74.79M 문서 / 97.58B 토큰(Gemma3) / 164B 자 | CC0(패키징만), "텍스트 소유권 없음" 명시 | 위와 동일한 회색지대 | HTTP 벌크(v3 kor 150.6GB) | 하 | **플래그** |
| 플래그 | CulturaX `ko` (mC4 + OSCAR) | 웹 크롤 | 20,557,310 문서 / 24.77B 토큰(카드 기준) | mC4(ODC-By) + OSCAR 약관 승계; OSCAR-2301은 현재 접근 중단(gated) | 회색지대 + OSCAR "TDM/연구 예외 내 사용" 문구 | HF parquet | 하 | **플래그** |
| 플래그 | 국립국어원 모두의 말뭉치 | 신문·문어·구어·웹 등 (2020년 초기 공개분 13종 18억 어절) | 수십억 어절(원시 말뭉치) | 이용 약정(승인된 목적 한정, 제3자 양도 금지, 결과물 외부 공개 전 사전 승인) | 상업 R&D는 가능하나 **모델 공개 전 국어원 사전 승인 필요** | 회원가입(본인인증) + 말뭉치별 신청·승인(5근무일) | 중 | **플래그**(절차·승인 의존) |
| 플래그 | AI Hub (행정 문서 기계독해, 법률 텍스트 등) | 행정문서 지문(약 60만 건 원시) | 미측정 | NIA 이용정책: 영리·비영리 R&D 가능, 출처(NIA) 표시, 국외 반출·제3자 제공 금지, "학습용으로만" | 모델 배포 허용 여부 명문 없음 | 로그인·**내국인만 신청** | 상 | **플래그** |
| 플래그 | KorQuAD 1.0/2.0 | 위키 기반 QA | 소규모 | CC BY-ND 2.0 KR | 변경 금지(ND) → 학습 데이터 가공 자체가 쟁점 | 다운로드 | 중 | **플래그**(사전학습 제외) |
| 플래그 | KLUE | 뉴스·위키 등 기반 NLU | 소규모(이미 `data/klue-mrc-source/`로 일부 사용) | CC BY-SA 4.0 | 가능하나 SA 전염 논점 | HF | 중 | 사전학습에는 불필요(평가·SFT용) |
| 플래그 | 법원 판결서 인터넷열람 | 판결문 | 2013년(형사)/2015·2023년(민사·행정·특허) 이후 | 저작권법 제7조이나 열람 서비스는 건당 수수료·비실명 처리 | 텍스트 자체는 가능 | 건별 유료 열람, 벌크 불가 | 상 | 법제처 판례 API로 대체 |
| 플래그 | 나무위키 | 커뮤니티 위키 | 2021-03 덤프 867,024문서(HF 미러, 3GB) | **CC BY-NC-SA 2.0 KR** | **상업 이용 불가** | 덤프 | 중하 | **제외** |

---

## 2. 평가 기준

1. **라이선스**: (a) 상업적 이용, (b) 학습 데이터로 가공(2차적 저작물 작성), (c) 학습된 모델의 배포. 셋 다 명시적으로 허용되거나 저작권법상 보호 대상이 아닌 경우만 "명확"으로 분류한다. 패키징만 CC0/ODC-By이고 본문 권리는 제3자에게 있는 웹 크롤은 한국 저작권법에 TDM 면책 조항이 없으므로(2026-10 현재, 본 조사에서 확인한 범위 내 [INFERENCE]) "회색지대"로 분류한다.
2. **규모**: 1차 출처 수치 또는 본 프로젝트 파일럿 측정치.
3. **접근 방식과 속도**: 벌크 덤프 > API > 크롤 > 신청·승인.
4. **도메인 근접성**: 공문·규정·공고·보도자료·법령과의 문체·어휘 유사성.
5. **품질·중복·개인정보 위험**.

---

## 3. 소스별 상세

### 3.1 법제처 국가법령정보센터 Open API (DRF)

- **텍스트**: 현행 법령(`law`), 행정규칙(`admrul`: 훈령·예규·고시), 자치법규(`ordin`: 조례·규칙), 판례(`prec`), 헌재결정례(`detc`), 법령해석례(`expc`), 행정심판재결례(`decc`). 목록 `http://www.law.go.kr/DRF/lawSearch.do`, 본문 `http://www.law.go.kr/DRF/lawService.do`. XML 응답 권장(JSON은 조문 내 줄바꿈이 사라짐 — 파일럿 provenance 기록).
- **규모(파일럿 외삽, 추정)**: 목록 건수 법령 5,628 / 행정규칙 24,217 / 자치법규 161,807 / 판례 175,279 / 헌재 38,736 / 해석례 8,903 / 재결례 35,225. 파일럿 700건(각 100건)에서 640건 유지, 4,851,618자, 2,890,164 토큰(1.68자/토큰). 외삽 합계 약 **1.108B 토큰**(법령 61M, 행정규칙 171M, 자치법규 361M, 판례 317M, 헌재 70M, 해석례 18M, 재결례 111M). 전체 수집 요청 454,305건, 측정 속도(약 1.09 s/req)로 **5.7일**. 출처: `data/law-ko-source/pilot-stats.json` `extrapolation_full_crawl`.
- **라이선스**: 저작권법 제7조(보호받지 못하는 저작물) 제1호 "헌법·법률·조약·명령·조례 및 규칙", 제2호 "국가 또는 지방자치단체의 고시·공고·훈령 그 밖에 이와 유사한 것", 제3호 "법원의 판결·결정·명령 및 심판이나 행정심판절차 그 밖에 이와 유사한 절차에 의한 의결·결정 등" — https://www.law.go.kr/lsLinkCommonInfo.do?lsJoLnkSeq=1029423769 . 법제처 공동활용 저작권 정책: "제공된 법령정보는 영리 목적의 이용을 포함하여 자유로운 활용이 보장됩니다" — https://open.law.go.kr/LSO/information/guide.do (본 조사에서 직접 확인). 조건: 출처를 법제처로 표기(미표기·타 기관 표기는 저작권법 제137·138조로 형사처벌 경고). **상업·학습·모델 배포 모두 가능.**
- **접근**: Open API. 공식 사용은 회원가입 후 활용신청·승인(1~2일, 운영 단계에서는 활용사례 등록). 파일럿은 공개 데모 키 `test`로 수행했으므로 **본 수집 전 정식 OC 키 발급이 필요**하다. `robots.txt`는 `Allow: /`. 속도 제한은 명시되지 않아 파일럿은 단일 연결 1.05 s 간격을 자율 적용.
- **도메인 근접성**: 최상. 조문 구조(제N조, ①②, 「」, 별표), 행정규칙의 고시·훈령 문체, 재결서·해석례의 질의-회신 문체는 RAG 대상 문서와 동일하다.
- **품질·중복·PII**: 파일럿 정확 중복 0.78%, 근사 중복 0.47%. 자치법규는 표준조례안 기반이라 문서 단위 근사 중복은 낮았으나(동일 제목 타 지자체 비교 11건, 최대 Jaccard 중앙값 0.076) **문단·조문 단위 상용구 반복**이 매우 많을 것이므로 n-gram/문단 단위 중복 제거를 권한다. 판례 29%는 본문 없음(`no_body`)으로 탈락. 판례·재결례는 원천에서 비실명 처리되어 있고, 파일럿 파이프라인이 전화·이메일·연락처 줄을 제거한다. 가공 텍스트를 "공식 원문"처럼 제시하지 말 것(제137조).
- **샘플 관찰**: `data/law-ko-source/docs.jsonl` 첫 레코드 = `law-go-kr:law:284117` "4ㆍ16세월호참사 진상규명 및 안전사회 건설 등을 위한 특별법", 조문 단위로 ` | ` 결합, `license: KR-Copyright-Act-Art7 (law.go.kr)`, `attribution: 출처: 법제처 국가법령정보센터`.

### 3.2 정책브리핑(korea.kr) 보도자료 (공공누리 제1유형)

- **텍스트**: 각 부처·기관 보도자료가 자동 연계되는 `https://www.korea.kr/briefing/pressReleaseList.do`. 본문은 HWPX > PDF > HWP > 페이지 텍스트 순으로 추출(파일럿).
- **규모(추정)**: 2015년 이후 목록 296,480건, 파일럿 148건 수집 → 137건 유지(공공누리 1유형 표시 통과율 97.97%), 평균 1,949 토큰/건, 1.60자/토큰. 외삽 **약 535M 토큰**, 원시 첨부 약 429GB, 요청 587,822건, 측정 1.88 s/req로 **약 12.8일**. 출처: `data/korea-kr-briefing-source/pilot-stats.json`.
- **라이선스**: 저작권 정책 https://www.korea.kr/guide/copyRight.do — "저작권법 제24조의2(공공저작물의 자유이용)에 따라 별도의 이용 허락 없이 자유이용이 가능 … 공공누리 제1유형을 부착하여 개방 … 공공누리 표시가 부착된 저작물인지를 확인한 이후에 **텍스트는** 자유롭게 이용", 사진·이미지는 제외(연합뉴스·통로이미지 저작권). 공공누리 제1유형 정의 https://www.kogl.or.kr/info/license.do — "출처표시, 상업적·비상업적 이용가능, 변형 등 2차적 저작물 작성 가능". **텍스트 한정, 건별 공공누리 표시 확인 조건으로 상업·학습·모델 배포 가능.** 출처표시 예시 형식도 같은 페이지에 있다.
- **접근**: 웹 크롤(`robots.txt` `Allow: /`). API 없음. 첨부 파싱에 `hwp5txt`, HWPX XML, `pdftotext` 필요(파일럿 도구).
- **도메인 근접성**: 최상(□ ○ - 글머리 구조, "~라고 밝혔다", 담당부서 헤더).
- **품질·중복·PII**: 파일럿 정확·근사 중복 0%. 보도자료 헤더의 담당자 성명·전화·이메일은 제거해야 한다(파일럿 파이프라인이 제거). PDF 추출 시 표·줄바꿈 깨짐이 잦아 샘플 첫 레코드(`korea-kr-press-156066837`, 국토교통부 BMW 리콜)에도 `| ( | | ) |` 같은 표 잔재가 남아 있었다. 정책뉴스(`policyNews`)는 보도자료를 재가공한 기사이므로 둘 다 넣으면 근사 중복이 커진다 — 보도자료만 권장.

### 3.3 국회회의록

- **텍스트**: 본회의·위원회·국정감사 속기록. 역대 회의록 50,523건(국회회의록 시스템 메인 통계, 검색 결과 스니펫 https://record.assembly.go.kr/assembly/mnts/main.do ; 본 조사 직접 접속은 HTTP 400으로 차단됨).
- **라이선스**: 국회회의록 저작권 정책 "저작권법 제24조의2에 따라 … 국회사무처가 저작재산권의 전부를 보유한 저작물의 경우에는 이용허락 없이 자유이용 … 공공누리 제1유형을 부착" (https://record.assembly.go.kr/assembly/mnts/summary/lawMntsCnts.do 하단, 검색 스니펫). 공공데이터포털 "국회 국회사무처_본회의 회의록" API(https://www.data.go.kr/data/15126007/openapi.do , 본 조사 직접 확인): 무료, **이용허락범위 제한 없음**, 개발단계 자동승인/운영단계 심의승인, API 유형 **LINK**(XML).
- **규모**: 미측정. 본회의 1회분이 수십만 자인 경우가 흔해 수백 M 토큰 이상일 가능성이 있다(추정, 근거 없음 → 파일럿 필요).
- **접근 문제**: data.go.kr의 LINK형 API가 회의록 **본문**을 주는지, 열린국회정보(open.assembly.go.kr) API 목록에 본문 엔드포인트가 있는지 본 조사에서 확인하지 못했다(open.assembly.go.kr API 목록·약관 페이지 모두 HTTP 400). 회의록 시스템은 PDF/HTML 열람 형태다.
- **도메인**: 구어체 정책 담화 + 정부 답변. 법령·공문과는 문체가 다르지만 기관명·정책 어휘가 풍부하다. 의원·공무원 실명은 공인 발언이므로 PII 위험은 낮다.

### 3.4 공공데이터포털(data.go.kr)

- 정책: https://www.data.go.kr/ugs/selectPortalPolicyView.do — "저작물이 포함된 공공데이터를 제공하는 경우 공공누리 유형을 부착", 데이터셋별 "이용허락범위"에 공공누리 유형이 표시된다. 본 조사에서 확인한 예: "문화체육관광부 국립국어원_말뭉치 통계 목록"(https://www.data.go.kr/data/15123464/fileData.do ) 제1유형, "행정안전부_대한민국 공공서비스(혜택) 정보"(https://www.data.go.kr/data/15113968/openapi.do ) "이용허락범위 제한 없음", REST JSON, 개발계정 트래픽 10,000/일.
- 텍스트 벌크로 쓸 만한 대형 데이터셋은 본 조사에서 발견하지 못했다. 관보 관련 API(목록·인사·상훈·재산공개)는 메타데이터 수준이고 고시·공고 전문은 아니다(https://www.data.go.kr/data/15109157/openapi.do 등). 공공서비스(혜택) 정보처럼 **행정 안내문 문체**의 소규모 데이터는 보강용으로 가치가 있다.
- 접근: 파일데이터는 로그인 없이 다운로드, API는 회원가입·활용신청(대부분 자동승인).

### 3.5 한국어 위키백과 / 위키문헌 / 위키뉴스

- **위키백과 전체**: 로컬 `data/wiki-ko-full-source/`(wikimedia/wikipedia 20231101.ko 3개 parquet). 파일럿 측정: 379,068문서 유지, 547,316,678자, **307.7M 토큰(추정)**, 1.78자/토큰, 정확 중복 2e-5. 라이선스 CC BY-SA 4.0 (https://dumps.wikimedia.org/legal.html ). 현재 코퍼스 v2(70,777 train 문서)는 첫 번째 shard에서만 뽑았으므로 전체 덤프 사용 시 기존 문서와의 중복을 split 해시로 그대로 이어받아야 한다.
- **위키문헌**: 사이트 통계(https://ko.wikisource.org/w/api.php?action=query&meta=siteinfo&siprop=statistics|rightsinfo&format=json ) articles 25,603, `cirrussearch-article-words` 86,804,309, 라이선스 CC BY-SA 4.0. 덤프 `kowikisource-latest-pages-articles.xml.bz2` 148MB (https://dumps.wikimedia.org/kowikisource/latest/ ). 조선왕조실록 번역·한문 고전·옛 법령 비중이 커 현대 공문 도메인과는 거리가 있으나, 헌법·법률 원문·판결문·공문서 사본 등 제7조 대상 문서도 섞여 있다. 선별(한자 비율·연대 필터) 후 소량 사용.
- **위키뉴스**: 본 조사에서 덤프(1.35MB)를 `/tmp/trid-survey/kowikinews.xml.bz2`로 받아 파싱: page 3,273 / ns0 917 / 리다이렉트 제외 851문서 / 1,467,079자(≈0.9M 토큰, 추정). 라이선스 CC BY 4.0. 규모가 무의미하게 작다.

### 3.6 FineWeb-2 `kor_Hang`

- **규모**: 언어 분포 CSV(https://github.com/huggingface/fineweb-2/blob/main/fineweb2-language-distribution.csv ) `kor_Hang,train`: 48,613,120,582 단어, 60,874,355 문서, 229,164,863,492 UTF-8 바이트(213.43GB), parquet 98.50GB. `kor_Hang_removed`(필터 탈락분)도 별도 제공.
- **라이선스**: 데이터셋 카드 `license: odc-by` (https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 ; API 메타데이터로 확인). ODC-By는 **데이터베이스 패키징**에 대한 허락이고, 원문 텍스트는 Common Crawl 이용약관과 각 사이트 저작권에 따른다. 한국 저작권법에는 TDM 면책이 없어 상업 모델 학습 근거가 약하다 [INFERENCE: 법률 자문 필요].
- **샘플 관찰**(datasets-server `rows?config=kor_Hang&offset=0&length=3`): (1) 영문 사진 설명 + 한국어 조류 설명이 섞인 페이지, `minhash_cluster_size 3`; (2) 다음 Q&A 페이지 — 사용자 ID가 `tjdg*****`로 원천 마스킹, 짧은 게시판 잡음; (3) **ko.wikipedia.org 원문**(게임 오버 문서, `minhash_cluster_size 59`) — 위키백과와 중복. 필드: `text,id,dump,url,date,file_path,language,language_score,language_script,minhash_cluster_size,top_langs`. `url` 필드가 있으므로 `.go.kr`/`.or.kr` 도메인 필터로 공공 웹 하위집합을 뽑는 실험이 가능하다(라이선스 문제는 그대로).
- **도메인**: 하. 품질은 세 웹 크롤 중 가장 공격적으로 필터링돼 있다.

### 3.7 HPLT 2.0 / 3.0 `kor_Hang`

- **규모**: v2 cleaned 카드 표 `kor_Hang`(열: segments / wcwords / chars / docs): 세그먼트 1.36e+09, 공백 분리 단어 1.97e+10, 문자 8.92e+10, 문서 3.89e+07 (https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned ). v3.0 카탈로그(https://hplt-project.org/datasets/v3.0 ): Korean (kor_Hang) Docs 74.79M, Tokens 97.58B(Gemma 3 토크나이저), Chars 164.31B, Segments 2.34B, 정렬 다운로드 150.61GB, 웹 레지스터 라벨 제공. 카드가 v2 대신 v3 사용을 권고.
- **라이선스**: 카드 `license: cc0-1.0`이지만 Terms of Use는 "We do not own any of the text … We license the actual packaging … under CC0", 이용자가 EU DSM 지침·GDPR 등 준수 책임(https://hplt-project.org/datasets/v3.0 Terms of Use). 즉 FineWeb-2와 같은 회색지대.
- **샘플 관찰**: `https://data.hplt-project.org/three/sorted/kor_Hang/10_1.jsonl.zst` 첫 3MB를 `/tmp/trid-survey/hplt3_kor_sample.jsonl.zst`로 받아 두 문서를 확인. 최상위 품질 구간(WDS=10)임에도 (1) `dbwlgns20.xyz` 들깨 효능 블로그(스팸성 애드센스 블로그, cluster_size 19), (2) `lawstudies.kr` 기계번역 투 광고 문구("가나 학사 대학을 찾아 자신에게 가장 맞는 사람을 찾기")였다. 메타데이터는 `u, ts, crawl_id, lang[3], prob, cluster_size, filter, pii[], doc_scores[10], web-register` — PII 오프셋과 레지스터 라벨은 유용하지만 품질 점수는 한국어에 대해 신뢰하기 어렵다.
- **접근**: HF에 호스팅되지 않음, `wget -i kor_Hang.map`으로 HTTP 벌크(샤드 단위). 150GB라 전체 다운로드는 본 과제 범위 밖.

### 3.8 CulturaX `ko` / mC4 / OSCAR

- **규모**: README 표 22행 `ko | Korean | 20,557,310 문서 | 24,765,448,392 토큰 | 0.39%` (https://huggingface.co/datasets/uonlp/CulturaX/blob/main/README.md ). mC4 3.1.0 + OSCAR 20.19~23.01 통합·정제본이므로 mC4 `ko`는 별도로 쓸 이유가 없다.
- **라이선스**: README "The licence terms for CulturaX strictly follows those of mC4 and OSCAR" — mC4(allenai/c4) `license: odc-by` (https://huggingface.co/datasets/allenai/c4 ), OSCAR-2301 `license: cc0-1.0`이나 **현재 수동 승인 gated·접근 일시 중단** 상태이며 gating 문구에 "메타데이터·주석만 CC0, 본문은 Common Crawl 크롤 데이터로 저작권 없음 … 프랑스에서는 TDM/연구 예외 밖 사용은 금지" 명시(https://huggingface.co/datasets/oscar-corpus/OSCAR-2301 ). README는 "personal and sensitive information might still contain" 경고. datasets-server 미리보기는 401(인증 필요)이라 샘플 확인 못 함.
- **판정**: FineWeb-2보다 라이선스 상태가 더 불리하다(OSCAR 승계). 플래그.

### 3.9 국립국어원 모두의 말뭉치

- **텍스트**: 신문(최근 10년), 문어(서적 20,188종), 구어·메신저, 웹(블로그·게시판 210만 건), 각종 분석 말뭉치. 2020-08 최초 공개분 13종 18억 어절(신아일보 2020-08-25 https://www.shinailbo.co.kr/news/articleView.html?idxno=1313074 ; 국어원 보도자료 재인용). 말뭉치별 어절 수는 공공데이터포털 "말뭉치 통계 목록"(335행 CSV, 제1유형)에서 받을 수 있으나 본 조사에서는 다운로드 링크가 JS 폼이라 수치를 가져오지 못했다.
- **라이선스**(FAQ https://kli.korean.go.kr/boards/faqList.do , 직접 확인): "국어와 언어 정보 처리 분야에 응용하기 위한 연구 및 기술 개발용으로 활용할 수 있으며, **승인된 이용 목적에 한하여**" 사용; "말뭉치를 활용한 결과물을 외부에 공개하고자 할 때에는 … 원문 일부 또는 전부가 포함되어서는 안 되며, **공개 전 국립국어원의 사전 승인**"; "제3자에게 양도, 대여, 이용 허락, 담보 제공할 수 없"음; 이용 기간 약정. 서울말 낭독체·세종 말뭉치는 연구용만. 상업 R&D 자체는 금지되지 않으나 **모델 공개(배포)마다 사전 승인**이 필요하고, 원문 암기·재생 우려(결과물에 원문 포함 금지)가 LM과 충돌할 수 있다.
- **접근**: 본인인증 회원가입 → 말뭉치별 신청(사유·과제명·기관·기간) → 5근무일 내 승인 → 약정서 → 다운로드(zip, cp949 파일명). 조사 시점 사이트에 "일부 기능 장애로 신청 불가" 공지.
- **판정**: 품질은 높지만 절차·승인·배포 제약 때문에 본 믹스 제외, 플래그.

### 3.10 AI Hub

- **대상 데이터셋**: "행정 문서 대상 기계독해 데이터"(https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=569 , 직접 확인): 원시데이터 행정문서 약 60만 건(HWP/PDF, 출처: 공공데이터포털·부처·지자체·공공기관), 라벨 411,840건, 지문 200~2,000자, 주제 공공행정·국토관리·환경 등. 그 밖에 "법률/규정(판결서, 약관 등) 텍스트 분석 데이터"(dataSetSn=580), "공공행정문서 OCR" 등.
- **라이선스**(이용정책 https://www.aihub.or.kr/intrcn/guid/usagepolicy.do?currMenu=151&topMenu=105 , 직접 확인): "영리적·비영리적 연구·개발 목적으로 활용 가능"; "반드시 한국지능정보사회진흥원의 사업결과임을 밝혀야 하며 2차적 저작물에도 동일"; "국외 소재 법인·개인 이용 및 국외 반출은 별도 합의"; "**인공지능 학습모델의 학습용으로만** 사용"; 승인 없는 제3자 열람·제공·양도·판매 금지; "데이터셋의 판매 등 상업적 이용"은 별도 협의. 데이터 페이지에 "**내국인만 데이터 신청이 가능**". 학습된 모델의 배포에 대한 명시 조항은 없다 — 데이터 재배포 금지와 "학습용" 조항으로 보아 모델 배포는 허용되는 것으로 읽히지만 [INFERENCE] 확인 필요. 모델을 국외 서비스에 배치하는 것이 "국외 반출"에 해당하는지도 불명.
- **판정**: 도메인은 최상이나 로그인·신청·국적 조건과 조항 해석 리스크가 있어 플래그. 원시 행정문서 60만 건은 애초에 공공데이터포털·부처 공개 문서이므로 원천에서 직접 수집하는 편이 라이선스가 깨끗하다.

### 3.11 KorQuAD / KLUE

- KorQuAD 1.0·2.0: "CC BY-ND 2.0 KR" (https://korquad.github.io/ ). 변경 금지(ND)이므로 학습용 가공·혼합이 2차적 저작물 작성에 해당할 소지가 있다. 사전학습 믹스에서 제외.
- KLUE: CC BY-SA 4.0 (https://github.com/KLUE-benchmark/KLUE , https://huggingface.co/datasets/klue/klue ). 상업 가능하나 ShareAlike가 모델에 전이되는지는 해석이 갈린다. 이미 `data/klue-mrc-source/`로 SFT·평가에 쓰고 있으며, 사전학습 규모(수십 M 토큰 미만)에 의미가 없어 사전학습에서는 제외.

### 3.12 법원 판결서 인터넷열람

- 범위: "2013년 1월 1일 이후 확정된 형사 사건 판결서와 2015년 1월 1일 이후 확정되거나 2023년 1월 1일 이후 선고된 민사·행정·특허 사건 판결서"를 임의어 검색으로 열람, 개인정보 비실명 처리(부산지법 안내 https://busan.scourt.go.kr/common/wcd/wcd_desc.jsp ; 대법원 포털 안내 페이지는 본 조사 접속 시 웹방화벽에 차단됨). 수수료 결제 절차가 있다(https://www.scourt.go.kr/portal/information/finalruling/guide/guide_02.jsp "수수료를 결제한 경우" 문구).
- 판정: 텍스트는 제7조 비보호 저작물이지만 건별 유료 열람이라 벌크 수집 수단이 없다. 법제처 판례 API(3.1, 대법원·하급심 주요 판례 175,279건)로 대체.

### 3.13 나무위키

- 라이선스: "이 저작물은 CC BY-NC-SA 2.0 KR에 따라 이용할 수 있습니다"(https://namu.wiki/w/이용약관 ; HF 미러 `heegyu/namuwiki` `license: cc-by-nc-sa-2.0`, 2022-03-01 덤프 867,024행 3GB https://huggingface.co/datasets/heegyu/namuwiki ). **비영리(NC)** 조건이라 상업화 가능 모델 학습에 쓸 수 없다. 제외.

### 3.14 추가 후보(간단 메모)

- **대한민국 관보**(고시·공고 전문): 저작권법 제7조 제2호 대상이나 공공데이터포털의 관보 API는 목록·인사·상훈 메타뿐이다. 전자관보(gwanbo.go.kr) PDF 크롤 가능성은 미조사.
- **각 부처·지자체 홈페이지 고시·공고/보도자료**: 공공누리 표시가 기관·게시물별로 다르다(제2·4유형이면 상업 불가). korea.kr 보도자료로 대부분 커버되므로 후순위.

---

## 4. 제안 믹스 (약 1.0B 토큰, 본 프로젝트 토크나이저 기준)

라이선스가 명확한 소스만 포함. 우선순위 순.

| 순위 | 소스 | 토큰 | 비율 | 비고 |
|---|---|---|---|---|
| 1 | 정책브리핑 보도자료 (KOGL 1, 텍스트) | 250M | 25% | 가용 ~535M 중 연도 균등 샘플. HWPX/PDF 우선, 헤더·연락처 제거, 정책뉴스는 제외 |
| 2 | 법제처 행정규칙 (훈령·예규·고시) | 150M | 15% | 가용 ~171M, 거의 전량. 별표·서식 텍스트 포함 |
| 3 | 법제처 판례 | 130M | 13% | 가용 ~317M 중 본문 있는 건 위주, 연도·법원 분산 샘플 |
| 4 | 법제처 자치법규 | 120M | 12% | 가용 ~361M. 지자체·제목 균등 샘플 + 문단 단위 상용구 중복 제거(표준조례 반복 억제) |
| 5 | 한국어 위키백과 전체 | 200M | 20% | 가용 ~308M. 일반 지식·고유명사·문장 다양성 담당. 기존 valid/test 해시 버킷 유지 |
| 6 | 법제처 현행 법령 | 60M | 6% | 전량(~61M). RAG 핵심 참조 대상 |
| 7 | 법제처 헌재결정례 + 법령해석례 + 행정심판재결례 | 60M | 6% | 가용 ~199M 중 재결례·해석례 우선(질의-회신 문체) |
| 8 | 국회회의록 | 30M | 3% | 본문 API 경로 확인 후. 확보 실패 시 보도자료로 대체 |
| 합계 | | 1,000M | 100% | |

보조 계획
- 공공데이터포털 소형 텍스트(공공서비스 안내 등)와 위키문헌 현대문 선별분은 합쳐 10M 이내로, 위 8번 미확보 시 메워 넣는다.
- 수집 소요: 법제처 전체 약 6일 + korea.kr 약 13일(병렬 실행 시 2주), 디스크는 korea.kr 첨부 원본 약 430GB(필요 시 파싱 후 원본 삭제).
- 중복 제거: 소스 간 교차 근사 중복(법령 ↔ 판례 인용문 ↔ 보도자료 법령 요약, 위키백과 ↔ 보도자료 기관 설명)을 혼합 단계에서 한 번 더 돌린다(기존 `scripts/prepare_korean_continuation.py`의 MinHash 경로 재사용).
- 홀드아웃: 소스별 5%/5% valid/test를 해시 버킷으로 분리해 도메인별 perplexity를 따로 본다.

### 플래그 소스 (본 믹스 제외, 법률 자문 또는 절차 완료 후 선택적 추가)

| 소스 | 이유 | 조건부 사용 아이디어 |
|---|---|---|
| FineWeb-2 `kor_Hang` | 패키징만 ODC-By, 본문 저작권 제3자, 국내 TDM 면책 부재 | `url`이 `.go.kr`인 문서만 뽑아 공공저작물 추정 하위집합(≤100M)으로 실험 — 기관별 공공누리 표시가 없어 여전히 회색 |
| HPLT 3.0 `kor_Hang` | 동일 + 샘플 품질 낮음(최상위 구간이 스팸·기계번역) | 레지스터 라벨로 뉴스·공공 문서만 필터해도 라이선스 문제 동일 |
| CulturaX `ko` | OSCAR 승계 약관, OSCAR 접근 중단 | 사용 비권장 |
| 국립국어원 모두의 말뭉치 | 승인 목적 한정, 모델 공개 전 사전 승인, 양도 금지 | 신문·문어 말뭉치를 신청해 두고, 승인·공개 심사 비용을 감수할 때만 추가 |
| AI Hub 행정문서 MRC·법률 텍스트 | 내국인 신청, 재배포·국외 반출 금지, 모델 배포 조항 부재 | 원천(공공데이터포털·부처)에서 직접 수집하는 쪽을 권장 |
| KorQuAD | CC BY-ND | 사전학습 제외 |
| KLUE | CC BY-SA(이미 SFT/평가에 사용) | 사전학습 제외 |
| 나무위키 | CC BY-NC-SA 2.0 KR | 제외 |
| 법원 판결서 열람 | 건별 유료, 벌크 불가 | 법제처 판례로 대체 |

---

## 5. 토크나이저에 대한 권고

- 현재 8192 BPE는 위키백과로만 학습됐다. 파일럿 측정치는 위키 1.78자/토큰, 법령 1.67~1.72, 보도자료 1.60으로, **기관 문서에서 6~10% 더 많은 토큰**을 쓴다(= 같은 1024 컨텍스트에 더 적은 내용이 들어가고, 조문 기호 `ㆍ ① 「」 □ ○ ※`와 "제N조의M", "시행령", "별표" 같은 고빈도 조각이 다자 토큰으로 묶여 있지 않을 가능성이 크다).
- **권고: 새 믹스(위 비율로 샘플링한 50~100M 토큰 분량)로 토크나이저를 재학습**하라. 어휘 크기는 8192 유지가 기본이되, 1B 토큰·91M 파라미터 규모에서는 12k~16k로 늘리는 것도 검토할 만하다(임베딩 파라미터가 늘어나므로 `reports/mac-model-capacity-v1.json`의 메모리 한계와 함께 판단). 재학습 시 기존 체크포인트·SFT 데이터 토큰화 산출물(`data/korean-foundation-tokens-v2/` 등)과 호환되지 않으므로, 차기 사전학습이 처음부터(from scratch) 가는 경우에만 바꾸고, 바이트 폴백·NFC 정규화·조문 기호 보존 여부를 `reports/korean-tokenizer-baseline.json`과 같은 방식으로 다시 측정해야 한다. 본 조사에서는 실행하지 않았다.

---

## 6. 미확인·차단 항목

1. **법제처 Open API 정식 인증키**: 파일럿은 데모 키 `test`. 본 수집은 회원가입·활용신청·승인(1~2일) 필요. 트래픽 상한이 문서화돼 있지 않다.
2. **국회회의록 본문 벌크 경로**: data.go.kr API는 LINK형이며 본문 제공 여부 미확인; open.assembly.go.kr API 목록·약관 페이지와 record.assembly.go.kr 메인은 본 환경에서 HTTP 400. 토큰 규모도 미측정.
3. **국립국어원 말뭉치별 어절 수**: 공공데이터포털 CSV(335행)는 JS 폼 다운로드라 미취득. 사이트 자체도 조사 시점 "신청 불가" 장애 공지.
4. **AI Hub**: 로그인·내국인 신청 필요, 모델 배포·국외 서비스 배치에 대한 조항 해석 미확정.
5. **CulturaX/OSCAR**: datasets-server 401, OSCAR gated. 샘플 미확인.
6. **HPLT 2.0** 샘플: datasets-server 500. (v3는 HTTP로 직접 샘플 확인함.)
7. **법원 판결서 인터넷열람 수수료 금액·약관**: 대법원 포털 페이지 웹방화벽 차단으로 1차 확인 못 함(지방법원 안내 페이지로 범위만 확인).
8. **웹 크롤(FineWeb-2/HPLT/CulturaX) 상업 학습의 국내 적법성**: 저작권법 TDM 면책 조항 부재는 본 조사자의 이해이며 [INFERENCE], 법률 자문 필요.
9. **위키문헌 현대 공문서 비중**: 86.8M 단어 중 제7조 대상 문서 비율 미측정.
10. **관보 전문** 수집 경로(전자관보 PDF) 미조사.

---

## 7. 본 조사에서 확보한 샘플 (/tmp, 저장소 미반영)

- `/tmp/trid-survey/kowikinews.xml.bz2` (1,346,377 B): 위키뉴스 전체 덤프. 파싱 결과 851 비리다이렉트 문서, 1,467,079자. 미디어위키 마크업(`{{날짜|…}}`, `[[파일:…]]`, `== 출처 ==`) 제거 필요.
- `/tmp/trid-survey/hplt3_kor_sample.jsonl.zst` (3,000,001 B, 범위 요청으로 잘린 샤드 앞부분): HPLT 3.0 `kor_Hang/10_1.jsonl.zst` 선두 2문서 확인(§3.7).
- FineWeb-2 `kor_Hang` 3행: datasets-server 응답을 직접 열람(§3.6), 파일 저장 없음.
- 법제처·korea.kr·위키백과 샘플은 동료 에이전트가 같은 날 `data/law-ko-source/`, `data/korea-kr-briefing-source/`, `data/wiki-ko-full-source/`에 적재한 파일럿을 열람했다(§3.1, §3.2, §3.5).

## 8. 인용 출처 목록

- 저작권법 제7조: https://www.law.go.kr/lsLinkCommonInfo.do?lsJoLnkSeq=1029423769
- 법제처 공동활용 이용안내·저작권 정책: https://open.law.go.kr/LSO/information/guide.do
- 정책브리핑 저작권 정책: https://www.korea.kr/guide/copyRight.do
- 공공누리 유형안내: https://www.kogl.or.kr/info/license.do
- 공공데이터포털 이용정책: https://www.data.go.kr/ugs/selectPortalPolicyView.do
- 국회 본회의 회의록 API: https://www.data.go.kr/data/15126007/openapi.do
- 공공서비스(혜택) 정보 API: https://www.data.go.kr/data/15113968/openapi.do
- 국립국어원 말뭉치 통계 목록: https://www.data.go.kr/data/15123464/fileData.do
- 국립국어원 모두의 말뭉치 FAQ: https://kli.korean.go.kr/boards/faqList.do
- 국립국어원 18억 어절 공개 보도(2020-08-25): https://www.shinailbo.co.kr/news/articleView.html?idxno=1313074
- AI Hub 이용정책: https://www.aihub.or.kr/intrcn/guid/usagepolicy.do?currMenu=151&topMenu=105
- AI Hub 행정 문서 대상 기계독해 데이터: https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=569
- 위키미디어 덤프 법적 고지: https://dumps.wikimedia.org/legal.html
- 위키문헌 통계·라이선스 API: https://ko.wikisource.org/w/api.php?action=query&meta=siteinfo&siprop=statistics|rightsinfo&format=json
- 위키뉴스 통계·라이선스 API: https://ko.wikinews.org/w/api.php?action=query&meta=siteinfo&siprop=statistics|rightsinfo&format=json
- 위키문헌·위키뉴스 덤프 목록: https://dumps.wikimedia.org/kowikisource/latest/ , https://dumps.wikimedia.org/kowikinews/latest/
- FineWeb-2 언어 분포 CSV: https://github.com/huggingface/fineweb-2/blob/main/fineweb2-language-distribution.csv
- FineWeb-2 데이터셋 카드: https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
- HPLT 2.0 cleaned 카드: https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned
- HPLT 3.0 카탈로그·Terms of Use: https://hplt-project.org/datasets/v3.0
- CulturaX README: https://huggingface.co/datasets/uonlp/CulturaX/blob/main/README.md
- allenai/c4 (mC4) 카드: https://huggingface.co/datasets/allenai/c4
- OSCAR-2301 카드(gated 문구): https://huggingface.co/datasets/oscar-corpus/OSCAR-2301
- KorQuAD: https://korquad.github.io/
- KLUE: https://github.com/KLUE-benchmark/KLUE , https://huggingface.co/datasets/klue/klue
- 판결서 인터넷열람 안내(부산지방법원): https://busan.scourt.go.kr/common/wcd/wcd_desc.jsp
- 판결서 인터넷열람 서비스 이용안내(대법원): https://www.scourt.go.kr/portal/information/finalruling/guide/guide_02.jsp
- 나무위키 이용약관: https://namu.wiki/w/이용약관 ; HF 미러: https://huggingface.co/datasets/heegyu/namuwiki
