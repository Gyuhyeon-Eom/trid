# Actual-law structure validation v1

Development analysis of 3,000 actual law.go.kr documents: first 1,000 complete unique rows for each of law, admrul, ordin at the initial file byte boundary. Live collection was read-only; complete-row snapshot is `/tmp/trid-law-structure-3000.jsonl`. No held-out claim.

Snapshot SHA-256: `d7992f96f7ae500c0190048d51a14d93275cfb98c265045281dbce35eaad413c`.

## Measured results

| Metric | Before | After |
|---|---:|---:|
| documents | 3000 | 3000 |
| detected_regulations | 2684 | 2684 |
| passages | 109119 | 109380 |
| resolved_local_edges | 72102 | 55026 |
| external_mentions | 25235 | 26659 |
| local_article_mentions | 160084 | 155932 |
| unresolved_article_mentions | 25713 | 23592 |
| invalid_spans | 0 | 0 |
| over_limit | 0 | 0 |
| exceptions | 0 | 0 |

Detected by kind: law 997/1000; admrul 687/1000; ordin 1000/1000. This is heuristic coverage, NOT precision/recall against independently labeled document classes. Many administrative interpretations are prose.

Passage character lengths: before min/median/p95/max = 2/234/540/550; after = 2/233/540/550. All 109,380 final passages are exact source slices and within 550 characters; zero parser exceptions.

`external_mentions` counts quoted-title citation chains, not unique target documents or resolved cross-document edges. `local_article_mentions` includes article headings and unbound external aliases; `unresolved_article_mentions` means the written article label is absent from parsed paths. It is a diagnostic proxy, NOT reference-resolution accuracy. Fewer local edges reflect removal of external citations incorrectly resolved locally, not a measured loss of recall.

## Corrections

- Recognize official `■ document title [별표 N]` / form headings and inserted annex numbers.
- Preserve line-start lists beginning after repealed entries and inserted `2의2.` entries; do not require uninterrupted 1,2,3 sequence.
- Keep explicit external article/subarticle chains (including repeated paragraph suffixes and parenthetical alias declarations) externally scoped. Unbound 법/영/규칙/조례 references are not guessed as local; explicit 이 법 remains local.
- Resolve a specific paragraph reference within the same subarticle instead of discarding all same-article references.

## Thirty diagnostic failure candidates inspected

These are 30 inspected occurrences, not 30 distinct documents or 30 confirmed bugs. Controls and remaining limitations are explicitly retained. Source excerpts came from pre-fix diagnostics; row IDs and official URLs allow retrieval. This review is engineering inspection, not legal expert adjudication.

### 1. unresolved_article — law-go-kr:law:253527

Source: http://www.law.go.kr/DRF/lawService.do?target=law&MST=253527&type=HTML&efYd=20230808

> 다.
> 제3조의2(벌칙 적용에서 공무원 의제) 위원회의 위원 중 공무원이 아닌 사람은 「형법」 제127조 및 제129조부터 제132조까지의 규정을 적용할 때에는 공무원으로 본다.
> 제5조(의료지원금)
> ① 10ㆍ27법난으로 인하여 부상 및 장애를 입은 사람 중에서 이 법 시행 당시 그 부상 및 장애로 인하여 계속적인 치료가 필요하거나 

External chain tail was misclassified as local; chain is now masked and exposed as external endpoints.

### 2. unresolved_article — law-go-kr:law:253527

Source: http://www.law.go.kr/DRF/lawService.do?target=law&MST=253527&type=HTML&efYd=20230808

> (벌칙 적용에서 공무원 의제) 위원회의 위원 중 공무원이 아닌 사람은 「형법」 제127조 및 제129조부터 제132조까지의 규정을 적용할 때에는 공무원으로 본다.
> 제5조(의료지원금)
> ① 10ㆍ27법난으로 인하여 부상 및 장애를 입은 사람 중에서 이 법 시행 당시 그 부상 및 장애로 인하여 계속적인 치료가 필요하거나 상시 간병 또는

External range endpoint was misclassified as local; fixed for explicit endpoints (not expansion of intermediate articles).

### 3. unresolved_article — law-go-kr:law:277805

Source: http://www.law.go.kr/DRF/lawService.do?target=law&MST=277805&type=HTML&efYd=20251001

>  있다. 다만, 공무원인 위원이 그 소관 업무와 직접 관련되어 참석하는 경우에는 그러하지 아니하다.
> ② 법 제7조에 따라 위원회 등에 출석한 증인 또는 참고인에 대하여는 예산의 범위에서 여비 및 실비를 지급할 수 있다.
> 제6조(명예회복 관련 지원에 관한 사항) 법 제3조제2항제5호에서 "대통령령으로 정하는 지원에 관한 사항"이

Unbound 법 alias: suppress local false edge; target remains unresolved without document-level alias binding.

### 4. unresolved_article — law-go-kr:law:277805

Source: http://www.law.go.kr/DRF/lawService.do?target=law&MST=277805&type=HTML&efYd=20251001

> 료지원금 지급신청을 한 자 중에서 자체조사를 통하여 신청내용에 대한 사실 확인이 불가능한 자에 대하여는 법 제7조제1항에 따라 그 내용을 국방부장관, 문화체육관광부장관 또는 경찰청장 등에게 통보하여 사실 여부 확인을 요청할 수 있다.
> ② 제1항에 따라 확인 요청을 받은 국방부장관, 문화체육관광부장관 또는 경찰청장 등은 요청을 받은 

Unbound 법 alias: suppress local false edge; not treated as successfully resolved external reference.

### 5. unresolved_article — law-go-kr:law:277805

Source: http://www.law.go.kr/DRF/lawService.do?target=law&MST=277805&type=HTML&efYd=20251001

>  수행하기 위하여 불가피한 경우 「개인정보 보호법」 제23조에 따른 건강에 관한 정보 또는 같은 법 시행령 제19조제1호에 따른 주민등록번호가 포함된 자료를 처리할 수 있다.
> 
> 부칙(정부조직 개편 반영을 위한 306개 법령의 일부개정에 관한 대통령령) <제35811호,2025.10.1>
> 
> 이 영은 공포한 날부터 시행한다. 다만,

같은 법 시행령 alias remains unbound; suppress local edge rather than invent target.

### 6. list_review — law-go-kr:law:253527

Source: http://www.law.go.kr/DRF/lawService.do?target=law&MST=253527&type=HTML&efYd=20230808

> 2. "피해자"란 10ㆍ27법난으로 인하여 사망한 사람 또는 부상 및 장애를 입은 사람을 말한다.

Ordinary consecutive list entry; review control, not a confirmed defect.

### 7. list_review — law-go-kr:law:253527

Source: http://www.law.go.kr/DRF/lawService.do?target=law&MST=253527&type=HTML&efYd=20230808

> 3. "피해종교단체"란 10ㆍ27법난으로 인하여 종교적 존엄성과 명예를 훼손당한 대한불교 조계종 및 10ㆍ27법난 당시 피해자가 소속된 사찰을 말한다.

Ordinary consecutive list entry; review control, not a confirmed defect.

### 8. list_review — law-go-kr:law:253527

Source: http://www.law.go.kr/DRF/lawService.do?target=law&MST=253527&type=HTML&efYd=20230808

> 2. 피해자등의 명예회복에 관한 사항

List starts at 2 after repeal: strict count-from-1 lost structure; now accept line-start gaps.

### 9. list_review — law-go-kr:law:253527

Source: http://www.law.go.kr/DRF/lawService.do?target=law&MST=253527&type=HTML&efYd=20230808

> 3. 피해자에 대한 의료지원금 지급에 관한 사항

Continuation of repealed-entry list; monotonic labels retained.

### 10. list_review — law-go-kr:law:253527

Source: http://www.law.go.kr/DRF/lawService.do?target=law&MST=253527&type=HTML&efYd=20230808

> 2. 잘못 지급된 경우

Ordinary second entry; review control, not a confirmed defect.

### 11. subarticle_review — law-go-kr:law:253527

Source: http://www.law.go.kr/DRF/lawService.do?target=law&MST=253527&type=HTML&efYd=20230808

> 제3조의2(벌칙 적용에서 공무원 의제) 위원회의 위원 중 공무원이 아닌 사람은 「형법」 제127조 및 제129조부터 제132조까지의 규정을 적용할 때에는 공무원으로 본다.

Subarticle heading already parsed; defect is external chain in its body, now scoped externally.

### 12. subarticle_review — law-go-kr:law:277805

Source: http://www.law.go.kr/DRF/lawService.do?target=law&MST=277805&type=HTML&efYd=20251001

> 제2조의2(위원의 해촉) 문화체육관광부장관은 제2조제1항제4호에 따른 위원이 다음 각 호의 어느 하나에 해당하는 경우에는 해당 위원을 해촉(解囑)할 수 있다.

Subarticle heading already parsed; preserve exact subarticle path. No heading defect demonstrated.

### 13. subarticle_review — law-go-kr:law:289579

Source: http://www.law.go.kr/DRF/lawService.do?target=law&MST=289579&type=HTML&efYd=20260915

> 제3조의2(2차 가해 방지) 누구든지 다음 각 호의 어느 하나에 해당하는 방법으로 10ㆍ29이태원참사 및 희생자 또는 피해자에 대하여 허위사실을 유포하거나 명예를 훼손하거나 모욕하는 행위(이하 "2차 가해"라 한다)를 하여서는 아니 된다.

Subarticle heading already parsed; review control, not a confirmed defect.

### 14. subarticle_review — law-go-kr:law:289579

Source: http://www.law.go.kr/DRF/lawService.do?target=law&MST=289579&type=HTML&efYd=20260915

> ③ 조사위원회가 제1항제2호에 따라 진술을 청취하는 경우 「형사소송법」 제147조부터 제149조까지 및 제244조의3을 준용한다.

External subarticle at end of range/list now remains in external scope.

### 15. subarticle_review — law-go-kr:law:289579

Source: http://www.law.go.kr/DRF/lawService.do?target=law&MST=289579&type=HTML&efYd=20260915

> ② 제1항의 경우 「형사소송법」 제361조, 제361조의3제1항ㆍ제3항, 제377조 및 제379조제1항ㆍ제4항의 기간은 각각 7일로 한다.

External subarticle with repeated paragraph suffix previously terminated scope early; corrected.

### 16. external_chain_review — law-go-kr:law:253527

Source: http://www.law.go.kr/DRF/lawService.do?target=law&MST=253527&type=HTML&efYd=20230808

> 제3조의2(벌칙 적용에서 공무원 의제) 위원회의 위원 중 공무원이 아닌 사람은 「형법」 제127조 및 제129조부터 제132조까지의 규정을 적용할 때에는 공무원으로 본다.

Explicit external list/range endpoint scope corrected.

### 17. external_chain_review — law-go-kr:law:277805

Source: http://www.law.go.kr/DRF/lawService.do?target=law&MST=277805&type=HTML&efYd=20251001

> ① 「10ㆍ27법난 피해자의 명예회복 등에 관한 법률」(이하 "법"이라 한다) 제3조에 따른 10ㆍ27법난피해자명예회복심의위원회(이하 "위원회"라 한다)는 문화체육관광부장관이 임명하는 제1호부터 제3호까지의 위원과 문화체육관광부장관이 성별을 고려하여 위촉하는 7명 이내의 제4호의 위원으로 구성한다.

Parenthetical alias declaration between quoted title and article now accepted.

### 18. external_chain_review — law-go-kr:law:289579

Source: http://www.law.go.kr/DRF/lawService.do?target=law&MST=289579&type=HTML&efYd=20260915

> ③ 조사위원회가 제1항제2호에 따라 진술을 청취하는 경우 「형사소송법」 제147조부터 제149조까지 및 제244조의3을 준용한다.

External range followed by subarticle now accepted; intermediate range articles not enumerated.

### 19. external_chain_review — law-go-kr:law:289579

Source: http://www.law.go.kr/DRF/lawService.do?target=law&MST=289579&type=HTML&efYd=20260915

> ④ 조사위원회가 제1항제3호에 따라 자료 또는 물건의 제출요구를 하는 경우 「형사소송법」 제110조부터 제112조까지, 제129조부터 제131조까지 및 제133조를 준용하되, 자료 또는 물건의 제출을 거부하는 경우 그 사유를 구체적으로 소명하여야 한다.

Multiple explicit ranges/lists now stay external; only written endpoints returned.

### 20. external_chain_review — law-go-kr:law:289579

Source: http://www.law.go.kr/DRF/lawService.do?target=law&MST=289579&type=HTML&efYd=20260915

> ② 제1항의 경우 「형사소송법」 제361조, 제361조의3제1항ㆍ제3항, 제377조 및 제379조제1항ㆍ제4항의 기간은 각각 7일로 한다.

External article/subarticle list with repeated paragraphs corrected.

### 21. appendix_review — law-go-kr:law:290167

Source: http://www.law.go.kr/DRF/lawService.do?target=law&MST=290167&type=HTML&efYd=20261002

> ② 조사위원회의 위원장(이하 "위원장"이라 한다), 상임위원과 제1항에 따라 조사위원회에 두는 공무원의 직급별 정원은 별표 1과 같다.

Annex reference could not resolve because official ■ title [별표 N] header was missed; header recognized.

### 22. appendix_review — law-go-kr:law:290167

Source: http://www.law.go.kr/DRF/lawService.do?target=law&MST=290167&type=HTML&efYd=20261002

> ③ 조사위원회는 실지조사를 할 때에는 미리 실지조사 대상 기관ㆍ시설 및 단체 등에 일시ㆍ장소 및 목적 등을 알려야 하며, 실지조사를 하는 조사위원회의 위원 및 소속 직원은 그 권한을 표시하는 별표 3의 증표를 관계인에게 보여 주어야 한다.

Annex 3 target now receives structural path through official header recognition.

### 23. appendix_review — law-go-kr:law:290167

Source: http://www.law.go.kr/DRF/lawService.do?target=law&MST=290167&type=HTML&efYd=20261002

> 제44조(과태료의 부과기준) 법 제80조제1항 및 제2항에 따른 과태료의 부과기준은 별표 4와 같다.

Annex 4 target now receives structural path; unbound 법 article is not local.

### 24. appendix_review — law-go-kr:law:290167

Source: http://www.law.go.kr/DRF/lawService.do?target=law&MST=290167&type=HTML&efYd=20261002

> ■ 10ㆍ29이태원참사 피해자 권리보장과 진상규명 및 재발방지를 위한 특별법 시행령 [별표 1]

Official ■ title [별표 1] heading recognized instead of being folded into prior block.

### 25. appendix_review — law-go-kr:law:290167

Source: http://www.law.go.kr/DRF/lawService.do?target=law&MST=290167&type=HTML&efYd=20261002

> ■ 10ㆍ29이태원참사 피해자 권리보장과 진상규명 및 재발방지를 위한 특별법 시행령 [별표 2]

Official ■ title [별표 2] heading recognized instead of being folded into prior block.

### 26. fallback — law-go-kr:law:24144

Source: http://www.law.go.kr/DRF/lawService.do?target=law&MST=24144&type=HTML&efYd=19691204

> 1945년이후종전의규정에의한학교졸업자자격인정령
> 제1조년 5월 31일 현재 종전의 규정에 의한 초급중학교 제3학년 및 중학교 제3학년을 수료한 자는 교육법에 의한 중학교 제3학년 수료자와 동등한 자격이 있는 자로 인정한다.
> 제2조년 5월 31일 현재 종전의 규정에 의한 초급중학교 전수학과를 졸업한 자 및 중학교 제4학년을 수료한 자는 교육법에 의한 중학교 제4학년을 졸업한 자와 동등한 자격이 있는 자로 인정한다.
> 제3조년 3월 31일 현재 종전의 규정에 의한 중학교 및 고급중학교를 졸업한 자는 교육법에 의한 고등학교 졸업자와 동등한 자격이 있는 자로 인정한다.
> 제4조년 5월 31일 현재 종전의 규정에 의한 사범학교를 졸업

Source has 제1조년 etc. malformed/untitled headings; retained paragraph fallback, not repaired source text.

### 27. fallback — law-go-kr:law:268137

Source: http://www.law.go.kr/DRF/lawService.do?target=law&MST=268137&type=HTML&efYd=20250107

> 각급 법원 판사 정원법
> 제1조(각급 법원 판사의 수) 「법원조직법」 제5조제3항 본문에 따른 각급 법원 판사의 수는 3,584명으로 한다.
> 
> 부칙 <제20659호,2025.1.7>
> 
> 이 법은 공포한 날부터 시행한다. 다만, 이 법에 따라 증원되는 판사의 정원 370명 중 90명의 증원은 2025년 1월 1일부터 시행하고, 80명의 증원은 2026년 1월 1일부터 시행하며, 70명의 증원은 2027년 1월 1일부터 시행하고, 70명의 증원은 2028년 1월 1일부터 시행하며, 60명의 증원은 2029년 1월 1일부터 시행한다.

Single-article act falls below two-heading heuristic; known detection false negative, unchanged.

### 28. fallback — law-go-kr:law:37067

Source: http://www.law.go.kr/DRF/lawService.do?target=law&MST=37067&type=HTML&efYd=19810729

> 국가보안유공자상금지급등에관한규정제8조단서의규정에의한무기지정규칙
> 국가보안유공자상금지급등에관한규정제8조단서의 규정에 의한 무기의 종류를 다음과 같이 정한다.
> 1. 총포류
> 2. 화약 기타 폭발물류
> 
> 부칙 <제226호,1981.7.29>
> 
> ①(시행일) 이 규칙은 공포한 날로부터 시행한다.
> 
> ②(볍령의 폐지) 법무부령 제137호 반공법 제15조제1항 단서의 규정에 의한 무기지정의건은 이를 폐지한다.
> 
> ③(경과조치) 이 규칙 시행당시 국가보안유공자심사위원회에 청구된 상금등 청구사건중 종전의 반공법 제15조제1항 단서의 규정에 의한 무기지정의건에 근거하여 청구된 사건은 이 규칙에 의하여 청구된 것으로 본다.

No article headings in operative text; paragraph fallback is reasonable, not a confirmed parser failure.

### 29. fallback — law-go-kr:admrul:2200000098191

Source: http://www.law.go.kr/DRF/lawService.do?target=admrul&ID=2200000098191&type=HTML

> 1인의 소유권등기명의인을 2인으로 경정하는 절차
> 갑으로부터 을·병 명의로의 소유권이전등기신청이 있었음에도 등기공무원의 착오로 을 단독명의로 소유권이전등기가 경료된 경우에는 등기공무원의 직권에 의한 경정등기는 불가능하고 병이 갑·을과 함께 공동신청을 하거나 판결에 기한 단독 신청을 하여야만 그와 같은 등기의 착오를 바로 잡는 경정등기를 할 수 있다.
> 91. 3. 6. 등기 제470호 질의회답

Administrative interpretation is prose, not article-structured regulation; fallback expected.

### 30. fallback — law-go-kr:admrul:2200000093477

Source: http://www.law.go.kr/DRF/lawService.do?target=admrul&ID=2200000093477&type=HTML

> 1인회사의 경우에 절차를 밟지 않은 주주총회결의의 효력
> 주식회사에 있어서 회사가 설립된 이후 총주식을 한사람이 소유하게 된 이른바 1인 회사의 경우에는 그 주주가 유일한 주주로서 주주총회에 출석하면 전원총회로서 성립하고 그 주주의 의사대로 결의가 될 것임이 명백하므로 따로이 총회소집절차가 필요없다 할 것이고 실제로 총회를 개최한 사실이 없었다 하더라도 그 1인 주주에 의하여 의결이 있었던 것으로 주주총회 의사록이 작성되었다면 특별한 사정이 없는 한 그 내용의 결의가 있었던 것으로 볼 수 있어 형식적인 사유에만 의하여 결의가 없었던 것으로 다툴 수는 없다 할 것이다.
> ( 76.4.13.선고74다1755판결)

Administrative interpretation is prose; fallback expected.

Additional observed inserted-list examples: law:290167 `2의2. 2026년 5월 11일 전에…`, law:281977 `1의2. 업무정지처분을 받은 후…`; 1,500 inserted-list lines and 2,582 official ■ annex/form header lines occurred in this snapshot.

## Execution evidence

All runs used CPU niceness 10; BLAS/OpenMP/MKL threads capped at 1. No MPS training or crawlers were stopped or modified. No test suite, build, lint, formatter, commit or push was run.

Initial pre-fix analysis:
```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 .venv/bin/python scripts/validate_structure.py
```
Output: 3,000 documents; 109,119 passages; 72,102 local edges; 0 invalid spans; 0 oversized passages. Wall time 5.37 s.

Exact snapshot restoration was exercised successfully:
```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 .venv/bin/python scripts/validate_structure.py --manifest data/korean-rag-eval-multihop-v1-source/provenance.json --output /tmp/trid-structure-after.json
```
The restored bytes matched the recorded snapshot SHA-256. It requires the original hash-matching source rows; fails explicitly if missing or changed.

Initial structure smoke and exploratory retrieval experiment (before the production benchmark replacement):
```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 .venv/bin/python scripts/validate_structure.py --reuse-snapshot --output /tmp/trid-structure-after.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 .venv/bin/python scripts/benchmark_graph_retrieval.py
```
Output: 3,000 documents; 109,380 passages; 55,026 directed local edges; 100 cases (40 two-edge, 40 one-edge, 20 controls); 0 parser exceptions/span mismatches/limit violations. Combined wall time 17.02 s. The initial regex-tokenizer BM25 indexing took 10.593 s. These are historical exploratory timings, not the production implementation timings. The benchmark script was subsequently replaced with production Index.build/search, keeping the 100 cases frozen; graph-retrieval-v1.json/md contain the production measurement instead.

Production replacement was executed with the same `nice -n 10`/single-thread BLAS command for `scripts/benchmark_graph_retrieval.py`; production Kiwi uses four workers. Wall time: 103.46 s; index build: 64.930 s. It indexed 55,026 local and 8,390 external reference edges. Actual `Index.search` exercised head/reference expansion and public ACL filtering; empty user tags returned zero results. The 100 cases, selected documents, and provenance files were hash-checked unchanged before/after execution. Production multi-evidence all-required/answer-passage recall: BM25@12 0.0625/0.2250 versus graph seeds3/hops2/cap12 0.1625/0.3125; BM25@20 0.1750/0.3375 versus graph seeds5/hops2/cap20 0.2250/0.4625. Graph underfills these caps (mean 4.70 and 7.92 candidates). Independent full production search latency medians were approximately 41–42 ms, rather than the superseded shared-scoring exploratory times.

Added regression tests were not run here; integration owner should run `.venv/bin/python -m unittest discover -s tests -p test_service_structure.py` with the combined integration checks.

## Data and limitations

- Source fixtures contain 100 auto-derived development cases, exact evidence spans/text/hash, document/source provenance, required passage IDs and directed path hops. Selected documents (~1.18 MB) are bundled, not the full ~3,000-document snapshot. Full-corpus manifest (~0.77 MB) records hashes and URLs. Cases (~0.32 MB) are not human gold.
- The replacement graph benchmark invokes production `Index.build` and `Index.search`: Kiwi tokenization, BM25, article heads, local/cross-document references, and ACL filtering. All documents use a public tag, so this is not a restricted-tenant security evaluation. Final candidate caps are applied after the production search API; its complete expansion work is charged to latency. The previous regex/local-edge measurement is superseded.
- Two-hop paths are parser-generated dependencies, not independently proven legal necessity. Terminal evidence passage is an answer-passage proxy, not human-annotated answer text. Exact spans are mechanically checked during the benchmark.
- Title/heading queries and selecting sources with linked passages favor graph retrieval; case-source documents are capped at 30,000 characters and the corpus is order-biased. All data is development after parser tuning.
- Single-article regulations still fall back. Malformed source headings are not repaired. External aliases, intermediate range articles, references split across passages, paragraph-level external targets and semantically ambiguous references remain incomplete. No external resolution accuracy or legal QA accuracy is claimed.
- Thirty inspected occurrences include already-correct subarticle headings, ordinary list controls and expected prose fallbacks. The count must not be presented as thirty independently confirmed defects.
