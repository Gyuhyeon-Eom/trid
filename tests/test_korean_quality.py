import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trid import korean_quality as kq

spec = importlib.util.spec_from_file_location('build_pretraining_mix', ROOT / 'scripts' / 'build_pretraining_mix.py')
mix = importlib.util.module_from_spec(spec); spec.loader.exec_module(mix)

PRESS = """국토교통부, 교통약자 이동편의 실태조사 결과 발표
- 저상버스 보급률 전년 대비 2.1%p 상승, 특별교통수단은 법정대수의 86.4% 확보 -

□ 국토교통부(장관 김철수)는 전국 8개 특별·광역시와 9개 도를 대상으로 실시한 2024년도 교통약자 이동편의 실태조사 결과를 12월 20일 발표했다.
○ 이번 조사는 「교통약자의 이동편의 증진법」 제25조에 따라 매년 실시하는 것으로, 교통수단·여객시설·도로의 이동편의시설 설치 현황과 교통약자의 이용 만족도를 조사한다.
□ 조사 결과 전국의 저상버스 보급률은 38.7%로 전년보다 2.1%p 상승했고, 휠체어 탑승이 가능한 특별교통수단은 법정 운행대수의 86.4%를 확보한 것으로 나타났다.
○ 다만 여객시설의 이동편의시설 기준 적합률은 76.2%로 지역별 편차가 커서, 국토교통부는 적합률이 낮은 지자체에 개선 계획 수립을 요청할 계획이다.
□ 국토교통부 관계자는 "교통약자가 불편 없이 이동할 수 있도록 저상버스 도입과 특별교통수단 확충을 지속적으로 지원하겠다"고 말했다.
○ 조사 결과 전문은 국토교통부 누리집에서 확인할 수 있으며, 지자체별 세부 지표는 내년 1월 중 공개될 예정이다.
"""
SENTENCES = """□ 국토교통부는 전국 17개 시도를 대상으로 실시한 2024년도 교통약자 이동편의 실태조사 결과를 12월 20일 발표했다.
○ 이번 조사는 「교통약자의 이동편의 증진법」 제25조에 따라 매년 실시하며, 교통수단과 여객시설의 설치 현황을 점검한다.
□ 전국의 저상버스 보급률은 38.7%로 전년보다 2.1%p 상승했다.
○ 휠체어 탑승이 가능한 특별교통수단은 법정 운행대수의 86.4%를 확보한 것으로 나타났다.
□ 여객시설의 이동편의시설 기준 적합률은 76.2%로 지역별 편차가 컸다.
○ 국토교통부는 적합률이 낮은 지자체에 개선 계획 수립을 요청할 계획이다.
□ 해양수산부는 연안여객선 운항 안전점검 결과를 같은 날 공개했다.
○ 점검 대상 선박 162척 가운데 11척에서 구명설비 정비 미흡이 확인되어 시정 명령을 내렸다.
□ 교육부는 2025학년도 초등학교 늘봄학교 운영 계획을 확정해 각 시도교육청에 통보했다.
○ 늘봄학교는 정규 수업 전후에 돌봄과 교육 프로그램을 제공하는 제도로, 내년에는 모든 초등학교로 확대된다.
□ 환경부는 수도권 미세먼지 계절관리제 시행 결과 초미세먼지 평균 농도가 전년 대비 8% 감소했다고 밝혔다.
○ 계절관리제 기간에는 5등급 차량 운행 제한과 석탄발전 감축이 함께 시행되었다.
□ 행정안전부는 지방자치단체 재난관리평가 결과를 발표하고 우수 기관 24곳을 선정했다.
○ 평가는 예방, 대비, 대응, 복구 네 분야의 52개 지표로 이루어졌다.
□ 기획재정부는 내년도 예산안의 국회 심의 일정에 맞추어 부처별 설명 자료를 배포했다.
○ 예산안에는 저출생 대응과 지역 균형 발전 사업이 중점적으로 반영되었다.
□ 고용노동부는 중소기업 재직자의 직무 훈련 참여율이 전년보다 6.3%p 높아졌다고 설명했다.
○ 훈련 참여 기업에는 인건비 일부와 훈련비가 지원된다.
□ 보건복지부는 장애인 활동지원 서비스의 월 최대 급여량을 내년 1월부터 상향한다고 밝혔다.
○ 이번 조치로 약 12만 명의 이용자가 추가 지원을 받게 된다.
□ 농림축산식품부는 쌀 수급 안정을 위해 공공비축 매입량을 45만 톤으로 확대했다.
○ 매입 가격은 수확기 산지 평균 가격을 기준으로 연말에 확정된다.
□ 중소벤처기업부는 소상공인 정책자금 금리를 0.5%p 인하하고 상환 기간을 연장했다.
○ 신청은 소상공인시장진흥공단 누리집과 지역 센터에서 받는다.""".splitlines()


def press(i):
    """8 sentences, stride 3: neighbours share 5/8 lines (shingle Jaccard ~0.45, below the near-dup threshold)."""
    return '\n'.join(SENTENCES[j % len(SENTENCES)] for j in range(3 * i, 3 * i + 8))


class QualityTests(unittest.TestCase):
    def test_press_release_is_kept_under_both_profiles(self):
        for profile in kq.PROFILES.values():
            self.assertEqual(kq.assess(PRESS, t=profile), (True, 'ok'))

    def test_repeated_paragraphs_are_dropped(self):
        para = PRESS.splitlines()[3]
        ok, reason = kq.assess('\n'.join([para] * 12))
        self.assertFalse(ok)
        self.assertEqual(reason, 'dup_line_frac')
        # same content but each line slightly different: caught by n-gram repetition instead
        ok, reason = kq.assess('\n'.join(f'{para} ({i}번)' for i in range(12)))
        self.assertFalse(ok)
        self.assertTrue(reason.startswith(('top_', 'duplicated_')), reason)

    def test_garbage_and_foreign_text_are_dropped(self):
        self.assertEqual(kq.assess('\x1b\x1a\x19 Secretariat for the Conference ' * 20)[1], 'hangul_ratio')
        self.assertEqual(kq.assess('')[1], 'empty')
        self.assertEqual(kq.assess('\n'.join(SENTENCES[2:4]))[1], 'gopher_short_doc')  # 48 morphemes < 50

    def test_profiles_share_fineweb2_values_except_three(self):
        self.assertEqual(kq.FINEWEB2.dup_n_grams[0], (5, 0.201))
        self.assertEqual((kq.FINEWEB2.min_alpha_words, kq.FINEWEB2.max_words), (0.791, 100_000))
        self.assertEqual(kq.PUBLIC_DOCS.dup_n_grams[0], (5, 0.402))
        self.assertEqual((kq.PUBLIC_DOCS.min_alpha_words, kq.PUBLIC_DOCS.max_words), (0.7, 1_000_000))
        self.assertEqual(kq.PUBLIC_DOCS.top_n_grams, kq.FINEWEB2.top_n_grams)
        self.assertEqual(len(kq.STOPWORDS), 21)

    def test_line_cleanup(self):
        text = PRESS + '\n이 페이지는 JavaScript를 사용합니다.\n로그인 | 회원가입 | 개인정보처리방침\n' + 'x' * 1001
        cleaned, dropped = kq.clean_lines(text)
        self.assertEqual(dropped, 3)
        self.assertEqual(cleaned, PRESS.strip())
        self.assertEqual(kq.clean_lines(PRESS), (PRESS, 0))

    def test_pii_is_replaced_with_fixed_tokens(self):
        text = ('담당자 홍길동(010-1234-5678, hong@korea.kr), 주민등록번호 900101-1234567, '
                '계좌 국민은행 123456-02-123456, 여권 M12345678, 운전면허 11-12-345678-90')
        out, counts = kq.redact(text)
        self.assertEqual(dict(counts), {'PHONE': 1, 'EMAIL': 1, 'RRN': 1, 'ACCOUNT': 1, 'PASSPORT': 1, 'DRIVER_LICENSE': 1})
        self.assertEqual(out, '담당자 홍길동([전화], [이메일]), 주민등록번호 [주민번호], 계좌 국민은행 [계좌], 여권 [여권], 운전면허 [운전면허]')
        self.assertEqual(kq.redact('제1조(목적) 이 법은 2014년 4월 16일에 시행한다.'), ('제1조(목적) 이 법은 2014년 4월 16일에 시행한다.', {}))

    def test_pii_inside_longer_code_or_table_is_left_alone(self):
        text = '② 사업코드: (예시: 061-1300-1334-500), 문의 061-1300-1334'
        self.assertEqual(kq.redact(text), ('② 사업코드: (예시: 061-1300-1334-500), 문의 [전화]', {'PHONE': 1}))
        tables = ['0 1 396 136.78900416 0 1 432 154.16341649', '5 5 4 4 4 4 4 3 3 3 2 1 1 1\n\n4 4 3 3 3 3 3 2 2 2 1 1 1 1',
                  '2010 2011 2012 2013 2014\n\n498 462 471', '분류번호\n\n1105 1106 2101 2402 2501 3102 3103 4110']
        for t in tables:
            self.assertEqual(kq.redact(t), (t, {}), t)

    def test_contacts_on_their_own_line_are_replaced_once(self):
        text = ('CUSTOMER CENTER\n010-2776-1821\n※ 안내\n[전화] / [전화]\n010-5496-7630\n010-6557-1132\n'
                '문의: (02)724-7836\n담당 02-123-4567\nhong@korea.kr')
        out, counts = kq.redact(text)
        self.assertEqual(out, 'CUSTOMER CENTER\n[전화]\n※ 안내\n[전화] / [전화]\n[전화]\n[전화]\n문의: [전화]\n담당 [전화]\n[이메일]')
        self.assertEqual(dict(counts), {'PHONE': 5, 'EMAIL': 1})
        self.assertEqual(kq.redact(out), (out, {}))
        text = ('문의. 02) 737-7378, www.x.org\n문의 : 052- 290-3333\nTEL:032)433-3777ㅣEmail\n'
                '강호준 010-1234-5678 hong@korea.kr\n주문일 2024.03.15, 금액 1,234,567원, ISBN 978-89-7277-350-4')
        out, counts = kq.redact(text)
        self.assertEqual(out, '문의. [전화], www.x.org\n문의 : [전화]\nTEL:[전화]ㅣEmail\n'
                              '강호준 [전화] [이메일]\n주문일 2024.03.15, 금액 1,234,567원, ISBN 978-89-7277-350-4')
        self.assertEqual(dict(counts), {'PHONE': 4, 'EMAIL': 1})

    def test_contacts_ko_pii_misses_are_replaced_and_business_or_structural_numbers_kept(self):
        cases = {
            '밥 해주는 남자\n050-7801-9760\n서울': '밥 해주는 남자\n[전화]\n서울',           # 050 안심번호: 개인 연결 가능
            '02.3444.0339TUE~SUN': '[전화]TUE~SUN',                                   # 영문에 붙은 번호
            '전화번호 072-468-7012': '전화번호 [전화]',                                  # 해외 지역번호: 보수적 치환
            'WhatsApp : +8615838182160 / 우리는': 'WhatsApp : [전화] / 우리는',
            '문의 080-272-1111': '문의 080-272-1111',                                  # 080 수신자 부담: 사업자 번호
            'ARS 060-707-1080': 'ARS 060-707-1080',                                   # 060 정보이용료 ARS
            '학원등록번호 110111-3171678': '학원등록번호 110111-3171678',                # 법인등록번호 검증식 통과
            '주민등록번호 110111-3171678': '주민등록번호 [주민번호]',
            '등록번호 900101-1234567': '등록번호 [주민번호]',                             # 법인 검증식 불일치
            'n?mode=LSD&oid=214&aid=0000949741 / 국민들은': 'n?mode=LSD&oid=214&aid=0000949741 / 국민들은',
            '계좌정보 신협 131021127875 주식회사': '계좌정보 신협 [계좌] 주식회사',
            '좌표 +37.5665 126.9780, 전년 대비 +3.2%': '좌표 +37.5665 126.9780, 전년 대비 +3.2%',
            '사업코드 061-1300-1334-500': '사업코드 061-1300-1334-500',
            '코팅제 shoe cream (8411250025125) ※': '코팅제 shoe cream (8411250025125) ※',   # EAN-13 바코드
            '축진참돈(상표: 4008152650000)과': '축진참돈(상표: 4008152650000)과',               # 상표등록번호
            '주민번호 8411250025125': '주민번호 [주민번호]',
            '개발전략수립 0101010101_15v1 정보': '개발전략수립 0101010101_15v1 정보',
            '구분 2017 2018 2019 2020 (A)': '구분 2017 2018 2019 2020 (A)',                      # 표 연도 열
            '연 락 처 2100-0645 2100-0647 / 쪽수': '연 락 처 2100-0645 2100-0647 / 쪽수',          # 내선번호 나열
            '카드 4111-1111-1111-1111 결제': '카드 [카드] 결제',
            '세정제 F-B01B-T00120001-A180 ㈜밀텍': '세정제 F-B01B-T00120001-A180 ㈜밀텍',         # 제품 검사 코드
            '삼흥종합건설㈜ 정대영 210111-0011511 건설': '삼흥종합건설㈜ 정대영 210111-0011511 건설',  # 7번째 자리 0
            '여권 M12345678 발급': '여권 [여권] 발급',
            '산학협력단 / 2023 2025 0401 1231 / 116': '산학협력단 / 2023 2025 0401 1231 / 116',      # 연구기간 열
            'https://www.facebook.com/koreacdcpr/posts/2369610576400219 질병': 'https://www.facebook.com/koreacdcpr/posts/2369610576400219 질병',
            '(http://news.chosun.com/site/data/html_dir/2011/02/20/2011022001177.html)': '(http://news.chosun.com/site/data/html_dir/2011/02/20/2011022001177.html)',
            '주민(법인)등록번호 900101-1234567': '주민(법인)등록번호 [주민번호]',
            '신경과 58 58 △3 52 48 +12 62 61 36 36': '신경과 58 58 △3 52 48 +12 62 61 36 36',          # 표의 증감 수치
            '연락처 +65 65323357, +84 938244404': '연락처 [전화], [전화]',
            '㈜와이즈스톤 110111-3701962 서울시 서초구': '㈜와이즈스톤 110111-3701962 서울시 서초구',  # 법인등록번호 열
            '출원 10-2019-0164070 우리흑돈의': '출원 10-2019-0164070 우리흑돈의',                    # 특허 출원번호
            '홍길동 900101-1234567 서울': '홍길동 [주민번호] 서울',
            '재단법인 포항테크노파크 1,017 20250923 규격가격': '재단법인 포항테크노파크 1,017 20250923 규격가격',  # 금액+날짜
        }
        for text, expected in cases.items():
            self.assertEqual(kq.redact(text)[0], expected, text)

    def test_builder_records_quality_and_pii_counts(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / 'press'
            src.mkdir()
            docs = [press(i) for i in range(6)] + [press(6) + '\n문의: 044-201-3456, pr@molit.go.kr',
                                                   '\n'.join([SENTENCES[0]] * 12)]
            by_split = {s: [f'g{i}' for i in range(500) if mix.split_for_group(f'g{i}', 10) == s] for s in ('train', 'valid', 'test')}
            with (src / 'docs.jsonl').open('w', encoding='utf-8') as f:
                for i, text in enumerate(docs):
                    group = by_split['valid'][0] if i == 0 else by_split['test'][0] if i == 1 else by_split['train'][i]
                    f.write(json.dumps({'id': f'p{i}', 'text': text, 'license': 'KOGL-1', 'group': group}, ensure_ascii=False) + '\n')
            mix.build([(src, 10**9)], Path(d) / 'out', 10)
            prov = json.loads((Path(d) / 'out/provenance.json').read_text())
            self.assertEqual(prov['filter_counts']['press:quality:dup_line_frac'], 1)
            self.assertEqual(prov['filter_counts']['press:pii:PHONE'], 1)
            self.assertEqual(prov['filter_counts']['press:pii:EMAIL'], 1)
            self.assertEqual(prov['quality']['profile'], 'public-docs')
            self.assertEqual(prov['quality']['thresholds']['min_alpha_words'], 0.7)
            removed = json.loads((Path(d) / 'out/quality-removed.json').read_text())
            self.assertEqual(removed, [{'id': 'p7', 'source': str(src), 'reason': 'dup_line_frac'}])
            rows = [json.loads(l) for s in ('train', 'valid', 'test') for l in (Path(d) / f'out/{s}.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 7)
            self.assertIn('문의: [전화], [이메일]', next(r['text'] for r in rows if r['id'] == 'p6'))


if __name__ == '__main__':
    unittest.main()
