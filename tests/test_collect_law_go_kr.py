import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.collect_law_go_kr import (AuthError, Client, cached, clean, extract, source_url, template_title,
                                       TARGETS)

FILLER = "이 법은 국민의 권리와 의무에 관한 사항을 정하여 공공복리의 증진에 이바지함을 목적으로 한다. " * 4

LAW = f"""<?xml version="1.0" encoding="UTF-8"?><법령>
<기본정보><법령명_한글><![CDATA[시험법]]></법령명_한글><전화번호>044-200-1234</전화번호>
<연락부서><부서단위><부서명>시험과</부서명><부서연락처>044-200-1234</부서연락처></부서단위></연락부서></기본정보>
<조문>
<조문단위><조문여부>전문</조문여부><조문내용>      제1장 총칙</조문내용></조문단위>
<조문단위><조문여부>조문</조문여부><조문내용>제1조(목적) {FILLER}&lt;개정 2020.1.1&gt;</조문내용></조문단위>
<조문단위><조문여부>조문</조문여부><조문내용>제2조 삭제 &lt;2019.1.15&gt;</조문내용>
<조문참고자료>[2019.1.15 법률 제1호에 의하여 위헌 결정된 이 조를 삭제함.]</조문참고자료></조문단위>
<조문단위><조문여부>조문</조문여부><조문내용>제3조(정의)</조문내용>
<항><항번호>①</항번호><항내용>① 용어의 뜻은 다음과 같다.</항내용>
<호><호번호>1.</호번호><호내용>1. "시험"이란 확인하는 일을 말한다.</호내용>
<목><목번호>가.</목번호><목내용>가. 서면 시험</목내용></목>
<목><목번호>나.</목번호><목내용>나. 삭제 &lt;2021.3.2&gt;</목내용></목></호>
<호><호번호>2.</호번호><호내용>2. 삭제 &lt;2021.3.2&gt;</호내용></호></항>
<항><항번호>②</항번호><항내용>② 삭제 &lt;2022.5.5&gt;</항내용></항>
</조문단위>
</조문>
<부칙>
<부칙단위><부칙공포일자>20200101</부칙공포일자><부칙내용>부칙 &lt;제1호,2020.1.1&gt; 옛 부칙</부칙내용></부칙단위>
<부칙단위><부칙공포일자>20230101</부칙공포일자><부칙내용>부칙 &lt;제9호,2023.1.1&gt;
제1조(시행일) 이 법은 공포한 날부터 시행한다.</부칙내용></부칙단위>
</부칙>
<별표><별표단위><별표구분>별표</별표구분><별표내용>■ 시험법 [별표 1]
┌──────┐
│ 수수료표 │
└──────┘</별표내용></별표단위>
<별표단위><별표구분>서식</별표구분><별표내용>■ [별지 제1호서식] 신청서 성명 주소</별표내용></별표단위></별표>
</법령>""".encode()


class CleaningTests(unittest.TestCase):
    def test_markup_and_pii_removed(self):
        text = clean("【주 문】<br/>원고의 청구를 기각한다.<br/><b>담당</b> 연락은 02-2100-1234 또는 "
                     "law@example.go.kr 로 한다.<p>1588-1234</p>")
        self.assertNotIn("<", text)
        self.assertIn("【주 문】\n원고의 청구를 기각한다.", text)
        for leaked in ("2100-1234", "@", "1588-1234"):
            self.assertNotIn(leaked, text)

    def test_legal_numbers_survive_pii_filter(self):
        text = clean("1950-1953년 사건 2004고단2657 판결, 2019. 1. 15. 선고, 대법원 2010다12345")
        for kept in ("1950-1953", "2004고단2657", "2019. 1. 15.", "2010다12345"):
            self.assertIn(kept, text)

    def test_deleted_stubs_and_contact_lines_dropped(self):
        text = clean("제5조 삭제 <2010.1.1>\n제6조의2 삭제\n② 삭제 <2019.1.15>\n3. 삭제 <2020.2.2>\n"
                     "가. 삭제\n담당부서: 법제과\n☎ 044-200-0000\n제7조(삭제된 조문의 효력) 삭제된 조문은 효력이 없다.")
        self.assertEqual(text, "제7조(삭제된 조문의 효력) 삭제된 조문은 효력이 없다.")

    def test_statute_units_buchik_and_tables(self):
        title, text, reason = extract("law", LAW, {})
        self.assertIsNone(reason)
        self.assertEqual(title, "시험법")
        lines = text.split("\n")
        self.assertEqual(lines[:3], ["시험법", "제1장 총칙", lines[2]])
        self.assertTrue(lines[2].startswith("제1조(목적)"))
        order = [text.index(s) for s in ("제3조(정의)", "① 용어", '1. "시험"', "가. 서면 시험", "부칙 <제9호", "수수료표")]
        self.assertEqual(order, sorted(order))
        for dropped in ("삭제", "개정 2020", "옛 부칙", "신청서", "044-200", "시험과", "위헌 결정", "┌", "│"):
            self.assertNotIn(dropped, text)

    def test_short_statute_dropped(self):
        body = "<LawService><자치법규기본정보><자치법규명>짧은 조례</자치법규명></자치법규기본정보>" \
               "<조문><조><조내용>제1조(목적) 짧다.</조내용></조></조문></LawService>".encode()
        self.assertEqual(extract("ordin", body, {}), (None, None, "short"))


class RulingFilterTests(unittest.TestCase):
    def test_missing_body_response(self):
        body = '<?xml version="1.0" encoding="UTF-8"?><Law>일치하는 판례가 없습니다.  판례명을 확인하여 주십시오.</Law>'
        self.assertEqual(extract("prec", body.encode(), {"사건명": "x"}), (None, None, "no_body"))

    def test_stub_ruling_dropped_even_with_long_title(self):
        title = "국세기본법위반 사건에 관한 매우 긴 사건명 " * 10
        body = f"<PrecService><사건명>{title}</사건명><판례내용>국세법령정보시스템 참조&lt;br/&gt;</판례내용></PrecService>"
        self.assertEqual(extract("prec", body.encode(), {}), (None, None, "stub_ruling"))

    def test_ruling_fields_kept_in_order(self):
        body = (f"<DetcService><사건명>위헌확인</사건명><판시사항>판시 {FILLER}</판시사항><결정요지></결정요지>"
                f"<전문>주 문&lt;br/&gt;심판청구를 각하한다.</전문></DetcService>").encode()
        title, text, reason = extract("detc", body, {})
        self.assertIsNone(reason)
        self.assertTrue(text.startswith("위헌확인\n【판시사항】\n판시 "))
        self.assertNotIn("【결정요지】", text)
        self.assertTrue(text.endswith("주 문\n심판청구를 각하한다."))


class ApiSafetyTests(unittest.TestCase):
    def test_source_url_drops_api_key(self):
        item = {"판례상세링크": "/DRF/lawService.do?OC=secret&target=prec&ID=625055&type=HTML&mobileYn="}
        url = source_url(item, TARGETS["prec"])
        self.assertEqual(url, "http://www.law.go.kr/DRF/lawService.do?target=prec&ID=625055&type=HTML")

    def test_rejected_key_raises_and_is_not_cached(self):
        client = Client("bad")
        client.robots = mock.Mock(can_fetch=mock.Mock(return_value=True))
        rejected = ('<?xml version="1.0" encoding="UTF-8"?>\n    <Response>\n        <result>사용자 정보 검증에 '
                    '실패하였습니다.</result>\n    </Response>').encode()
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(Client, "_open", return_value=rejected):
            path = Path(tmp) / "prec" / "1.xml"
            with self.assertRaises(AuthError):
                cached(path, lambda: client.get("lawService.do", {"target": "prec", "ID": "1"}))
            self.assertFalse(path.exists())

    def test_ordinance_template_title(self):
        self.assertEqual(template_title("구미시 주차장 설치 및 관리 조례", "경상북도 구미시"), "주차장 설치 및 관리 조례")
        self.assertEqual(template_title("서울특별시 강남구 도시계획 조례", "서울특별시 강남구"), "도시계획 조례")


if __name__ == "__main__":
    unittest.main()
