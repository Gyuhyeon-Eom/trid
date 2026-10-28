import io
import unittest
import zipfile

from scripts.collect_korea_briefing import (CREDIT, attribution, build_doc, clean_text, doclang_text, hwpx_text,
                                            join_pdf_lines, parse_list, parse_view, pick_attachments)

BODY = ("□ 정부는 지난 5월에 정상화가 필요한 164개 사안을 1차 과제로 선정하고, 국민 눈높이와 맞지 않는 제도와 관행, "
        "편법행위를 개선하는 데에 역량을 집중해왔다. 그 결과 대부분의 과제가 차질없이 추진되고 있으며, 먹거리 담합과 "
        "휴가철 바가지요금 등 국민생활과 밀접한 분야를 중심으로 구체적인 개선성과가 나타나고 있으며, 정부는 남은 "
        "과제도 끝까지 책임 있게 이행할 계획이다.")

VIEW = """<html><title>제목 - 보도자료 | 브리핑룸 | 대한민국 정책브리핑</title>
<input type="hidden" id="titleText" name="titleText" value="재외공관장 &amp; 방산" />
<div class="info">
  <span>2015.03.31</span>
<div class="filedown"><dd>
<span><a href="/common/download.do?fileId=11&amp;tblKey=GMN">
  <img src="/images/icon/icon_ipdf.gif" alt="PDF파일">보도자료(최종).pdf
  </a></span>
<span><a href="/common/download.do?fileId=12&amp;tblKey=GMN">
  <img src="/images/icon/icon_ihangul.gif" alt="한글파일">보도자료(최종).hwpx
  </a></span>
<span><a href="/common/download.do?fileId=13&amp;tblKey=GMN">
  <img src="/images/icon/icon_ihangul.gif" alt="한글파일">붙임 참고자료.hwpx
  </a></span>
<span><a href="/common/download.do?fileId=14&amp;tblKey=GMN">
  <img src="/images/icon/icon_isetup.gif" alt="첨부파일">KDR_0073.JPG
  </a></span>
</dd></div>
<div class="article_body"><div class="view_cont">
  <iframe src="/docViewer/x"></iframe>
  <p>본문 첫 문단</p>
  <p class="remark">“이 자료는 외교부의 보도자료를 전재하여 제공함을 알려드립니다.”</p>
</div></div>
<div class="article_footer"><div class="kogl">
<span>이 자료는 <strong>텍스트에 한하여 공공누리 제1유형(출처표시)의 조건</strong>에 따라 이용할 수 있습니다.</span>
</div></div></html>"""


def record(**kw):
    base = {"news_id": "156043581", "url": "https://www.korea.kr/briefing/pressReleaseView.do?newsId=156043581",
            "title": "제목", "date": "2015-03-31", "list_date": "2015-03-31", "list_ministry": "외교부",
            "kogl_type1": True, "ministry_line": "이 자료는 외교부의 보도자료를 전재하여 제공함을 알려드립니다.",
            "body_source": "hwpx", "body": BODY, "error": None}
    return {**base, **kw}


class CleaningTest(unittest.TestCase):
    def test_header_contacts_pictures_and_page_markers_removed(self):
        raw = "\n".join([
            "보도자료",
            "보도시점 2026. 10. 20.(화) 13:00 배포 2026. 10. 20.(화) 10:00",
            "2026. 10. 20.(화) 13:00",
            "제15-184호",
            "담당 부서 국무조정실 기획총괄정책관실 책임자 국 장 김기한 (044-200-2046)",
            "담당자 사무관 박찬우 (044-200-2059)",
            "문 의 : 양자경제외교국 김영준 심의관(☎:02-2100-7663)",
            "그림입니다.",
            "원본 그림의 이름: CLP000012.bmp",
            "원본 그림의 크기: 가로 640pixel, 세로 480pixel",
            "<표>",
            BODY,
            "- 1 -",
            "3 / 7",
            "이 보도자료와 관련하여 보다 자세한 내용이나 취재를 원하시면 홍보팀으로 연락 주시기 바랍니다.",
        ])
        text = clean_text(raw)
        self.assertEqual(text, BODY)

    def test_phone_and_email_stripped_inside_long_body_lines(self):
        line = BODY + " 자세한 사항은 콜센터(1588-1234) 또는 help@korea.kr, 대표전화 02-2100-7663으로 안내한다."
        text = clean_text(line)
        for leak in ("1588-1234", "help@korea.kr", "02-2100-7663", "()"):
            self.assertNotIn(leak, text)
        self.assertIn("2015", clean_text(BODY + " 2015-2016년 예산은 1,234억원이다."))

    def test_body_sentences_mentioning_keywords_survive(self):
        line = "담당자는 현장 책임자와 함께 점검하였다. 배포된 자료는 보도시점 이후 공개된다."
        self.assertEqual(clean_text(line), line)

    def test_private_use_glyphs_and_nfc(self):
        decomposed = "\u1100\u1161 \uf02e \U000f02ef 항목"
        self.assertEqual(clean_text(decomposed), "가 항목")

    def test_pdf_lines_rejoined_but_bullets_and_markers_kept_apart(self):
        text = join_pdf_lines("□ 관행적으로 이어져 온 부당이득을 막기 위한 제도적\n장치도 마련하였다.\nㅇ 둘째 항목\n- 1 -\n다음")
        self.assertEqual(text.splitlines(),
                         ["□ 관행적으로 이어져 온 부당이득을 막기 위한 제도적 장치도 마련하였다.", "ㅇ 둘째 항목", "- 1 -", "다음"])


class LicenseFilterTest(unittest.TestCase):
    def test_view_page_parsed(self):
        view = parse_view(VIEW)
        self.assertTrue(view["kogl_type1"])  # notice split by <strong> still matches
        self.assertEqual(view["title"], "재외공관장 & 방산")
        self.assertEqual(view["date"], "2015-03-31")
        self.assertEqual(view["ministry_line"], "이 자료는 외교부의 보도자료를 전재하여 제공함을 알려드립니다.")
        self.assertEqual(view["page_text"], "본문 첫 문단")
        self.assertEqual([a["file_id"] for a in pick_attachments(view["attachments"])], ["12", "11"])

    def test_page_without_type1_notice_is_dropped(self):
        page = VIEW.replace("공공누리 제1유형(출처표시)", "공공누리 제4유형(출처표시+상업적이용금지+변경금지)")
        self.assertFalse(parse_view(page)["kogl_type1"])
        older = VIEW.replace("텍스트에 한하여 공공누리", "공공누리가 부착되지 않은 자료는 담당자와 협의한 후에 사용")
        self.assertFalse(parse_view(older)["kogl_type1"])
        self.assertEqual(build_doc(record(kogl_type1=False)), (None, "no_kogl_type1"))

    def test_stub_dropped_and_kept_row_fields(self):
        self.assertEqual(build_doc(record(body="보도시점 2015. 3. 31.\n짧은 본문"))[1], "too_short")
        self.assertEqual(build_doc(record(error="view: HTTPError"))[1], "fetch_error")
        row, reason = build_doc(record())
        self.assertIsNone(reason)
        self.assertEqual((row["license"], row["group"], row["kind"], row["date"]),
                         ("KOGL-Type1", "korea-kr:156043581", "press_release", "2015-03-31"))
        self.assertTrue(row["attribution"].startswith("이 자료는 외교부의 보도자료를 전재"))
        self.assertTrue(row["attribution"].endswith(CREDIT))
        self.assertEqual(attribution(None, "산림청"), f"이 자료는 산림청의 보도자료를 전재하여 제공함을 알려드립니다. {CREDIT}")

    def test_list_page_ignores_sidebar_links(self):
        page = ('<div class="result"> 총 <strong>1,753</strong>건</div>'
                '<a href="/briefing/pressReleaseView.do?newsId=156043581&amp;pageIndex=1&amp;startDate=2015-03-01">'
                '<strong>t</strong><span class="source"><span>2015.03.31</span><span>외교부</span></span></a>'
                "<a href='/briefing/pressReleaseView.do?newsId=156786677&pWise=sub&pWiseSub=C5' >x</a>")
        self.assertEqual(parse_list(page), (1753, [("156043581", "2015-03-31", "외교부")]))


class HwpxTest(unittest.TestCase):
    def test_table_rows_flattened_so_header_rows_are_removed(self):
        ns = 'xmlns:hp="http://www.hancom.co.kr/hwpml/2011/paragraph"'

        def p(text, inner=""):
            return f"<hp:p><hp:run><hp:t>{text}</hp:t>{inner}</hp:run></hp:p>"

        def cell(text):
            return f"<hp:tc><hp:subList>{p(text)}</hp:subList></hp:tc>"
        table = ("<hp:tbl><hp:tr>" + cell("보도시점") + cell("2026. 10. 20.(화) 13:00") + cell("배포") +
                 cell("2026. 10. 20.(화) 10:00") + "</hp:tr></hp:tbl>")
        pic = "<hp:pic><hp:shapeComment>그림입니다. 원본 그림의 이름: a.png</hp:shapeComment></hp:pic>"
        xml = f'<hs:sec {ns} xmlns:hs="x">{p("", table)}{p("본문<hp:tab/>첫 줄", pic)}</hs:sec>'
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("Contents/section0.xml", xml)
            z.writestr("Contents/header.xml", "<x/>")
        text = hwpx_text(buf.getvalue())
        self.assertIn("보도시점 2026. 10. 20.(화) 13:00 배포 2026. 10. 20.(화) 10:00", text.splitlines())
        self.assertEqual(clean_text(text), "본문 첫 줄")


class DoclangTest(unittest.TestCase):
    def test_rhwp_doclang_rows_cells_and_headers(self):
        xml = ('<doclang version="0.6"><table><fcel/><picture><src uri="a.jpg"/></picture><fcel/><text><bold>보도시점'
               '</bold></text><fcel/><text>2026. 10. 20.(화) 13:00</text><nl/><fcel/><text>제목 한 줄</text><lcel/><nl/>'
               '</table><page_header><text>- 1 -</text></page_header><text>본문\u2007첫  줄</text>'
               '<footnote><text>각주</text></footnote></doclang>')
        text = doclang_text(xml.encode())
        self.assertEqual(text.splitlines(), ["보도시점 2026. 10. 20.(화) 13:00", "제목 한 줄", "본문 첫 줄", "각주"])
        self.assertEqual(clean_text(text), "제목 한 줄\n본문 첫 줄\n각주")


if __name__ == "__main__":
    unittest.main()
