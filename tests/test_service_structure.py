import unittest

from trid.service.structure import CIRCLED, MAX_CHARS, external_references, is_regulation, parse_document

REG = """출장여비규정
제정 2019. 3. 1.

제1장 총칙

제1조(목적) 이 규정은 한국공공데이터원 임직원의 출장여비 지급 기준을 정함을 목적으로 한다.
제2조(정의) 이 규정에서 사용하는 용어의 뜻은 다음과 같다.
1. "국내출장"이란 근무지 외의 국내 지역으로 가는 출장을 말한다.
2. "숙박비"란 출장지에서 숙박에 드는 비용을 말한다.

제2장 국내출장

제12조(국내출장여비) ① 국내출장 여비는 운임, 일비, 숙박비, 식비로 구분하여 지급한다.
② 숙박비의 상한은 1박당 8만 원으로 한다. 다만, 제13조에 따른 특별시 출장은 별표 2의 상한을 적용한다.
③ 일비는 1일 2만 5천 원으로 한다.
제13조(특별시 출장) 서울특별시로 가는 출장은 제12조제2항의 상한을 1박당 10만 원으로 한다.
제14조(정산) 출장자는 귀임 후 7일 이내에 별지 제1호서식에 따라 정산서를 제출하여야 한다. 「공공기관 회계규칙」 제5조를 준용한다.

부칙
이 규정은 2019년 3월 1일부터 시행한다.

[별표 2] 특별시 숙박비 상한
구분 상한액
특별시 1박당 10만 원
광역시 1박당 9만 원

[별지 제1호서식] 출장 정산서
성명 부서 출장기간 금액
"""


def by_path(passages):
    return {p.path[1:]: p for p in passages}


class RegulationTest(unittest.TestCase):
    def setUp(self):
        self.passages, self.refs = parse_document(REG, "여비", "출장여비규정")
        self.paths = by_path(self.passages)

    def test_paths_follow_장_조_별표_부칙(self):
        self.assertTrue(is_regulation(REG))
        self.assertEqual(list(self.paths), [("전문",), ("제1장", "제1조"), ("제1장", "제2조"), ("제2장", "제12조"),
                                            ("제2장", "제13조"), ("제2장", "제14조"), ("부칙",), ("별표 2",), ("서식 1",)])
        self.assertTrue(all(p.path[0] == "출장여비규정" for p in self.passages))
        self.assertEqual([p.id for p in self.passages][:2], ["여비:0001", "여비:0002"])

    def test_passages_are_verbatim_slices_within_limit(self):
        for p in self.passages:
            self.assertEqual(REG[p.char_start:p.char_end], p.text)
            self.assertLessEqual(len(p.text), MAX_CHARS)
            self.assertEqual(p.text, p.text.strip())
        self.assertTrue(self.paths[("제2장", "제12조")].text.startswith("제12조(국내출장여비) ①"))
        self.assertTrue(self.paths[("제2장", "제12조")].text.endswith("2만 5천 원으로 한다."))
        self.assertNotIn("제2장 국내출장", "".join(p.text for p in self.passages))  # 장 lines are path only

    def test_references_resolve_to_조_별표_서식(self):
        pid = lambda *path: self.paths[path].id
        self.assertEqual(set(self.refs), {
            (pid("제2장", "제12조"), pid("제2장", "제13조")),   # "제13조에 따른"
            (pid("제2장", "제12조"), pid("별표 2")),            # "별표 2의 상한"
            (pid("제2장", "제13조"), pid("제2장", "제12조")),   # "제12조제2항"
            (pid("제2장", "제14조"), pid("서식 1")),            # "별지 제1호서식"
        })
        self.assertEqual(external_references(self.paths[("제2장", "제14조")].text), [("공공기관 회계규칙", "제5조")])


class SplittingTest(unittest.TestCase):
    def test_long_조_splits_by_항_and_항_references_hit_the_right_piece(self):
        hang = lambda n: f"{CIRCLED[n - 1]} " + "출장자는 숙박비 영수증을 첨부하여 정산서를 제출하여야 하며 초과 금액은 지급하지 아니한다. " * 3
        text = "여비규정\n제1조(목적) 목적.\n제2조(지급) " + "\n".join(hang(i) for i in range(1, 8)) + "\n제3조(기타) 제2조제5항에 따른다.\n"
        passages, refs = parse_document(text, "r", "여비규정")
        paths = [p.path[1:] for p in passages]
        self.assertEqual(paths, [("전문",), ("제1조",), ("제2조", "①~③"), ("제2조", "④~⑥"), ("제2조", "⑦"), ("제3조",)])
        self.assertTrue(all(len(p.text) <= MAX_CHARS and text[p.char_start:p.char_end] == p.text for p in passages))
        self.assertTrue(passages[2].text.startswith("제2조(지급) ①"))
        self.assertEqual(refs, [("r:0006", "r:0004")])  # 제5항 lives in the ④~⑥ piece

    def test_inline_항_호_and_hard_split_stay_within_limit(self):
        text = ("규정\n제1조(목적) 목적.\n제2조(지급) ① 다음 각 호와 같다. 1. 가항 " + "내용 " * 100 + " 2. 나항 " + "내용 " * 100
                + "② 둘째 " + "내용 " * 200 + "\n제3조(끝) 끝.")
        passages, _ = parse_document(text, "r", "규정")
        self.assertEqual([p.path[1:] for p in passages],
                         [("전문",), ("제1조",), ("제2조", "①", "1."), ("제2조", "①", "2."), ("제2조", "②", "1"), ("제2조", "②", "2"), ("제3조",)])
        self.assertTrue(all(len(p.text) <= MAX_CHARS and text[p.char_start:p.char_end] == p.text for p in passages))
        self.assertTrue(passages[2].text.startswith("제2조(지급) ① 다음 각 호"))

    def test_press_release_falls_back_to_paragraphs(self):
        text = "보도자료 제목\n\n" + "\n\n".join(f"문단 {i} 입니다. " * 20 for i in range(5))
        self.assertFalse(is_regulation(text))
        passages, refs = parse_document(text, "pr", "보도자료 제목")
        self.assertEqual([p.path for p in passages], [("보도자료 제목", f"¶{n}") for n in range(1, len(passages) + 1)])
        self.assertTrue(all(len(p.text) <= MAX_CHARS and text[p.char_start:p.char_end] == p.text for p in passages))
        self.assertTrue(all(len(p.text) > 300 for p in passages[:-1]))  # packed, not one passage per line
        self.assertEqual(refs, [])

    def test_reference_at_line_start_is_not_a_heading(self):
        text = "규정\n제1조(목적) 목적.\n제12조에 따른 여비는 다음과 같다.\n제2조(지급) ① 지급.\n"
        passages, _ = parse_document(text, "r", "규정")
        self.assertEqual([p.path[1:] for p in passages], [("전문",), ("제1조",), ("제2조",)])
        self.assertIn("제12조에 따른", passages[1].text)


if __name__ == "__main__":
    unittest.main()
