"""Self-distilled chat rows must not teach invented facts or the teacher's identity."""
import unittest
from scripts.distill_trid_chat import screen


class ScreenTests(unittest.TestCase):
    def row(self, answer, prompt='오늘 며칠이야?', history=()):
        return {'prompt': prompt, 'history': list(history), 'answer': answer}

    def test_invented_quantities_are_dropped_but_quantities_from_the_conversation_kept(self):
        self.assertEqual(screen(self.row('오늘은 2024년 6월 19일입니다.'))[1], 'unsupported_quantity')
        self.assertEqual(screen(self.row('보통 출장 종료 후 7일 이내입니다.', '출장 정산 기한은?'))[1], 'unsupported_quantity')
        kept, _ = screen(self.row('네, 3일 전까지 신청하시면 됩니다.', '3일 전까지 신청하면 되지?'))
        self.assertIsNotNone(kept)
        kept, _ = screen(self.row('윤다현 님이 30만 원이라고 하셨어요.', '얼마였지?',
                                  [{'user': '예산은 30만 원이야', 'assistant': '알겠습니다.'}]))
        self.assertIsNotNone(kept)
        self.assertIsNotNone(screen(self.row('1. 목적을 정하고\n2. 일정을 잡으세요.', '순서 알려줘'))[0])

    def test_vendor_names_dropped_and_assistant_name_normalized(self):
        self.assertEqual(screen(self.row('저는 카카오의 Kanana입니다.', '누구야?'))[1], 'vendor_name')
        kept, _ = screen(self.row('저는 트리디(TRID)라고 불리는 AI입니다.', '누구야?'))
        self.assertEqual(kept['answer'], '저는 trid라고 불리는 AI입니다.')


if __name__ == '__main__':
    unittest.main()
