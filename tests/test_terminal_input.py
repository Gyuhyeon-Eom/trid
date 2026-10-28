import unittest
from trid.terminal_input import classify_input


class TerminalInputTests(unittest.TestCase):
    def test_pasted_shell_command_never_becomes_model_question(self):
        for text in [".venv/bin/python scripts/institution_demo.py --source T01", "cd '/tmp'", "python3 run.py", "./.venv/bin/trid chat"]:
            self.assertEqual(classify_input(text)[0], "notice")

    def test_controls_and_normal_questions(self):
        self.assertEqual(classify_input('/ exit'), ('command', 'exit'))
        self.assertEqual(classify_input('/examplde')[0], 'notice')
        self.assertEqual(classify_input('\x1b[B')[0], 'notice')
        self.assertEqual(classify_input('^[[B')[0], 'notice')
        self.assertEqual(classify_input('파이썬이 뭐야?'), ('question', '파이썬이 뭐야?'))
