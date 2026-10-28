import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from service_fixtures import make_corpus  # noqa: E402

from trid.service.acl import document_tags, load_users, user_tags, visible
from trid.service.audit import AuditLog
from trid.service.types import Answer, Document, Passage


class AclTest(unittest.TestCase):
    def test_deny_by_default(self):
        self.assertFalse(visible(set(), {"staff"}))
        self.assertFalse(visible({"staff"}, set()))
        self.assertFalse(visible({"staff"}, {"hr"}))
        self.assertTrue(visible({"staff", "hr"}, {"hr"}))
        self.assertEqual(user_tags({"kim": {"staff"}}, "unknown"), set())
        self.assertEqual(load_users("/nonexistent/users.json"), {})

    def test_sidecar_beats_directory_config_and_nested_dirs_inherit(self):
        with tempfile.TemporaryDirectory() as d:
            docs = make_corpus(d)
            self.assertEqual(document_tags(docs / "급여규정.txt", docs), {"hr"})
            self.assertEqual(document_tags(docs / "출장여비규정.txt", docs), {"staff"})
            (docs / "sub").mkdir()
            (docs / "sub" / "a.txt").write_text("x", encoding="utf-8")
            self.assertEqual(document_tags(docs / "sub" / "a.txt", docs), {"staff"})
            (docs / "acl.json").write_text(json.dumps({"files": {"sub/a.txt": ["exec"]}}), encoding="utf-8")
            self.assertEqual(document_tags(docs / "sub" / "a.txt", docs), {"exec"})  # keyed relative to the config dir
            (docs / "sub" / "acl.json").write_text(json.dumps({"files": {"a.txt": ["sub"]}}), encoding="utf-8")
            self.assertEqual(document_tags(docs / "sub" / "a.txt", docs), {"sub"})  # nearest config wins
            self.assertEqual(document_tags(docs / "출장여비규정.txt", docs), set())  # no default any more → nobody
            users = load_users(Path(d) / "users.json")
            self.assertEqual(user_tags(users, "hr_lee"), {"staff", "hr"})
            self.assertEqual(user_tags(users, "notags"), set())


class AuditTest(unittest.TestCase):
    def test_row_has_hashes_and_offsets_but_no_text(self):
        text = "급여규정\n제2조(지급일) 급여는 매월 25일에 지급한다."
        p = Passage("급여:0001", "급여", text[5:], 5, len(text), ("급여규정", "제2조"))
        doc = Document("급여", "급여규정", "ab" * 32, {"hr"}, text, [p], [])
        answer = Answer("매월 25일 [D1]", "answer", "매월 25일", p, text.index("매월"), -1.5, [(p.id, 2.0, None)])
        with tempfile.TemporaryDirectory() as d:
            log = AuditLog(Path(d) / "audit.jsonl")
            log.record("kim", "급여는 언제 지급하나?", answer, {"급여": doc}, 0.42)
            log.record("kim", "급여는 언제 지급하나?", Answer("확인 불가", "확인 불가", None, None, None, 0.0, []), {"급여": doc})
            lines = (Path(d) / "audit.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        row = json.loads(lines[0])
        self.assertEqual(row["user"], "kim")
        self.assertEqual(len(row["query_sha256"]), 64)
        self.assertEqual(row["source"], {"doc_id": "급여", "sha256": "ab" * 32, "passage_id": "급여:0001",
                                        "path": ["급여규정", "제2조"], "char_start": 18, "char_end": 24})
        self.assertEqual(text[row["source"]["char_start"]:row["source"]["char_end"]], "매월 25일")
        self.assertEqual(row["candidate_docs"], [{"doc_id": "급여", "sha256": "ab" * 32}])
        for secret in ("매월 25일", "지급한다", "급여는 언제"):
            self.assertNotIn(secret, lines[0])
        self.assertEqual(json.loads(lines[1])["kind"], "확인 불가")
        self.assertIsNone(json.loads(lines[1])["source"])


if __name__ == "__main__":
    unittest.main()
