import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from service_fixtures import HR, make_corpus  # noqa: E402

from trid.service.index import Index, block_key
from trid.service.ingest import ingest_dir, normalize


class IngestIndexTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.docs = make_corpus(cls.tmp.name)
        cls.documents = ingest_dir(cls.docs)
        cls.index = Index.build(cls.documents)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def doc(self, doc_id):
        return self.index.documents[doc_id]

    def passage(self, doc_id, *path):
        return next(p for p in self.doc(doc_id).passages if p.path[1:] == path)

    def test_ingest_sets_sha256_acl_redaction_and_offsets(self):
        hr = self.doc("급여규정")
        self.assertEqual(hr.sha256, hashlib.sha256(HR.encode("utf-8")).hexdigest())
        self.assertEqual(hr.acl, {"hr"})
        self.assertEqual(self.doc("출장여비규정").acl, {"staff"})
        self.assertIn("[전화]", hr.text)
        self.assertNotIn("010-1234-5678", hr.text)
        for d in self.documents:
            for p in d.passages:
                self.assertEqual(d.text[p.char_start:p.char_end], p.text)
        self.assertEqual(normalize("a\r\n\n\n\nb  \n"), "a\n\nb")

    def test_cross_document_reference_by_title(self):
        src = self.passage("급여규정", "제3조")
        dst = self.passage("출장여비규정", "제2장", "제12조")
        self.assertIn((src.id, dst.id), self.index.edges)
        self.assertEqual(self.index.refs[src.id], [dst.id])

    def test_search_is_acl_filtered_and_expands_referenced_조(self):
        self.assertEqual(self.index.search("숙박비 상한", set(), k=3), [])
        self.assertEqual(self.index.search("숙박비 상한", {"hr"}, k=3), [])
        hits = self.index.search("특별시 출장의 숙박비 상한", {"staff"}, k=2)
        self.assertEqual([h.via for h in hits[:2]], [None, None])
        self.assertEqual({h.passage.path[1:] for h in hits[:2]}, {("별표 2",), ("제2장", "제13조")})
        hits = self.index.search("서울특별시로 가는 출장", {"staff"}, k=1)
        self.assertEqual(hits[0].passage.path[1:], ("제2장", "제13조"))
        self.assertEqual([(h.passage.path[1:], h.via) for h in hits[1:]],
                         [(("제2장", "제12조"), f"ref←{hits[0].passage.id}")])  # 제13조 cites 제12조제2항
        self.assertEqual(self.index.search("서울특별시로 가는 출장", {"staff"}, k=1, hops=0)[1:], [])
        hits = self.index.search("출장 시 여비", {"staff", "hr"}, k=5)
        ids = [h.passage.id for h in hits]
        self.assertIn(self.passage("급여규정", "제3조").id, ids)
        self.assertIn(self.passage("출장여비규정", "제2장", "제12조").id, ids)
        self.assertTrue(all(h.passage.doc_id == "출장여비규정" for h in self.index.search("출장 시 여비", {"staff"}, k=5)))

    def test_head_of_split_조_is_pulled_in(self):
        from trid.service.types import Document, Passage
        text = "규정\n제1조(목적) 목적.\n제2조(지급) ① 숙박비 상한은 8만 원이다.\n② 일비는 2만 원이다.\n"
        a = Passage("r:0001", "r", text[3:14], 3, 14, ("규정", "제1조"))
        b = Passage("r:0002", "r", "제2조(지급) ① 숙박비 상한은 8만 원이다.", 15, 39, ("규정", "제2조", "①"))
        c = Passage("r:0003", "r", "② 일비는 2만 원이다.", 40, 53, ("규정", "제2조", "②"))
        index = Index.build([Document("r", "규정", "0" * 64, {"all"}, text, [a, b, c], [])])
        self.assertEqual(block_key(c.path), ("규정", "제2조"))
        hits = index.search("일비", {"all"}, k=1)
        self.assertEqual([(h.passage.id, h.via) for h in hits], [("r:0003", None), ("r:0002", "head←r:0003")])

    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            self.index.save(d)
            loaded = Index.load(d)
        self.assertEqual({k: v.to_dict() for k, v in loaded.documents.items()},
                         {k: v.to_dict() for k, v in self.index.documents.items()})
        self.assertEqual(loaded.edges, self.index.edges)
        before = [(h.passage.id, round(h.score, 5), h.via) for h in self.index.search("숙박비 상한", {"staff"}, k=3)]
        after = [(h.passage.id, round(h.score, 5), h.via) for h in loaded.search("숙박비 상한", {"staff"}, k=3)]
        self.assertEqual(before, after)
        self.assertTrue(before)


if __name__ == "__main__":
    unittest.main()
