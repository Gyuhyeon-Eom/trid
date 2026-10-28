import contextlib
import http.client
import io
import json
import socket
import sys
import tempfile
import threading
import unittest
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from service_fixtures import REG, USERS, make_corpus  # noqa: E402

import torch
from safetensors.torch import save_file

from trid.model import Config, Trid
from trid.prompts import rag_passages
from trid.service.acl import load_users
from trid.service.audit import AuditLog
from trid.service.engine import Engine, SpanScoringBackend, locate
from trid.service.index import Index
from trid.service.ingest import ingest_dir
from trid.service.server import make_server
from trid.text import train_tokenizer


class FakeBackend:
    """Answers with a scripted RAG-grammar string; records the prompt it saw."""
    name = "fake"
    context = 1024

    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def tokens(self, prompt):
        return len(prompt) // 2

    def answer(self, prompt, passages):
        self.prompts.append((prompt, passages))
        return self.reply(passages), -3.0


def tiny_run(root):
    """A random 1-layer model + 512-token BPE trained on the fixture, saved like a real run."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rows = [{"prompt": line, "answer": "확인 불가 [D1]"} for line in REG.splitlines() if line.strip()] * 2
    tok = train_tokenizer(rows, root / "tokenizer.json", vocab_size=512)
    torch.manual_seed(0)
    cfg = Config(vocab_size=tok.get_vocab_size(), dim=32, layers=1, heads=4, hidden_dim=64, context_length=1024)
    (root / "config.json").write_text(json.dumps(asdict(cfg)))
    save_file(Trid(cfg).state_dict(), str(root / "model.safetensors"))
    return root


class EngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.index = Index.build(ingest_dir(make_corpus(cls.root)))
        cls.users = load_users(cls.root / "users.json")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def engine(self, backend, **kw):
        return Engine(self.index, backend, AuditLog(self.root / "audit.jsonl"), self.users, **kw)

    def test_answer_text_is_inside_chosen_passage_with_document_offset(self):
        backend = FakeBackend(lambda passages: f"8만 원 [{next(sid for sid, t in passages if '8만 원' in t)}]")
        answer = self.engine(backend, k=2).ask("숙박비의 상한은 1박당 얼마인가?", "kim")
        self.assertEqual(answer.kind, "answer")
        self.assertEqual(answer.value, "8만 원")
        self.assertIn(answer.value, answer.source.text)
        doc = self.index.documents[answer.source.doc_id]
        self.assertEqual(doc.text[answer.char_offset:answer.char_offset + len(answer.value)], "8만 원")
        self.assertEqual(answer.source.path[1:], ("제2장", "제12조"))
        prompt, passages = backend.prompts[-1]
        self.assertEqual(rag_passages(prompt), passages)  # prompt is the shared RAG format, [D1..] in hit order
        self.assertEqual([c[0] for c in answer.candidates], [h.passage.id for h in self.index.search("숙박비의 상한은 1박당 얼마인가?", {"staff"}, k=2)])
        self.assertTrue(all("\n" not in text for _, text in passages))
        row = json.loads((self.root / "audit.jsonl").read_text(encoding="utf-8").splitlines()[-1])
        self.assertEqual(row["source"]["sha256"], doc.sha256)
        self.assertEqual(doc.text[row["source"]["char_start"]:row["source"]["char_end"]], "8만 원")
        self.assertNotIn("8만 원", json.dumps(row, ensure_ascii=False))

    def test_value_spanning_a_line_break_is_located_verbatim(self):
        backend = FakeBackend(lambda passages: f"상한액 특별시 1박당 [{passages[0][0]}]")
        answer = self.engine(backend, k=1).ask("특별시 숙박비 상한액", "kim")
        self.assertEqual(answer.source.path[1:], ("별표 2",))
        self.assertEqual(answer.value, "상한액\n특별시 1박당")
        self.assertEqual(locate("a b", "x a\n  b"), (2, "a\n  b"))

    def test_text_not_in_passage_is_refused(self):
        with self.assertRaises(RuntimeError):
            self.engine(FakeBackend(lambda passages: f"12만 원 [{passages[0][0]}]")).ask("숙박비 상한", "kim")
        with self.assertRaises(RuntimeError):
            self.engine(FakeBackend(lambda passages: "8만 원입니다.")).ask("숙박비 상한", "kim")

    def test_abstention_and_acl(self):
        engine = self.engine(FakeBackend(lambda passages: "근거 충돌"))
        answer = engine.ask("숙박비 상한", "kim")
        self.assertEqual((answer.kind, answer.value, answer.source), ("근거 충돌", None, None))
        self.assertTrue(answer.candidates)
        for user in ("nobody", "notags"):
            answer = engine.ask("숙박비 상한", user)
            self.assertEqual((answer.kind, answer.candidates), ("확인 불가", []))
        hr_hits = engine.ask("급여 지급일", "hr_lee").candidates
        self.assertTrue(any(c[0].startswith("급여규정") for c in hr_hits))
        self.assertTrue(all(c[0].startswith("출장여비규정") for c in engine.ask("급여 지급일", "kim").candidates))  # kim: staff only

    def test_prompt_respects_context_and_max_passages(self):
        backend = FakeBackend(lambda passages: "확인 불가")
        backend.tokens = len
        backend.context = 400
        engine = self.engine(backend, k=5)
        hits = self.index.search("출장 여비 숙박비 일비", {"staff"}, k=5)
        engine.ask("출장 여비 숙박비 일비", "kim")
        prompt, passages = backend.prompts[-1]
        self.assertLessEqual(len(prompt) + 48, 400)
        self.assertLess(len(passages), len(hits))
        backend.context = 10 ** 6
        self.engine(backend, k=5, max_passages=5).ask("출장 여비 숙박비 일비", "kim")
        self.assertEqual(len(backend.prompts[-1][1]), min(5, len(hits)))
        self.engine(backend, k=5, max_passages=2).ask("출장 여비 숙박비 일비", "kim")
        self.assertEqual(len(backend.prompts[-1][1]), 2)

    def test_span_scoring_backend_copies_from_a_cited_passage(self):
        backend = SpanScoringBackend(tiny_run(self.root / "run"), "cpu")
        answer = self.engine(backend, k=1).ask("숙박비의 상한은?", "kim")
        if answer.kind == "answer":
            self.assertIn(answer.value, answer.source.text)
            self.assertIn(answer.source.id, [c[0] for c in answer.candidates])
            self.assertTrue(answer.text.endswith(f"[D{[c[0] for c in answer.candidates].index(answer.source.id) + 1}]"))
        else:
            self.assertIn(answer.kind, ("확인 불가", "근거 충돌"))
            self.assertEqual(answer.text, answer.kind)


class OfflineTest(unittest.TestCase):
    def test_ingest_index_and_ask_open_no_socket(self):
        def boom(*a, **k):
            raise AssertionError("network call attempted")

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            docs = make_corpus(root)
            run = tiny_run(root / "run")
            original = socket.socket
            socket.socket = boom
            try:
                index = Index.build(ingest_dir(docs))
                index.save(root / "index")
                index = Index.load(root / "index")
                engine = Engine(index, SpanScoringBackend(run, "cpu"), AuditLog(root / "audit.jsonl"), load_users(root / "users.json"), k=1)
                answer = engine.ask("일비는 얼마인가?", "kim")
            finally:
                socket.socket = original
        self.assertIn(answer.kind, ("answer", "확인 불가", "근거 충돌"))


class ServerTest(unittest.TestCase):
    def test_identity_comes_from_bearer_token_not_body(self):
        from trid.service.acl import issue_token, load_users
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            index = Index.build(ingest_dir(make_corpus(root)))
            cite = lambda passages: f"8만 원 [{next((sid for sid, t in passages if '8만 원' in t), passages[0][0])}]"
            users_path = root / "users.json"
            users_path.write_text(json.dumps(USERS), encoding="utf-8")
            with self.assertRaises(ValueError):  # list-form users: nobody can authenticate → refuse to serve
                make_server(Engine(index, FakeBackend(cite), AuditLog(root / "a.jsonl"), load_users(users_path), k=2), "127.0.0.1", 0)
            kim_token = issue_token(users_path, "kim")
            self.assertNotIn(kim_token, users_path.read_text())  # only the hash is stored
            engine = Engine(index, FakeBackend(cite), AuditLog(root / "audit.jsonl"), load_users(users_path), k=2)
            server = make_server(engine, "127.0.0.1", 0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
                def call(method, path, payload=None, token=None):
                    headers = {"Content-Type": "application/json"} if payload else {}
                    if token:
                        headers["Authorization"] = f"Bearer {token}"
                    conn.request(method, path, payload, headers)
                    r = conn.getresponse()
                    return r.status, json.loads(r.read())
                status, health = call("GET", "/health")
                self.assertEqual((status, health["documents"], health["backend"]), (200, 2, "fake"))
                hr_query = json.dumps({"query": "급여는 매월 며칠에 지급하는가?", "user": "hr_lee"}, ensure_ascii=False).encode("utf-8")
                self.assertEqual(call("POST", "/ask", hr_query)[0], 401)  # no token
                self.assertEqual(call("POST", "/ask", hr_query, token="wrong")[0], 401)
                status, answer = call("POST", "/ask", hr_query, token=kim_token)  # body claims hr_lee; token says kim
                self.assertEqual((status, answer["user"]), (200, "kim"))
                self.assertNotIn("급여", answer["source"]["doc_id"] if answer["source"] else "")  # hr document never offered
                body = json.dumps({"query": "숙박비의 상한은 1박당 얼마인가?"}, ensure_ascii=False).encode("utf-8")
                status, answer = call("POST", "/ask", body, token=kim_token)
                self.assertEqual((status, answer["value"], answer["kind"], answer["source"]["path"][-1]), (200, "8만 원", "answer", "제12조"))
                self.assertEqual(call("POST", "/ask", b"{}", token=kim_token)[0], 400)
                self.assertEqual(call("GET", "/other")[0], 404)
                engine.backend.reply = lambda passages: f"12만 원 [{passages[0][0]}]"  # not in any passage → refused
                with contextlib.redirect_stderr(io.StringIO()):
                    status, payload = call("POST", "/ask", body, token=kim_token)
                self.assertEqual((status, payload["error"]), (500, "RuntimeError"))
            finally:
                conn.close()
                server.shutdown()
                server.server_close()
            rows = [json.loads(l) for l in (root / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(rows and all(r["user"] == "kim" for r in rows))  # spoofed hr_lee never reached the audit log
            self.assertFalse(any("급여" in c["doc_id"] for r in rows for c in r["candidate_docs"]))


if __name__ == "__main__":
    unittest.main()
