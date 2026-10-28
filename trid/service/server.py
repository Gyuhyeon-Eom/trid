"""Stdlib JSON API, loopback by default. Requests are served one at a time (one model, one CPU box).

  GET  /health                         {"status": "ok", "documents": n, "passages": m, "backend": "span_scoring"}
  POST /ask  Authorization: Bearer t   {"query": "..."} -> Answer.to_dict() + {"user": ..., "seconds": ...}

Identity comes only from the bearer token (`trid token --users users.json --user kim`); a `user` field in the
body is ignored. Starting without any token-bearing user refuses, so an index with list-form users.json
cannot be exposed over HTTP by accident.
"""
import json
import sys
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_BODY = 64 * 1024


def make_server(engine, host="127.0.0.1", port=8321):
    from .acl import authenticate
    if not any(getattr(tags, "token_sha256", None) for tags in engine.users.values()):
        raise ValueError("HTTP 서비스에는 토큰이 있는 사용자가 한 명 이상 필요합니다: trid token --users <users.json> --user <이름>")
    lock = threading.Lock()  # one model: questions are answered one at a time, connections are not

    class Handler(BaseHTTPRequestHandler):
        server_version = "trid/0.2"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # audit log is the record; no request lines on stderr
            pass

        def _json(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/health":
                return self._json(404, {"error": "not found"})
            self._json(200, {"status": "ok", "documents": len(engine.index.documents),
                             "passages": len(engine.index.passages), "backend": engine.backend.name})

        def do_POST(self):
            if self.path != "/ask":
                return self._json(404, {"error": "not found"})
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(min(length, MAX_BODY + 1)) if length > 0 else b""  # keep the connection in sync
            auth = self.headers.get("Authorization") or ""
            user = authenticate(engine.users, auth[7:]) if auth.startswith("Bearer ") else None
            if user is None:
                return self._json(401, {"error": "Authorization: Bearer <token> 이 필요합니다."})
            if length <= 0 or length > MAX_BODY:
                return self._json(413 if length > MAX_BODY else 400, {"error": "본문이 비어 있거나 너무 깁니다."})
            try:
                payload = json.loads(raw.decode("utf-8"))
                query = payload["query"]
                if not isinstance(query, str):
                    raise ValueError
            except (ValueError, KeyError, TypeError):
                return self._json(400, {"error": '{"query": str} JSON이 필요합니다. 사용자는 토큰으로 결정됩니다.'})
            started = time.perf_counter()
            try:
                with lock:
                    answer = engine.ask(query, user)
            except ValueError as e:
                return self._json(400, {"error": str(e)})
            except Exception as e:  # keep the connection protocol-clean; details stay on the server side
                sys.stderr.write(f"trid serve: {type(e).__name__}: {e}\n")
                return self._json(500, {"error": type(e).__name__})
            self._json(200, {**answer.to_dict(), "user": user, "seconds": round(time.perf_counter() - started, 4)})

    return ThreadingHTTPServer((host, port), Handler)


def serve(engine, host="127.0.0.1", port=8321):
    server = make_server(engine, host, port)
    print(f"trid serve | http://{host}:{server.server_address[1]} | 문서 {len(engine.index.documents)} · "
          f"문단 {len(engine.index.passages)} · {engine.backend.name} | Ctrl-C 종료")
    try:
        server.serve_forever()
    finally:
        server.server_close()
