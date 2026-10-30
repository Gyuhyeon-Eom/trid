"""Throwaway B-009: request spacing at the real policy interval (1.0 s) against a local HTTP server, measured on the
server (arrival = handler entry) and in the client request log. No remote requests.
Scenarios: single process; 3 processes (one with 3 threads); the same with late wake-ups (+0-120 ms per sleep) and
pre-send stalls (0-200 ms); and the old reservation design (commit 940a83c) under the same misbehaviour."""
import http.server
import json
import multiprocessing
import random
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = '/Users/eomgyuhyeon/사업/trid'
INTERVAL = 1.0


class Server:
    def __enter__(self):
        self.arrivals = []
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                outer.arrivals.append(time.time())
                time.sleep(random.uniform(0, 0.3))
                self.send_response(200)
                self.send_header('Content-Length', '2')
                self.end_headers()
                self.wfile.write(b'ok')

            def log_message(self, *a):
                pass
        self.httpd = http.server.ThreadingHTTPServer(('127.0.0.1', 0), H)
        self.url = f'http://127.0.0.1:{self.httpd.server_address[1]}/'
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *e):
        self.httpd.shutdown()
        self.httpd.server_close()


def burst(module_dir, shared_dir, url, n, threads, misbehave, seed):
    random.seed(seed)
    sys.path.insert(0, module_dir)
    sys.path.insert(0, ROOT)
    import collect_assembly_minutes as am
    if misbehave:
        real_sleep = time.sleep
        am.time.sleep = lambda s: real_sleep(s + random.uniform(0, 0.12))
    log = Path(tempfile.mkdtemp()) / 'r.jsonl'
    client = am.Client(am.HostLimiter(INTERVAL, host='proof-local', shared_dir=shared_dir), log)
    if misbehave:
        send = client.session.request

        def stalled(*a, **kw):
            time.sleep(random.uniform(0, 0.2))
            return send(*a, **kw)
        client.session.request = stalled
    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(lambda _: client.request('GET', url, am.Metrics()), range(n)))
    return [json.loads(x)['t'] for x in log.read_text().split('\n') if x]


def scenario(name, module_dir, plan):
    with Server() as server, tempfile.TemporaryDirectory() as d:
        with multiprocessing.get_context('spawn').Pool(len(plan)) as pool:
            logged = pool.starmap(burst, [(module_dir, d, server.url, *p, i) for i, p in enumerate(plan)])
        arr = sorted(server.arrivals)
    starts = sorted(t for b in logged for t in b)
    gaps = [b - a for a, b in zip(arr, arr[1:])]
    lgaps = [b - a for a, b in zip(starts, starts[1:])]
    out = {'scenario': name, 'processes': len(plan), 'threads': [p[1] for p in plan], 'requests': len(arr),
           'server_min_gap_s': round(min(gaps), 4), 'server_gaps_below_1s': sum(g < INTERVAL for g in gaps),
           'server_max_in_1s_window': max(sum(1 for y in arr[i:] if y - x < INTERVAL) for i, x in enumerate(arr)),
           'log_min_gap_s': round(min(lgaps), 4), 'log_gaps_below_1s': sum(g < INTERVAL for g in lgaps),
           'mean_req_per_s': round((len(arr) - 1) / (arr[-1] - arr[0]), 3)}
    print(json.dumps(out), flush=True)
    return out


if __name__ == '__main__':
    new, old = f'{ROOT}/scripts', '/tmp/trid-corpus/old-limiter'
    results = [
        scenario('single process, 1 thread', new, [(10, 1, False)]),
        scenario('3 processes (one with 3 threads)', new, [(4, 3, False), (4, 1, False), (4, 1, False)]),
        scenario('3 processes + late wake-ups + pre-send stalls', new, [(4, 3, True), (4, 1, True), (4, 1, True)]),
        scenario('OLD reservation design (940a83c), same misbehaviour', old, [(4, 3, True), (4, 1, True), (4, 1, True)]),
    ]
    json.dump(results, open('/tmp/trid-corpus/qa/proof-actual-gap.json', 'w'), indent=1)
