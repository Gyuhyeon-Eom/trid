"""One logged, host-limited HTTP request for ad hoc checks (robots.txt, a detail page, a script file).

Every request B makes goes through the shared HostLimiter (one request in flight per host across processes, next start
>= previous end + --interval) and is appended to logs/requests-<host>.jsonl, so ad hoc checks follow the same rule as
collectors (B-017 review: unlogged curl checks with 1 s gaps). Writes the body to --out (or prints status/size).

  uv run -q --no-project --python 3.12 --with requests python scripts/logged_get.py URL [--out FILE] [--post k=v ...]
"""
import argparse
import sys
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def host_key(url):
    host = urlsplit(url).hostname or ''
    return host[4:] if host.startswith('www.') else host


def main():
    import collect_assembly_minutes as am
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('urls', nargs='+')
    p.add_argument('--out', help='file (one URL) or directory (several URLs; files named by position)')
    p.add_argument('--post', nargs='*', help='form fields k=v; sends POST instead of GET')
    p.add_argument('--interval', type=float, default=2.0)
    args = p.parse_args()
    metrics = am.Metrics()
    clients = {}
    for i, url in enumerate(args.urls):
        host = host_key(url)
        if host not in clients:
            clients[host] = am.Client(am.HostLimiter(args.interval, host=host), ROOT / f'logs/requests-{host}.jsonl')
        data = dict(kv.split('=', 1) for kv in args.post) if args.post is not None else None
        r = clients[host].request('POST' if data is not None else 'GET', url, metrics, data=data)
        if args.out:
            dest = Path(args.out) / f'{i:03d}' if len(args.urls) > 1 else Path(args.out)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(r.content)
        print(r.status_code, len(r.content), r.headers.get('Content-Type'), url)


if __name__ == '__main__':
    main()
