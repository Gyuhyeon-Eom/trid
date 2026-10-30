"""Start/stop the temporary Mac-hosted demo and deploy its Cloudflare proxy."""
import argparse
import glob
import http.client
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / '.shared-demo'
DEPLOY = ROOT / 'deploy/shared-demo'
PORT = 51873
WORKER = 'trid-model-test'


def save(name, value):
    path = RUNTIME / name
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.chmod(0o600)
    temporary.replace(path)


def status():
    path = RUNTIME / 'state.json'
    return json.loads(path.read_text()) if path.exists() else {}


def alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def stop_child(child):
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill(); child.wait()


def tunnel_ready():
    connection = http.client.HTTPConnection('127.0.0.1', 20241, timeout=3)
    try:
        connection.request('GET', '/ready')
        response = connection.getresponse()
        body = json.loads(response.read())
        return response.status == 200 and body.get('readyConnections', 0) > 0
    except (OSError, ValueError, http.client.HTTPException):
        return False
    finally:
        connection.close()


def external_ready(origin, secret):
    # Keep the proxy credential out of process arguments and request logs.
    config = f'header = "X-Trid-Proxy: {secret}"\nheader = "X-Trid-Host: test.tridatum.co"\n'
    try:
        # New tunnel hostnames may be negatively cached by a local DNS resolver.
        result = subprocess.run(['/usr/bin/curl', '-fsS', '--doh-url', 'https://1.1.1.1/dns-query',
                                 '--max-time', '8', '--config', '-', origin + '/healthz'],
                                input=config, capture_output=True, text=True, timeout=10)
        return result.returncode == 0 and json.loads(result.stdout).get('ok') is True
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def start_tunnel(children, secret):
    tunnel_bin = shutil.which('cloudflared')
    if not tunnel_bin:
        raise RuntimeError('cloudflared not installed')
    with (RUNTIME / 'tunnel.log').open('wb') as log:
        tunnel = subprocess.Popen([tunnel_bin, 'tunnel', '--no-autoupdate', '--protocol', 'http2',
                                   '--metrics', '127.0.0.1:20241', '--url', f'http://127.0.0.1:{PORT}'],
                                  stdout=log, stderr=log)
    children.append(tunnel)
    try:
        deadline = time.monotonic() + 60
        origin = None
        while time.monotonic() < deadline:
            if tunnel.poll() is not None:
                raise RuntimeError('Tunnel exited during startup')
            matches = re.findall(r'https://[a-z0-9-]+\.trycloudflare\.com', (RUNTIME / 'tunnel.log').read_text(errors='replace'))
            if matches:
                origin = matches[0]
                if tunnel_ready() and external_ready(origin, secret):
                    return tunnel, origin
            time.sleep(1)
        raise RuntimeError('Tunnel readiness timed out')
    except BaseException:
        stop_child(tunnel)
        raise


def supervise(hours):
    RUNTIME.mkdir(mode=0o700, exist_ok=True)
    os.umask(0o077)
    secret_path = RUNTIME / 'proxy-secret'
    if not secret_path.exists():
        secret_path.write_text(secrets.token_urlsafe(48))
    secret_path.chmod(0o600)
    children = []
    def stop(*args):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    info = {'supervisor_pid': os.getpid(), 'expires_at': time.time() + hours * 3600, 'status': 'starting'}
    save('state.json', info)
    try:
        with (RUNTIME / 'server.log').open('wb') as log:
            backend = subprocess.Popen([sys.executable, str(ROOT / 'scripts/shared_demo.py'), '--secret-file', str(secret_path), '--port', str(PORT)], cwd=ROOT, stdout=log, stderr=log)
        children.append(backend)
        secret = secret_path.read_text().strip()
        ready = False
        for _ in range(240):  # versions file may start llama.cpp servers and hash their model files
            if backend.poll() is not None:
                raise RuntimeError('Model server failed. See .shared-demo/server.log')
            try:
                c = http.client.HTTPConnection('127.0.0.1', PORT, timeout=1)
                c.request('GET', '/healthz', headers={'X-Trid-Proxy': secret, 'X-Trid-Host': 'test.tridatum.co'})
                r = c.getresponse(); r.read(); c.close()
                if r.status == 200:
                    ready = True; break
            except OSError:
                pass
            time.sleep(.5)
        if not ready:
            raise RuntimeError('Model server readiness timeout')
        tunnel, origin = start_tunnel(children, secret)
        caffeine = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        children.append(caffeine)
        info.update(status='running', origin_url=origin, backend_pid=backend.pid, tunnel_pid=tunnel.pid,
                    connection_status='ready', recoveries=0, last_health_at=time.time())
        save('state.json', info)
        failures = 0
        next_check = time.monotonic() + 30
        needs_publish = False
        while time.time() < info['expires_at']:
            if backend.poll() is not None or caffeine.poll() is not None:
                raise RuntimeError('Model server or sleep prevention process exited')
            if time.monotonic() >= next_check:
                healthy = tunnel.poll() is None and tunnel_ready() and external_ready(origin, secret)
                failures = 0 if healthy else failures + 1
                info.update(last_health_at=time.time(), connection_status='ready' if healthy else 'unhealthy')
                save('state.json', info)
                if failures >= 2:
                    print('Tunnel health failed twice; reconnecting.', flush=True)
                    info['connection_status'] = 'recovering'; save('state.json', info)
                    stop_child(tunnel)
                    try:
                        tunnel, origin = start_tunnel(children, secret)
                        info.update(origin_url=origin, tunnel_pid=tunnel.pid, recoveries=info['recoveries'] + 1)
                        needs_publish = True
                        failures = 0
                        save('state.json', info)
                    except (RuntimeError, OSError) as error:
                        print(f'Tunnel recovery failed: {type(error).__name__}; retrying after 30 seconds.', flush=True)
                if needs_publish and failures == 0:
                    try:
                        deploy(put_secret=False)
                        needs_publish = False
                        info['connection_status'] = 'ready'; save('state.json', info)
                        print('Tunnel recovered and domain updated.', flush=True)
                    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
                        info['connection_status'] = 'publish_pending'; save('state.json', info)
                        print(f'Domain update failed: {type(error).__name__}; retrying after 30 seconds.', flush=True)
                next_check = time.monotonic() + 30
            time.sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        for child in reversed(children):
            stop_child(child)
        info['status'] = 'stopped'
        save('state.json', info)


def wrangler():
    installed = glob.glob(str(Path.home() / '.npm/_npx/*/node_modules/wrangler/bin/wrangler.js'))
    if not installed or not shutil.which('node'):
        raise RuntimeError('Node and Wrangler must be installed')
    return [shutil.which('node'), installed[0]]


def deploy(put_secret=True):
    info = status()
    if info.get('status') != 'running' or not alive(info.get('supervisor_pid')):
        raise RuntimeError('Run start first')
    cfg = {'name': WORKER, 'account_id': 'a57b51f32c8d50f90c4f30debb508aa7', 'main': 'worker.mjs',
           'compatibility_date': '2026-10-01', 'workers_dev': False, 'preview_urls': False,
           'routes': [{'pattern': 'test.tridatum.co', 'custom_domain': True}],
           'vars': {'ORIGIN_URL': info['origin_url']},
           'observability': {'enabled': False},
           'ratelimits': [{'name': 'DEMO_RATE_LIMIT', 'namespace_id': '2026102701', 'simple': {'limit': 30, 'period': 60}}]}
    (DEPLOY / 'wrangler.json').write_text(json.dumps(cfg, indent=2))
    cmd = wrangler()
    secret = (RUNTIME / 'proxy-secret').read_text().strip()
    if put_secret:
        subprocess.run(cmd + ['secret', 'put', 'PROXY_SECRET'], cwd=DEPLOY, input=secret + '\n', text=True, check=True, timeout=90)
    subprocess.run(cmd + ['deploy'], cwd=DEPLOY, check=True, timeout=90)
    print('https://test.tridatum.co', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['start', 'stop', 'status', 'deploy', 'supervise'])
    parser.add_argument('--hours', type=float, default=24)
    args = parser.parse_args()
    if not 0 < args.hours <= 168:
        parser.error('--hours must be between 0 and 168')
    if args.command == 'supervise':
        supervise(args.hours)
    elif args.command == 'status':
        info = status()
        info['supervisor_alive'] = alive(info.get('supervisor_pid'))
        print(json.dumps(info, ensure_ascii=False, indent=2))
    elif args.command == 'start':
        info = status()
        if alive(info.get('supervisor_pid')):
            print('Demo supervisor is already running.'); return
        RUNTIME.mkdir(mode=0o700, exist_ok=True)
        with (RUNTIME / 'supervisor.log').open('wb') as log:
            p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'supervise', '--hours', str(args.hours)], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        for _ in range(180):
            info = status()
            if info.get('supervisor_pid') == p.pid and info.get('status') == 'running':
                print('Demo ready. Run deploy to connect test.tridatum.co.'); return
            if p.poll() is not None:
                raise RuntimeError('Startup failed. See .shared-demo/supervisor.log')
            time.sleep(.5)
        raise RuntimeError('Startup pending; inspect status and logs.')
    elif args.command == 'stop':
        info = status(); pid = info.get('supervisor_pid')
        if alive(pid):
            command = subprocess.run(['/bin/ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True).stdout
            if 'share_demo_control.py supervise' not in command:
                raise RuntimeError('PID belongs to another process; refusing to stop it')
            os.kill(pid, signal.SIGTERM)
            print('Stopping temporary demo. Domain will show the offline page.')
        else:
            print('Demo is stopped.')
    else:
        deploy()


if __name__ == '__main__':
    main()
