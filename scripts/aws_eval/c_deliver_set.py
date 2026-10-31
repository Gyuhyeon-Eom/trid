"""AWS-EVAL-006: C delivers a dialogue set to the eval instance over SSM, encrypted to the instance's own key.

Run by C (profile trid-aws-eval-006-c-runner) — A never runs this with the private set. The instance generated a
key pair at boot, keeps the private key in RAM (/run tmpfs) and published only the certificate to
runs/<run>/input-cert.pem; its phase.json names the certificate's SHA-256 fingerprint and the set sha256 it expects.
This tool checks both, encrypts the set to that certificate (CMS, AES-256) locally, and sends the base64 ciphertext in
chunks with AWS-RunShellScript, then the ".ready" marker. What AWS keeps: SSM command history and CloudTrail hold the
ciphertext chunks and commands (no plaintext); the plaintext exists only on C's machine and in the instance's RAM
while it runs (shredded afterwards; the instance and its encrypted volume are deleted at termination).

  python3 scripts/aws_eval/c_deliver_set.py --run-id <run> --smoke             # the public 2-turn smoke set
  python3 scripts/aws_eval/c_deliver_set.py --run-id <run> --set <private.jsonl>  # C only
"""
import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGION, BUCKET, PREFIX = 'ap-northeast-2', 'trid-training-148761639313', 'eval/aws-eval-006'
CHUNK = 24000  # base64 characters per SendCommand, well under the parameter size limit


def aws(*args, data=False):
    r = subprocess.run(['aws', *args, '--region', REGION, *([] if data else ['--output', 'json'])], capture_output=True,
                       env={**os.environ, 'AWS_PAGER': '', 'AWS_PROFILE': os.environ.get('AWS_PROFILE', 'trid-aws-eval-006-c-runner')})
    if r.returncode:
        sys.exit(f'aws {" ".join(args[:2])} failed: {r.stderr.decode().strip()[:400]}')
    return r.stdout if data else json.loads(r.stdout or b'{}')


def smoke_set():
    """The same bytes the instance expects: first two turns of the first public CHAT-EVAL-004 scenario."""
    first = json.loads((ROOT / 'reports/coordination/CHAT-EVAL-004/public-20.jsonl').read_text().splitlines()[0])
    return (json.dumps({'id': first['id'], 'turns': first['turns'][:2]}, ensure_ascii=False) + '\n').encode()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-id', required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--smoke', action='store_true')
    g.add_argument('--set', type=Path)
    a = p.parse_args()

    payload = smoke_set() if a.smoke else a.set.read_bytes()
    base = f's3://{BUCKET}/{PREFIX}/runs/{a.run_id}'
    phase = json.loads(aws('s3', 'cp', f'{base}/phase.json', '-', data=True))
    if phase.get('phase') != 'awaiting-input':
        sys.exit(f'instance is not waiting for a set (phase {phase.get("phase")})')
    detail = phase['detail']
    if hashlib.sha256(payload).hexdigest() not in detail:
        sys.exit('this set is not the one the instance expects (sha256 differs); nothing sent')
    with tempfile.TemporaryDirectory() as tmp:
        cert = Path(tmp) / 'input-cert.pem'
        cert.write_bytes(aws('s3', 'cp', f'{base}/input-cert.pem', '-', data=True))
        fingerprint = subprocess.run(['openssl', 'x509', '-in', str(cert), '-noout', '-fingerprint', '-sha256'],
                                     capture_output=True, text=True, check=True).stdout.strip().split('=', 1)[1]
        if fingerprint not in detail:
            sys.exit('certificate fingerprint does not match the instance phase; nothing sent')
        cipher = subprocess.run(['openssl', 'cms', '-encrypt', '-binary', '-aes256', '-outform', 'DER', str(cert)],
                                input=payload, capture_output=True, check=True).stdout
    encoded = base64.b64encode(cipher).decode()
    instance = aws('ec2', 'describe-instances', '--filters', f'Name=tag:trid:run-id,Values={a.run_id}',
                   'Name=instance-state-name,Values=running', '--query', 'Reservations[].Instances[].InstanceId')
    if len(instance) != 1:
        sys.exit(f'expected one running instance for {a.run_id}, found {instance}')
    chunks = [encoded[i:i + CHUNK] for i in range(0, len(encoded), CHUNK)]
    commands = [f"printf '%s' '{c}' >> /run/trid-input/set.cms.b64" for c in chunks]
    commands.append('touch /run/trid-input/set.cms.b64.ready')
    for n, command in enumerate(commands, 1):
        sent = aws('ssm', 'send-command', '--instance-ids', instance[0], '--document-name', 'AWS-RunShellScript',
                   '--comment', f'aws-eval-006 set delivery {n}/{len(commands)}',
                   '--parameters', json.dumps({'commands': [command]}))
        command_id = sent['Command']['CommandId']
        for _ in range(60):
            time.sleep(2)
            r = subprocess.run(['aws', 'ssm', 'get-command-invocation', '--region', REGION, '--command-id', command_id,
                                '--instance-id', instance[0], '--output', 'json'], capture_output=True, text=True,
                               env={**os.environ, 'AWS_PAGER': ''})
            status = json.loads(r.stdout).get('Status') if r.returncode == 0 else 'Pending'
            if status in ('Success', 'Failed', 'Cancelled', 'TimedOut'):
                break
        if status != 'Success':
            sys.exit(f'chunk {n}/{len(commands)}: {status}; stopped (the instance fails the run if the set is incomplete)')
    print(json.dumps({'run_id': a.run_id, 'instance': instance[0], 'ciphertext_bytes': len(cipher),
                      'commands': len(commands), 'set_sha256': hashlib.sha256(payload).hexdigest()}))


if __name__ == '__main__':
    main()
