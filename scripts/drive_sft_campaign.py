"""Drive a long SFT campaign on SageMaker: chain resumable segments, evaluate milestones, prune S3 recovery checkpoints,
publish a structured status. Runs locally with the AWS CLI; each loop is idempotent, so the driver itself can be
restarted at any time (state in --state, the authoritative training state in S3).

Per arm: when no segment job is running, the next segment (scripts/sm_sft_campaign.sh, same checkpoint S3 URI) is
launched until training.json reports `complete`. Two consecutive failed/stopped segments halt that arm (no endless
retries); the failure reasons are kept in the state. A milestone directory (milestones/step-N with state.json) gets
one evaluation job (scripts/sm_eval_weights.sh); after completion the validation-selected weights are evaluated too.
S3 keeps the newest two complete recovery checkpoints; older ones and stale `.tmp` objects are deleted (SageMaker's
checkpoint sync never deletes). Milestones, best-valid weights and A-003/A-004 runs are never pruned.

  python scripts/drive_sft_campaign.py --campaign reports/coordination/A-006/campaign.json
"""
import argparse
import datetime
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGION = 'ap-northeast-2'
AWS_CLI = os.environ.get('TRID_AWS_CLI', 'aws')  # the Mac mini sets /opt/homebrew/bin/aws (CLI v2, `aws login`)
# Error text of expired or missing credentials (CLI v1/v2, STS, `aws login`/SSO sessions).
AUTH_MARKERS = ('ExpiredToken', 'InvalidClientTokenId', 'Unable to locate credentials', 'could not be found',
                'Token has expired', 'session has expired', 'UnauthorizedSSOToken', 'RequestExpired',
                'aws login', 'Error loading SSO Token', 'UnrecognizedClientException')


def aws(*args, check=True):
    if os.environ.get('TRID_AWS_TRACE'):  # records every AWS call (for least-privilege policy review)
        with open(os.environ['TRID_AWS_TRACE'], 'a') as f:
            f.write(json.dumps(args) + '\n')
    out = subprocess.run([AWS_CLI, '--region', REGION, *args], capture_output=True, text=True)
    if check and out.returncode:
        raise RuntimeError(f'aws {" ".join(args[:3])}: {out.stderr.strip()[-400:]}')
    return out


def s3_json(uri):
    out = aws('s3', 'cp', uri, '-', check=False)
    return json.loads(out.stdout) if out.returncode == 0 and out.stdout.strip() else None


def s3_keys(uri):
    bucket, _, prefix = uri.removeprefix('s3://').partition('/')
    keys, token = [], None
    while True:
        args = ['s3api', 'list-objects-v2', '--bucket', bucket, '--prefix', prefix, '--output', 'json']
        if token:
            args += ['--continuation-token', token]
        page = json.loads(aws(*args).stdout or '{}')
        keys += [o['Key'] for o in page.get('Contents', [])]
        token = page.get('NextContinuationToken')
        if not token:
            return bucket, keys


def job_status(name):
    d = json.loads(aws('sagemaker', 'describe-training-job', '--training-job-name', name).stdout)
    return d['TrainingJobStatus'], d.get('FailureReason')


def launch(entry, run, config, instance, max_runtime, data_s3, checkpoint_s3, env, purpose, spot=False):
    out = subprocess.run(['.venv/bin/python', 'scripts/launch_sm_pretrain.py', '--entry', entry, '--run', run, '--config', config,
                          '--steps', '0', '--instance-type', instance, '--max-runtime', str(max_runtime), '--data-s3', data_s3,
                          '--checkpoint-s3', checkpoint_s3, '--env', *[f'{k}={v}' for k, v in env.items()], '--purpose', purpose,
                          *(['--spot'] if spot else [])],
                         cwd=ROOT, capture_output=True, text=True)
    if out.returncode:
        return None, out.stderr.strip()[-400:]
    return json.loads(out.stdout)['job'], None


def busy_types():
    """(instance type, managed spot) of every in-progress training job in the account; on-demand and spot quotas
    are separate and 1 per g6e type."""
    names = json.loads(aws('sagemaker', 'list-training-jobs', '--status-equals', 'InProgress', '--max-results', '100').stdout)
    busy = set()
    for j in names['TrainingJobSummaries']:
        d = json.loads(aws('sagemaker', 'describe-training-job', '--training-job-name', j['TrainingJobName']).stdout)
        busy.add((d['ResourceConfig']['InstanceType'], bool(d.get('EnableManagedSpotTraining'))))
    return busy


def quota_wait(err):
    return err is not None and 'ResourceLimitExceeded' in err


def prune(arm):
    """Keep the newest two complete recovery checkpoints in S3; delete older ones and stale .tmp objects."""
    bucket, keys = s3_keys(f"{arm['checkpoint_s3']}{arm['run']}/sft/checkpoints/")
    dirs = {}
    for k in keys:
        name = k.split('/sft/checkpoints/')[1].split('/')[0]
        dirs.setdefault(name, []).append(k)
    complete = sorted(n for n, ks in dirs.items() if not n.endswith('.tmp') and any(k.endswith('/state.json') for k in ks))
    keep = set(complete[-2:])
    doomed = [k for n, ks in dirs.items() if n not in keep for k in ks]
    for i in range(0, len(doomed), 1000):
        objects = {'Objects': [{'Key': k} for k in doomed[i:i + 1000]], 'Quiet': True}
        aws('s3api', 'delete-objects', '--bucket', bucket, '--delete', json.dumps(objects))
    return sorted(keep), len(doomed)


def running_job(run):
    """An in-progress job launched for this run (launcher names jobs trid-pt-<run>-<stamp>), e.g. after a driver restart."""
    prefix = f'trid-pt-{run}'[:48].rstrip('-')
    jobs = json.loads(aws('sagemaker', 'list-training-jobs', '--status-equals', 'InProgress', '--name-contains', prefix,
                          '--max-results', '10').stdout)['TrainingJobSummaries']
    return jobs[0]['TrainingJobName'] if jobs else None


def spent_usd(state, prices):
    """Billable seconds (elapsed seconds while running) x the configured hourly price, over every campaign job."""
    total = 0.0
    for a in state['arms'].values():
        for j in [*a['segments'], *a['evals'].values()]:
            if not j.get('job'):
                continue
            d = json.loads(aws('sagemaker', 'describe-training-job', '--training-job-name', j['job']).stdout)
            seconds = d.get('BillableTimeInSeconds') or d.get('TrainingTimeInSeconds') or 0
            if d['TrainingJobStatus'] == 'InProgress':
                start = d.get('TrainingStartTime') or d['CreationTime']
                seconds = max(seconds, (datetime.datetime.now(datetime.timezone.utc)
                                        - datetime.datetime.fromisoformat(str(start))).total_seconds())
            total += seconds / 3600 * prices[d['ResourceConfig']['InstanceType']]
    return total


def step(campaign, state):
    now = datetime.datetime.now().isoformat(timespec='seconds')
    common = campaign['common']
    monitor = bool(common.get('monitor_only'))
    busy = set() if monitor else busy_types()
    cap, prices = common.get('cost_cap_usd'), common.get('prices_usd_per_hour', {})
    state['spent_usd'] = round(spent_usd(state, prices), 2) if cap else None
    for name, arm in campaign['arms'].items():
        s = state['arms'].setdefault(name, {'segments': [], 'consecutive_failures': 0, 'evals': {}, 'halted': None, 'done': False})
        if not s['segments'] and arm.get('job'):  # a job launched elsewhere (approved stage) to monitor from the start
            s['segments'].append({'job': arm['job'], 'status': 'InProgress', 'failure': None, 'launched_at': now, 'seeded': True})
        run_s3 = f"{arm['checkpoint_s3']}{arm['run']}/sft/"
        training = s3_json(run_s3 + 'training.json') or {}
        metrics = s3_json(run_s3 + 'metrics.json') or []
        seg = s['segments'][-1] if s['segments'] else None
        if seg and seg['status'] == 'InProgress':
            seg['status'], seg['failure'] = job_status(seg['job'])
            if seg['status'] in ('Failed', 'Stopped'):
                s['consecutive_failures'] += 1
            elif seg['status'] == 'Completed':
                s['consecutive_failures'] = 0
        running = seg is not None and seg['status'] == 'InProgress'
        if training.get('status') == 'complete' and not running:
            s['done'] = True
        if not running and not s['done'] and not s['halted']:
            limit = arm.get('max_segments', common.get('max_segments'))
            launches = arm.get('max_launches', common.get('max_launches'))
            adopted = running_job(arm['run'])
            if adopted:
                s['segments'].append({'job': adopted, 'status': 'InProgress', 'failure': None, 'launched_at': now,
                                      'adopted': True, 'from_step': training.get('completed_steps')})
            elif common.get('monitor_only'):
                # Watch and collect only: jobs are launched elsewhere under an approved stage; never launch or retry.
                if seg and seg['status'] in ('Failed', 'Stopped'):
                    s['halted'] = f"{now}: monitored job ended {seg['status']}: {seg.get('failure')}"
                else:
                    s['waiting'] = f'{now}: monitor_only; no running job for {arm["run"]}'
            elif launches is not None and len(s['segments']) >= launches:
                s['halted'] = f'{now}: launch limit {launches} reached (no retries beyond the approved stage)'
            elif limit and len([g for g in s['segments'] if g['status'] == 'Completed']) >= limit:
                s['halted'] = f'{now}: segment limit {limit} reached (smoke/limited campaign)'
            elif s['consecutive_failures'] >= 2:
                s['halted'] = f"{now}: two consecutive failed segments; last: {seg.get('failure') if seg else None}"
            elif cap and state['spent_usd'] + common['max_runtime'] / 3600 * prices[arm['instance']] > cap:
                s['halted'] = f"{now}: cost cap {cap} USD: spent {state['spent_usd']} + next segment worst case exceeds it"
            else:
                session_steps = arm.get('session_steps', common.get('session_steps'))
                env = {'INIT_S3': arm['init_s3'], 'INIT_STEP': arm['init_step'], 'SFT_STEPS': common['steps'],
                       'SAVE_STEPS': ','.join(map(str, common['save_steps'])), 'CHECKPOINT_EVERY': common['checkpoint_every'],
                       **({'SESSION_STEPS': session_steps} if session_steps else {'SEGMENT_SECONDS': common['segment_seconds']})}
                spot = bool(arm.get('spot', common.get('spot')))
                if (arm['instance'], spot) in busy:
                    s['waiting'] = f'{now}: {arm["instance"]} busy; segment launch deferred'
                    job, err = None, 'deferred'
                else:
                    job, err = launch('sm_sft_campaign.sh', arm['run'], arm['config'], arm['instance'], common['max_runtime'],
                                      common['data_s3'], arm['checkpoint_s3'], env,
                                      f"{campaign['id']} {name} segment {len(s['segments']) + 1} (resume from S3)", spot)
                if job or not (err == 'deferred' or quota_wait(err)):
                    s['segments'].append({'job': job, 'status': 'InProgress' if job else 'LaunchFailed', 'failure': err,
                                          'launched_at': now, 'from_step': training.get('completed_steps')})
                    if job:
                        busy.add((arm['instance'], spot))
                    else:
                        s['consecutive_failures'] += 1
        if not monitor:  # pruning deletes S3 objects; a monitor never writes outside its status key
            kept, deleted = prune(arm)
            s['recovery_kept'], s['recovery_pruned_objects'] = kept, s.get('recovery_pruned_objects', 0) + deleted
        # Milestones and, at the end, the validation-selected weights get one evaluation job each.
        _, keys = s3_keys(run_s3 + 'milestones/')
        ready = sorted({k.split('/milestones/')[1].split('/')[0] for k in keys if k.endswith('/state.json')})
        targets = [(m, f"{run_s3}milestones/{m}/") for m in ready]
        if s['done']:
            targets.append(('best-valid', run_s3))
        for label, weights in targets:
            ev = s['evals'].get(label)
            if ev and ev.get('status') == 'InProgress':
                ev['status'], ev['failure'] = job_status(ev['job'])
                continue
            if ev and (ev['status'] == 'Completed' or ev.get('attempts', 0) >= 2):
                continue
            free = [i for i in common['eval_instances'] if (i, False) not in busy]
            if not free or (cap and state['spent_usd'] + common['eval_max_runtime'] / 3600 * prices[free[0]] > cap):
                continue
            job, err = launch('sm_eval_weights.sh', f"{arm['run']}-eval", arm['config'], free[0], common['eval_max_runtime'],
                              common['data_s3'], f"{arm['checkpoint_s3']}{arm['run']}-eval/",
                              {'RUN_S3': run_s3, 'WEIGHTS_S3': weights, 'LABEL': label}, f"{campaign['id']} {name} eval {label}")
            if quota_wait(err):
                continue
            s['evals'][label] = {'job': job, 'status': 'InProgress' if job else 'LaunchFailed', 'failure': err,
                                 'instance': free[0], 'attempts': (ev or {}).get('attempts', 0) + 1}
            if job:
                busy.add((free[0], False))
        # A monitor launches nothing; its jobs evaluate in-job, so nothing is pending on the controller.
        s['evals_pending'] = [] if monitor else sorted(
            label for label, _ in targets if (s['evals'].get(label) or {}).get('status') != 'Completed'
            and (s['evals'].get(label) or {}).get('attempts', 0) < 2)
        last = next((m for m in reversed(metrics) if 'train_loss' in m), None)
        rate = None
        if len(metrics) > 20:
            a, b = metrics[-21], metrics[-1]
            if b['elapsed_seconds'] > a['elapsed_seconds']:
                rate = (b['step'] - a['step']) / (b['elapsed_seconds'] - a['elapsed_seconds'])
        s['progress'] = {'global_step': training.get('completed_steps') or (last or {}).get('step'), 'target': common['steps'],
                         'last_eval': last, 'best_step': training.get('best_step'),
                         'best_validation_loss': training.get('best_validation_loss'),
                         'steps_per_second': round(rate, 3) if rate else None,
                         'eta_hours': round((common['steps'] - last['step']) / rate / 3600, 1) if rate and last else None}
    state['updated_at'] = now
    return len(state['arms']) == len(campaign['arms']) and all(
        (a.get('done') or a.get('halted')) and not a.get('evals_pending')
        and all(e.get('status') != 'InProgress' for e in a['evals'].values()) for a in state['arms'].values())


def identity_guard(campaign, state):
    """Record which kind of identity the controller runs as. Root (e.g. a user's `aws login` session) is accepted only
    for monitoring; launching jobs requires a dedicated least-privilege identity, so a non-monitor campaign stops."""
    arn = aws('sts', 'get-caller-identity', '--query', 'Arn', '--output', 'text').stdout.strip()
    root = arn.endswith(':root')
    state['identity'] = {'kind': 'root' if root else 'iam',
                         'warning': '루트 임시 로그인 세션: 감시 전용으로만 사용, job 발행 금지' if root else None}
    if root and not campaign['common'].get('monitor_only'):
        raise SystemExit('Refusing to drive a launching campaign with root credentials; use a least-privilege role.')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--state', type=Path, help='default: <campaign dir>/state.json')
    p.add_argument('--interval', type=int, default=300)
    p.add_argument('--once', action='store_true')
    a = p.parse_args()
    campaign = json.loads(a.campaign.read_text())
    path = a.state or a.campaign.with_name('state.json')
    state = json.loads(path.read_text()) if path.exists() else {'campaign': campaign['id'], 'arms': {}}
    while True:
        try:
            identity_guard(campaign, state)
            finished = step(campaign, state)
            state.pop('auth', None)
        except RuntimeError as exc:
            if not any(marker in str(exc) for marker in AUTH_MARKERS):
                raise
            # Expired/missing login: nothing can be observed or launched. Record it and wait for re-authentication
            # instead of exiting (which would make launchd restart the driver in a loop). No job is created here.
            state['auth'] = {'status': 'reauth_required',
                             'since': state.get('auth', {}).get('since') or datetime.datetime.now().isoformat(timespec='seconds'),
                             'message': '사용자 재인증 필요 (aws login). 제어기는 대기 중이며 job을 만들지 않는다.'}
            path.write_text(json.dumps(state, ensure_ascii=False, indent=1))
            print(datetime.datetime.now().isoformat(timespec='seconds'), 'reauth_required', flush=True)
            if a.once:
                return
            time.sleep(a.interval)
            continue
        path.write_text(json.dumps(state, ensure_ascii=False, indent=1))
        aws('s3', 'cp', '--only-show-errors', str(path), campaign['status_s3'], check=False)
        print(state['updated_at'], json.dumps({n: (s['progress'], s['segments'][-1]['status'] if s['segments'] else None)
                                               for n, s in state['arms'].items()}, ensure_ascii=False), flush=True)
        if finished or a.once:
            return
        time.sleep(a.interval)


if __name__ == '__main__':
    main()
