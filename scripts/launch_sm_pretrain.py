"""Launch scripts/sm_pretrain.sh as a SageMaker training job (AWS CLI; no boto3 needed locally).

Uploads the working-tree code needed for pretraining (trid/*.py, scripts/train_medium_cpt_gpu.py,
scripts/sm_pretrain.sh, configs/*.json) to an immutable content-addressed prefix, then creates a job whose
CheckpointConfig maps /opt/ml/checkpoints to --checkpoint-s3. A second job with the same --checkpoint-s3 and --run
restores the run directory and resumes from its latest checkpoint (exact step, optimizer, RNG and sampler state).

Reuses the existing execution role `trid-sagemaker-lora` (S3 scope: corpus/* and open-weight-lora/*), so code, data
and checkpoints stay under corpus/. The job record (name, owner, stop condition, S3 paths, code hash) is appended to
--log as one JSON line.

  python scripts/launch_sm_pretrain.py --run size-pilot-91m-mha --config korean-medium-91m.json --steps 306 \
      --instance-type ml.g6e.2xlarge --max-sessions 1 --stop-after 153 --wait
"""
import argparse
import datetime
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUCKET = 'trid-training-148761639313'
REGION = 'ap-northeast-2'
ROLE = 'arn:aws:iam::148761639313:role/trid-sagemaker-lora'
IMAGE = '763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-sagemaker'


def aws(*args, capture=True):
    out = subprocess.run(['aws', '--region', REGION, *args], capture_output=True, text=True)
    if out.returncode:
        raise SystemExit(f'aws {args[0]} {args[1]} failed: {out.stderr.strip()}')
    return out.stdout if capture else None


def code_files():
    files = sorted(p for p in (ROOT / 'trid').rglob('*.py') if '__pycache__' not in p.parts)
    files += [ROOT / f'scripts/{name}' for name in ('train_medium_cpt_gpu.py', 'sm_pretrain.sh', 'sm_sft_eval.sh',
                                                     'sm_span_eval.sh', 'sm_sft_campaign.sh', 'sm_eval_weights.sh',
                                                     'train_public_dialogue.py', 'evaluate_rag_eval.py',
                                                     'evaluate_question_sensitivity.py')]
    files += sorted((ROOT / 'configs').glob('*.json'))
    return files


def upload_code():
    files = code_files()
    digest = hashlib.sha256()
    for f in files:
        digest.update(str(f.relative_to(ROOT)).encode() + b'\0' + f.read_bytes())
    key = digest.hexdigest()[:16]
    prefix = f'corpus/code/pretrain/{key}/'
    for f in files:
        aws('s3', 'cp', '--only-show-errors', str(f), f's3://{BUCKET}/{prefix}{f.relative_to(ROOT)}', capture=False)
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--', *[str(f.relative_to(ROOT)) for f in files]],
                                cwd=ROOT, capture_output=True, text=True).stdout.strip())
    return prefix, key, head, dirty


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', required=True, help='run directory name under the checkpoint path')
    p.add_argument('--config', required=True, help='file name under configs/')
    p.add_argument('--steps', type=int, required=True)
    p.add_argument('--stop-after', type=int)
    p.add_argument('--max-sessions', type=int, default=0)
    p.add_argument('--instance-type', default='ml.g6e.2xlarge')
    p.add_argument('--data-s3', default=f's3://{BUCKET}/corpus/tokens/mix-v3-basetok-tokens/')
    p.add_argument('--checkpoint-s3', help=f'default s3://{BUCKET}/corpus/runs/pretrain/<run>/')
    p.add_argument('--env', nargs='*', default=[], metavar='KEY=VALUE', help='extra entrypoint variables (SEED, LR, ...)')
    p.add_argument('--max-runtime', type=int, default=7200, help='seconds; SageMaker stops the job after this')
    p.add_argument('--owner', default='A (01a18739)')
    p.add_argument('--purpose', default='')
    p.add_argument('--log', type=Path, default=ROOT / 'reports/coordination/sm-jobs-A.jsonl')
    p.add_argument('--wait', action='store_true')
    p.add_argument('--entry', default='sm_pretrain.sh',
                   choices=['sm_pretrain.sh', 'sm_sft_eval.sh', 'sm_span_eval.sh', 'sm_sft_campaign.sh', 'sm_eval_weights.sh'],
                   help='entrypoint under scripts/: pretraining, SFT + evaluation, span diagnostic, SFT campaign segment, '
                        'evaluation of saved weights')
    a = p.parse_args()
    checkpoint = a.checkpoint_s3 or f's3://{BUCKET}/corpus/runs/pretrain/{a.run}/'
    prefix, key, head, dirty = upload_code()
    stamp = datetime.datetime.now().strftime('%m%d%H%M%S')
    name = f'trid-pt-{a.run}'[:48].rstrip('-') + f'-{stamp}'
    env = {'RUN': a.run, 'CONFIG': a.config, 'STEPS': str(a.steps), 'MAX_SESSIONS': str(a.max_sessions)}
    if a.stop_after:
        env['STOP_AFTER'] = str(a.stop_after)
    env.update(item.split('=', 1) for item in a.env)
    channel = lambda n, uri: {'ChannelName': n, 'InputMode': 'File', 'DataSource': {'S3DataSource': {
        'S3DataType': 'S3Prefix', 'S3Uri': uri, 'S3DataDistributionType': 'FullyReplicated'}}}
    spec = {
        'TrainingJobName': name, 'RoleArn': ROLE,
        'AlgorithmSpecification': {'TrainingImage': IMAGE, 'TrainingInputMode': 'File',
                                   'ContainerEntrypoint': ['bash', f'/opt/ml/input/data/code/scripts/{a.entry}']},
        'InputDataConfig': [channel('code', f's3://{BUCKET}/{prefix}'), channel('data', a.data_s3)],
        'OutputDataConfig': {'S3OutputPath': f's3://{BUCKET}/corpus/sm-output/'},
        'ResourceConfig': {'InstanceType': a.instance_type, 'InstanceCount': 1, 'VolumeSizeInGB': 100},
        'StoppingCondition': {'MaxRuntimeInSeconds': a.max_runtime},
        'CheckpointConfig': {'S3Uri': checkpoint, 'LocalPath': '/opt/ml/checkpoints'},
        'Environment': env,
        'Tags': [{'Key': 'trid-owner', 'Value': 'A'}, {'Key': 'trid-run', 'Value': a.run}],
    }
    aws('sagemaker', 'create-training-job', '--cli-input-json', json.dumps(spec))
    record = {'job': name, 'owner': a.owner, 'purpose': a.purpose, 'entry': a.entry, 'run': a.run, 'config': a.config, 'steps': a.steps,
              'instance_type': a.instance_type, 'stop_condition': f'run complete, MAX_SESSIONS={a.max_sessions} or '
              f'MaxRuntimeInSeconds={a.max_runtime}; instance terminates when the job ends', 'checkpoint_s3': checkpoint,
              'data_s3': a.data_s3, 'code_s3': f's3://{BUCKET}/{prefix}', 'code_sha': key, 'git_head': head,
              'code_dirty_vs_head': dirty, 'environment': env, 'created_at': datetime.datetime.now().isoformat(timespec='seconds')}
    a.log.parent.mkdir(parents=True, exist_ok=True)
    with a.log.open('a') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(json.dumps(record, ensure_ascii=False))
    if a.wait:
        while True:
            d = json.loads(aws('sagemaker', 'describe-training-job', '--training-job-name', name))
            if d['TrainingJobStatus'] in ('Completed', 'Failed', 'Stopped'):
                print(json.dumps({'job': name, 'status': d['TrainingJobStatus'], 'secondary': d.get('SecondaryStatus'),
                                  'failure': d.get('FailureReason'), 'billable_seconds': d.get('BillableTimeInSeconds')}))
                return
            time.sleep(30)


if __name__ == '__main__':
    main()
