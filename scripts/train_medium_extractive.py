"""Medium extractive head (trid.extractive) on data/korean-rag-curriculum-v2 from the medium base, then the
held-out evaluation (reports/closed-network-architecture-survey.md §8 experiment 1).

Waits for the SFT v2 job (scripts/train_medium_sft_v2.py, which itself waits for the pipeline) to be complete so the
MPS device is free, then trains 2,000 steps (batch 16, lr 1e-4, seed 271029) on mps, evaluates with
scripts/evaluate_extractive.py and writes reports/medium-extractive-v1.json. Never exports, activates or deploys.
"""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
UPSTREAM=ROOT/'reports/medium-sft-v2-job.json'
UPSTREAM_SCRIPT='train_medium_sft_v2.py'
STATE=ROOT/'reports/medium-extractive-job.json'
BASE=ROOT/'runs/trid-korean-medium-base-v1'
RUN_ROOT=ROOT/'runs/trid-korean-medium-extractive-v1'
DATA='data/korean-rag-curriculum-v2'
REPORT=ROOT/'reports/medium-extractive-v1.json'
POLL_SECONDS=300
TRAIN_ARGS=['--steps','2000','--batch-size','16','--lr','0.0001','--seed','271029','--device','mps','--context','1024','--eval-every','100']

def read(path):return json.loads(path.read_text()) if path.exists() else {}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_state(state,**updates):
    state.update(updates,updated_at_unix=time.time())
    pending=STATE.with_suffix('.tmp');pending.write_text(json.dumps(state,ensure_ascii=False,indent=2));pending.replace(STATE)

def pid_running(pid,name):
    if not isinstance(pid,int):return False
    result=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True)
    return result.returncode==0 and name in result.stdout

def wait_for_upstream(state):
    while True:
        job=read(UPSTREAM)
        if job.get('status')=='complete':return job
        if job.get('status') in ('failed','stopped','interrupted') or not pid_running(job.get('pid'),UPSTREAM_SCRIPT):
            raise RuntimeError(f'SFT v2 job is not running and not complete: {job.get("status")}; resolve it, then rerun')
        write_state(state,stage='waiting_for_sft_v2',upstream_stage=job.get('stage'))
        time.sleep(POLL_SECONDS)

def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group()
    g.add_argument('--status',action='store_true');g.add_argument('--no-wait',action='store_true',help='require the SFT v2 job to be complete already')
    a=p.parse_args();os.chdir(ROOT);os.umask(0o077)
    previous=read(STATE)
    if a.status:print(json.dumps({**previous,'process_running':pid_running(previous.get('pid'),Path(__file__).name)},ensure_ascii=False,indent=2));return
    if pid_running(previous.get('pid'),Path(__file__).name) and previous['pid']!=os.getpid():raise ValueError('An extractive job is already running')
    provenance=read(ROOT/DATA/'provenance.json')
    if not provenance or any(sha(ROOT/DATA/f'{s}.jsonl')!=provenance['hashes'][s] for s in ('train','valid')):raise ValueError('Curriculum v2 data missing or changed')
    state={'pid':os.getpid(),'started_at_unix':time.time(),'status':'running','stage':'initializing','data':DATA,
           'base_run':str(BASE),'run_root':str(RUN_ROOT),'auto_activate':False}
    write_state(state)
    awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        if a.no_wait and read(UPSTREAM).get('status')!='complete':raise RuntimeError('SFT v2 job is not complete')
        wait_for_upstream(state)
        import torch
        if not torch.backends.mps.is_available():raise ValueError('This job requires the Mac MPS device')
        attempts=sorted(RUN_ROOT.glob('attempt-*')) if RUN_ROOT.exists() else []
        done=[x for x in attempts if read(x/'training.json').get('status')=='complete']
        if done:run=done[-1]
        else:
            RUN_ROOT.mkdir(parents=True,exist_ok=True);run=RUN_ROOT/f'attempt-{len(attempts)+1:03d}'
            write_state(state,stage='training',run=str(run))
            subprocess.run([sys.executable,'scripts/train_extractive.py','--init-run',str(BASE),'--data',DATA,'--output',str(run),*TRAIN_ARGS],check=True,cwd=ROOT)
        write_state(state,stage='evaluation',run=str(run))
        if REPORT.exists():
            if read(REPORT).get('model_sha256')!=sha(run/'reader.safetensors'):raise ValueError(f'{REPORT} belongs to different weights; choose a new report name')
        else:subprocess.run([sys.executable,'scripts/evaluate_extractive.py','--run',str(run),'--output',str(REPORT),'--device','cpu'],check=True,cwd=ROOT)
        report=read(REPORT);keys=('strict','value','lenient_value','source','abstained','skipped_context','total')
        write_state(state,status='complete',stage='review_required',run=str(run),report=str(REPORT),
                    held_out={k:{m:report['summary'][k].get(m,0) for m in keys} for k in ('all','answerable','unanswerable') if k in report['summary']},
                    note='Compare with reports/medium-sft-v1-v2-comparison.json before any adoption decision. No export, activation or deployment.')
    except KeyboardInterrupt:
        write_state(state,status='interrupted');raise
    except Exception as error:
        write_state(state,status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:
        if awake.poll() is None:awake.terminate()

if __name__=='__main__':main()
