"""Medium SFT v2 (data/korean-rag-curriculum-v2) and held-out comparison against the pipeline's SFT v1.

Same base weights and SFT hyperparameters as scripts/train_medium_pipeline.py; only the SFT data differs.
Waits for the medium pipeline to finish (it owns the MPS device), then trains, evaluates and writes
reports/medium-sft-v1-v2-comparison.json. Never exports, activates or deploys a model.
"""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PIPELINE=ROOT/'reports/medium-training-job.json'
STATE=ROOT/'reports/medium-sft-v2-job.json'
BASE=ROOT/'runs/trid-korean-medium-base-v1'
SFT_ROOT=ROOT/'runs/trid-korean-medium-rag-v2'
DATA='data/korean-rag-curriculum-v2'
COMPARISON=ROOT/'reports/medium-sft-v1-v2-comparison.json'
POLL_SECONDS=300

def read(path):return json.loads(path.read_text()) if path.exists() else {}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_state(state,**updates):
    state.update(updates,updated_at_unix=time.time())
    pending=STATE.with_suffix('.tmp');pending.write_text(json.dumps(state,ensure_ascii=False,indent=2));pending.replace(STATE)

def pid_running(pid,name):
    if not isinstance(pid,int):return False
    result=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True)
    return result.returncode==0 and name in result.stdout

def wait_for_pipeline(state):
    while True:
        job=read(PIPELINE)
        if job.get('status')=='complete':return job
        if job.get('status') in ('failed','stopped','interrupted') or not pid_running(job.get('pid'),'train_medium_pipeline.py'):
            raise RuntimeError(f'Medium pipeline is not running and not complete: {job.get("status")}; resolve it, then rerun')
        write_state(state,stage='waiting_for_pipeline',pipeline_stage=job.get('stage'),pipeline_steps=job.get('completed_pretrain_steps'))
        time.sleep(POLL_SECONDS)

def evaluate(script,run,output,*args):
    output=ROOT/output
    if output.exists():
        if read(output).get('model_sha256')!=sha(Path(run)/'model.safetensors'):
            raise ValueError(f'{output} belongs to different weights; choose a new report name')
    else:subprocess.run([sys.executable,script,'--run',str(run),'--output',str(output),*args],check=True,cwd=ROOT)
    return read(output)

def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group()
    g.add_argument('--status',action='store_true');g.add_argument('--no-wait',action='store_true',help='require the pipeline to be complete already')
    a=p.parse_args();os.chdir(ROOT);os.umask(0o077)
    previous=read(STATE)
    if a.status:print(json.dumps({**previous,'process_running':pid_running(previous.get('pid'),Path(__file__).name)},ensure_ascii=False,indent=2));return
    if pid_running(previous.get('pid'),Path(__file__).name) and previous['pid']!=os.getpid():raise ValueError('An SFT v2 job is already running')
    provenance=read(ROOT/DATA/'provenance.json')
    if not provenance or any(sha(ROOT/DATA/f'{s}.jsonl')!=provenance['hashes'][s] for s in ('train','valid')):raise ValueError('SFT v2 data missing or changed')
    state={'pid':os.getpid(),'started_at_unix':time.time(),'status':'running','stage':'initializing','data':DATA,
           'base_run':str(BASE),'sft_root':str(SFT_ROOT),'auto_activate':False}
    write_state(state)
    awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        job=read(PIPELINE)
        if a.no_wait and job.get('status')!='complete':raise RuntimeError('Medium pipeline is not complete')
        job=wait_for_pipeline(state)
        sft_v1=Path(job['sft_run'])
        import torch
        from trid.training import train
        torch.set_num_threads(4)
        if not torch.backends.mps.is_available():raise ValueError('This job requires the Mac MPS device')
        attempts=sorted(SFT_ROOT.glob('attempt-*')) if SFT_ROOT.exists() else []
        done=[x for x in attempts if read(x/'training.json').get('status')=='complete']
        if done:sft_v2=done[-1]
        else:
            SFT_ROOT.mkdir(parents=True,exist_ok=True);sft_v2=SFT_ROOT/f'attempt-{len(attempts)+1:03d}'
            write_state(state,stage='sft_v2',sft_v2_run=str(sft_v2))
            # Identical to the pipeline's SFT v1 call except the data directory.
            train(DATA,str(sft_v2),steps=2000,batch_size=16,lr=.0001,device='mps',seed=271029,init_run=str(BASE),
                  eval_every=100,validation_limit=0,intent_weight=0,context_length=1024,loss_normalization='example')
            torch.mps.empty_cache()
        write_state(state,stage='evaluation',sft_v1_run=str(sft_v1),sft_v2_run=str(sft_v2))
        held={}  # '<sft>/<decoding>' -> report; evidence decoding copies values from the cited passage
        for name,run in (('sft_v1',sft_v1),('sft_v2',sft_v2)):
            for decoding in ('free','evidence'):
                suffix='' if decoding=='free' else '-evidence'
                held[f'{name}/{decoding}']=evaluate('scripts/evaluate_rag_eval.py',run,f'reports/medium-rag-eval-v1-{name.replace("_","-")}{suffix}.json','--decoding',decoding)
        old_diag=evaluate('scripts/evaluate_rag_curriculum.py',sft_v2,'reports/medium-rag-diagnostic-sft-v2.json')
        base_check=evaluate('scripts/evaluate_base_dialogue.py',sft_v2,'reports/medium-base-diagnostic-sft-v2.json','--data',DATA,'--suite','base-check')
        regression=evaluate('scripts/evaluate_base_dialogue.py',sft_v2,'reports/medium-base-regression-sft-v2.json')
        keys=('strict','value','lenient_value','source','abstained','skipped_context','total')
        comparison={'protocol':'Same base weights and SFT hyperparameters; only SFT data differs (v1 vs v2). Held-out set data/korean-rag-eval-v1 was not used for training or checkpoint selection.',
          'sft_v1':{'run':str(sft_v1),'model_sha256':held['sft_v1/free']['model_sha256']},
          'sft_v2':{'run':str(sft_v2),'model_sha256':held['sft_v2/free']['model_sha256']},
          'held_out':{k:{name:{m:held[name]['summary'][k].get(m,0) for m in keys} for name in held} for k in held['sft_v2/free']['summary']},
          'sft_v2_previous_diagnostic':{'total_pass':old_diag.get('total_pass'),'total':old_diag.get('total')},
          'sft_v1_previous_diagnostic':{k:read(ROOT/'reports/medium-rag-diagnostic.json').get(k) for k in ('total_pass','total')},
          'base_reports':{'sft_v2_base_check':'reports/medium-base-diagnostic-sft-v2.json','sft_v2_regression':'reports/medium-base-regression-sft-v2.json',
                          'sft_v1_base_check':'reports/medium-base-diagnostic.json','sft_v1_regression':'reports/medium-base-regression.json'},
          'note':'Base dialogue reports need semantic review before any adoption decision. No export, activation or deployment.'}
        COMPARISON.write_text(json.dumps(comparison,ensure_ascii=False,indent=2))
        write_state(state,status='complete',stage='review_required',comparison=str(COMPARISON),
                    held_out_strict={n:comparison['held_out']['all'][n]['strict'] for n in held},held_out_total=comparison['held_out']['all']['sft_v2/free']['total'])
    except KeyboardInterrupt:
        write_state(state,status='interrupted');raise
    except Exception as error:
        write_state(state,status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:
        if awake.poll() is None:awake.terminate()

if __name__=='__main__':main()
