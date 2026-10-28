"""Bounded local pretraining -> mixed SFT -> diagnostics. Never activates a model."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'reports/medium-training-job.json'
STOP=ROOT/'reports/medium-training-stop.flag'
BASE=ROOT/'runs/trid-korean-medium-base-v1'
SFT_ROOT=ROOT/'runs/trid-korean-medium-rag-v1'
TARGET=ROOT/'runs/trid-medium-v0.4-candidate'
PLANNED_STEPS=48829  # 100,001,792 tokens at batch 2 and context 1024.

def read(path):return json.loads(path.read_text()) if path.exists() else {}

def write_state(state,**updates):
    state.update(updates,updated_at_unix=time.time())
    pending=STATE.with_suffix('.tmp');pending.write_text(json.dumps(state,ensure_ascii=False,indent=2));pending.replace(STATE)

def active_pid(state):
    pid=state.get('pid')
    if not isinstance(pid,int):return False
    result=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True)
    return result.returncode==0 and str(Path(__file__).name) in result.stdout

def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group()
    g.add_argument('--resume',action='store_true');g.add_argument('--status',action='store_true');g.add_argument('--stop',action='store_true')
    a=p.parse_args();os.chdir(ROOT);os.umask(0o077)
    previous=read(STATE)
    if a.status:
        metrics=read(BASE/'metrics.json')
        print(json.dumps({**previous,'process_running':active_pid(previous),'latest_metric':metrics[-1] if metrics else None},ensure_ascii=False,indent=2));return
    if a.stop:
        if not active_pid(previous):raise ValueError('No matching training process is running')
        STOP.touch(exist_ok=True);print('현재 학습 구간 종료 후 중단합니다. 사전학습은 최대 250단계, 대화 학습은 현재 실행 종료까지 걸릴 수 있습니다.');return
    if STATE.exists() and not a.resume:raise ValueError('Job exists; inspect --status and use --resume')
    if active_pid(previous) and previous['pid']!=os.getpid():raise ValueError('The recorded training job is still running')
    if STOP.exists():STOP.unlink()  # Explicit resume clears only this job's stop request.
    state={'pid':os.getpid(),'started_at_unix':time.time(),'status':'running','stage':'initializing',
           'planned_pretrain_steps':PLANNED_STEPS,'planned_pretrain_tokens':PLANNED_STEPS*2048,
           'base_run':str(BASE),'output_candidate':str(TARGET),'auto_activate':False,
           'log':str(ROOT/'reports/medium-training-pipeline.log')}
    if previous:state['previous_start_unix']=previous.get('started_at_unix')
    write_state(state)
    awake=None
    try:
        import torch
        from trid.pretraining import pretrain
        from trid.training import train
        torch.set_num_threads(4)
        if not torch.backends.mps.is_available():raise ValueError('This job requires the profiled Mac MPS device')
        awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        write_state(state,stage='pretraining')
        while read(BASE/'training.json').get('completed_steps',0)<PLANNED_STEPS:
            if STOP.exists():write_state(state,status='stopped');return
            resume=(BASE/'latest.json').exists()
            pretrain('data/korean-foundation-tokens-v2',str(BASE),
                     config_path=None if resume else 'configs/korean-medium-91m.json',
                     steps=PLANNED_STEPS,stop_after=250,batch_size=2,lr=.0003,device='mps',seed=271028,
                     resume=resume,eval_every=250,validation_batches=8)
            metadata=read(BASE/'training.json')
            write_state(state,completed_pretrain_steps=metadata['completed_steps'],pretrain_tokens_seen=metadata['tokens_seen'])
            if metadata['status']=='interrupted':write_state(state,status='interrupted');return
            torch.mps.empty_cache()
        if STOP.exists():write_state(state,status='stopped');return
        write_state(state,stage='mixed_sft')
        # Interrupted SFT has no optimizer checkpoint. Preserve it and start a new attempt on explicit resume.
        attempts=sorted(SFT_ROOT.glob('attempt-*')) if SFT_ROOT.exists() else []
        completed=[path for path in attempts if read(path/'training.json').get('status')=='complete']
        if completed:sft=completed[-1]
        else:
            SFT_ROOT.mkdir(parents=True,exist_ok=True);sft=SFT_ROOT/f'attempt-{len(attempts)+1:03d}'
            write_state(state,sft_run=str(sft))
            train('data/korean-rag-curriculum-v1',str(sft),steps=2000,batch_size=16,lr=.0001,
                  device='mps',seed=271029,init_run=str(BASE),eval_every=100,validation_limit=0,
                  intent_weight=0,context_length=1024,loss_normalization='example')
        if STOP.exists():write_state(state,status='stopped',sft_run=str(sft));return
        torch.mps.empty_cache();write_state(state,stage='evaluation',sft_run=str(sft))
        jobs=[
          ('scripts/evaluate_rag_curriculum.py',['--output','reports/medium-rag-diagnostic.json']),
          ('scripts/evaluate_base_dialogue.py',['--data','data/korean-rag-curriculum-v1','--suite','base-check','--output','reports/medium-base-diagnostic.json']),
          ('scripts/evaluate_base_dialogue.py',['--output','reports/medium-base-regression.json'])]
        for script,args in jobs:
            output=ROOT/args[args.index('--output')+1]
            if output.exists():
                import hashlib
                if read(output).get('model_sha256')!=hashlib.sha256((sft/'model.safetensors').read_bytes()).hexdigest():
                    raise ValueError('Existing diagnostic belongs to different weights; preserve it and choose a new report name')
            else:subprocess.run([sys.executable,script,'--run',str(sft),*args],check=True)
        write_state(state,stage='export')
        if not TARGET.exists():
            subprocess.run([sys.executable,'scripts/export_dialogue.py','--run',str(sft),'--output',str(TARGET),'--version','0.4-medium-experimental'],check=True)
        else:
            import hashlib
            if read(TARGET/'training.json').get('source_weights_sha256')!=hashlib.sha256((sft/'model.safetensors').read_bytes()).hexdigest():
                raise ValueError('Existing export belongs to different weights')
        rag=read(ROOT/'reports/medium-rag-diagnostic.json')
        write_state(state,status='complete',stage='review_required',rag_exact_pass=rag['total_pass'],rag_total=rag['total'],
                    note='Training and diagnostics finished. Semantic review pending. No model activation or website deployment performed. These suites were previously inspected development diagnostics.')
    except KeyboardInterrupt:
        write_state(state,status='interrupted');raise
    except Exception as error:
        write_state(state,status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:
        if awake is not None and awake.poll() is None:awake.terminate()

if __name__=='__main__':main()
