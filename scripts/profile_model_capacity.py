"""Measure larger own architectures on this Mac; random tokens are NOT language training."""
import argparse,gc,json,platform,time
from dataclasses import asdict
from pathlib import Path
import torch
from trid.model import Config,Trid

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise ValueError('Use a new output file')
    if not torch.backends.mps.is_available():raise ValueError('MPS required for this local sizing measurement')
    torch.set_num_threads(4);torch.manual_seed(271028);results=[]
    for dim,layers,hidden in [(512,12,1536),(768,12,2048)]:
        for length in [512,1024]:
            cfg=Config(vocab_size=8192,dim=dim,layers=layers,heads=8 if dim==512 else 12,hidden_dim=hidden,context_length=length)
            model=Trid(cfg).to('mps');optimizer=torch.optim.AdamW(model.parameters(),lr=.0001)
            x=torch.randint(0,cfg.vocab_size,(2,length),device='mps');y=torch.roll(x,-1,1)
            elapsed=[];allocated=[];driver=[]
            for i in range(6):
                optimizer.zero_grad(set_to_none=True);torch.mps.synchronize();start=time.perf_counter()
                logits,_=model(x);loss=torch.nn.functional.cross_entropy(logits.reshape(-1,cfg.vocab_size),y.reshape(-1))
                if not torch.isfinite(loss):raise ValueError('Non-finite loss')
                loss.backward();allocated.append(torch.mps.current_allocated_memory());driver.append(torch.mps.driver_allocated_memory())
                optimizer.step();torch.mps.synchronize()
                if i>=2:elapsed.append(time.perf_counter()-start)
            row={'config':asdict(cfg),'parameters':model.parameter_count(),'batch_size':2,'measured_steps':4,
                 'seconds_per_step':sum(elapsed)/len(elapsed),'tokens_per_second':(2*length*len(elapsed))/sum(elapsed),
                 'max_observed_mps_allocated_bytes_after_backward':max(allocated),'max_observed_driver_bytes_after_backward':max(driver)}
            results.append(row);print(json.dumps(row),flush=True)
            del optimizer,model,x,y,logits,loss;gc.collect();torch.mps.empty_cache()
    report={'platform':platform.platform(),'torch':torch.__version__,'device':'mps','measured_at_unix':time.time(),
            'scope':'Random-token full precision AdamW sizing, batch=2, two warmup + four timed steps. No useful model weights saved; no semantic capability claimed. Allocator samples are not guaranteed peak memory; not a sustained thermal/load benchmark.', 'results':results}
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
