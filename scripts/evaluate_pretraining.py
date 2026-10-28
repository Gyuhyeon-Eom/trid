"""Full-split next-token loss, no training or random validation sampling."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from trid.corpus import sha256,write_json
from trid.runtime import load_run,sync


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True)
    p.add_argument('--data',default='data/korean-foundation-tokens-v1')
    p.add_argument('--split',choices=['valid','test'],default='valid')
    p.add_argument('--output',required=True)
    p.add_argument('--batch-size',type=int,default=16)
    a=p.parse_args();root=Path(a.data);out=Path(a.output)
    if out.exists():raise ValueError('Use a new report path.')
    if a.batch_size<1:raise ValueError('Batch size must be positive.')
    manifest=json.loads((root/'manifest.json').read_text())
    if sha256(root/f'{a.split}.bin')!=manifest['sha256'][a.split]:raise ValueError('Data changed.')
    if sha256(Path(a.run)/'tokenizer.json')!=manifest['tokenizer_sha256']:raise ValueError('Tokenizer mismatch.')
    weight_hash=sha256(Path(a.run)/'model.safetensors')
    torch.set_num_threads(4);model,tok,device=load_run(a.run)
    if sha256(Path(a.run)/'model.safetensors')!=weight_hash:raise ValueError('Model changed during loading; use a completed run.')
    data=np.memmap(root/f'{a.split}.bin',dtype='<u2',mode='r');length=model.cfg.context_length
    total,count=0.,0
    starts=list(range(0,len(data)-1,length))
    sync(device);start=time.perf_counter()
    with torch.inference_mode():
        for offset in range(0,len(starts),a.batch_size):
            batch=starts[offset:offset+a.batch_size]
            x=torch.zeros((len(batch),length),dtype=torch.long)
            y=torch.full_like(x,-100)
            for i,pos in enumerate(batch):
                values=torch.from_numpy(np.array(data[pos:pos+length+1],dtype=np.int64))
                n=len(values)-1;x[i,:n]=values[:-1];y[i,:n]=values[1:]
            x,y=x.to(device),y.to(device)
            logits,_=model(x)
            total+=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1),ignore_index=-100,reduction='sum').item()
            count+=int((y!=-100).sum().item())
    sync(device);loss=total/count
    assert count==len(data)-1
    write_json(out,{'run':a.run,'weights_sha256':weight_hash,'data_sha256':manifest['sha256'][a.split],'split':a.split,'target_tokens':count,'context_length':length,'loss':loss,'perplexity':math.exp(min(loss,30)),'seconds':time.perf_counter()-start,'evaluated_at':datetime.now(timezone.utc).isoformat(),'protocol':'All split next-token positions exactly once; context resets every context_length tokens; no sampling, training, or checkpoint selection. Lower loss is not evidence of dialogue correctness.'})
    print(out.read_text(),flush=True)

if __name__=='__main__':main()
