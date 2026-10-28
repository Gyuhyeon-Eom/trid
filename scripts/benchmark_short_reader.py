"""Same-weight reference/cached-RoPE parity and interleaved local latency benchmark."""
import argparse,json,statistics,time
from pathlib import Path
import torch
import trid.model as model_module
from trid.short_reader import load_pipeline,read
from trid.corpus import sha256,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--cases',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',choices=['cpu','mps','auto'],default='auto');a=p.parse_args()
    if Path(a.output).exists():raise ValueError('Use a new report path')
    cases=json.loads(Path(a.cases).read_text())['cases'];torch.set_num_threads(4)
    span,presence,tok,device=load_pipeline(a.run,a.device)
    original=model_module.rotary
    def reference(x,offset,theta,tables=None):return original(x,offset,theta)
    times={'reference':[],'cached':[]};different=[];max_delta=0.;outputs=[]
    paths=[Path(a.run)/c/'model.safetensors' for c in ['span','presence']]
    hashes=[sha256(path) for path in paths]
    try:
        for function in [reference,original]:
            model_module.rotary=function
            for _ in range(3):read(span,presence,tok,cases[0]['document'],cases[0]['question'],device)
        for i,case in enumerate(cases):
            pair={}
            order=['reference','cached'] if i%2==0 else ['cached','reference']
            for mode in order:
                model_module.rotary=reference if mode=='reference' else original
                started=time.perf_counter();r=read(span,presence,tok,case['document'],case['question'],device)
                times[mode].append((time.perf_counter()-started)*1000);pair[mode]=r
            keys=['answer','abstains','source_character_span','best_span_positions']
            if any(pair['reference'][k]!=pair['cached'][k] for k in keys):different.append(case['id'])
            max_delta=max(max_delta,abs(pair['reference']['raw_absent_softmax']-pair['cached']['raw_absent_softmax']))
            outputs.append({'id':case['id'],'answer':pair['cached']['answer'],'input_tokens':pair['cached']['input_tokens']})
    finally:model_module.rotary=original
    assert hashes==[sha256(path) for path in paths]
    summary={mode:{'median_ms':round(statistics.median(v),3),'p95_ms':round(sorted(v)[int(.95*(len(v)-1))],3),'requests':len(v)} for mode,v in times.items()}
    cache_bytes=sum(b.numel()*b.element_size() for m in [span,presence] for name,b in m.named_buffers() if 'rope_' in name)
    report={'run':a.run,'device':device,'dtype':'float32','threads':4,'weights_sha256':hashes,'weight_bytes':sum(path.stat().st_size for path in paths),'rotary_cache_bytes':cache_bytes,
            'protocol':'Three warm-ups per mode, then alternate mode order per question. Wall time includes tokenization, CPU preprocessing and synchronized inference. Same process and weights; no concurrent training. Local hardware only, not Raspberry Pi. Small workload, timing is descriptive, not a guarantee.',
            'cases_sha256':sha256(a.cases),'latency':summary,'different_predictions':different,'max_absence_probability_delta':max_delta,'outputs':outputs}
    write_json(a.output,report);print(json.dumps({k:report[k] for k in ['device','latency','different_predictions','max_absence_probability_delta','rotary_cache_bytes']},indent=2))


if __name__=='__main__':main()
