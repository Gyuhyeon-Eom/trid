"""Save unmodified model outputs for semantic review. No answer lookup or reranking."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import torch
from trid.runtime import load_run, generate


def repetition(text):
    compact = re.sub(r"\s+", " ", text).strip()
    # Flag consecutive repeated spans of at least two characters, repeated >= 3 times.
    return bool(re.search(r"(.{2,30}?)\1\1", compact))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--input', default='data/dialogue-eval-dev.jsonl')
    p.add_argument('--output', required=True)
    p.add_argument('--device', choices=['auto','cpu','mps'], default='auto')
    args = p.parse_args()
    out = Path(args.output)
    if out.exists():
        raise SystemExit('Use a new report path.')
    torch.set_num_threads(4)
    model,tok,device=load_run(args.run,args.device)
    results=[]
    for line in Path(args.input).read_text().splitlines():
        row=json.loads(line)
        result=generate(model,tok,row['prompt'],device,max_tokens=64,history=row.get('history'))
        result.update(row)
        result['repeated_span']=repetition(result['answer'])
        results.append(result)
        print(json.dumps({k:result[k] for k in ['id','prompt','answer','repeated_span']},ensure_ascii=False),flush=True)
    report={'run':args.run,'weights_sha256':hashlib.sha256((Path(args.run)/'model.safetensors').read_bytes()).hexdigest(),
            'prompts_sha256':hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
            'decoding':'greedy; no repetition suppression, retrieval, canned answers or external LLM',
            'repeat_count':sum(r['repeated_span'] for r in results),
            'eos_count':sum(r['stopped_on_eos'] for r in results),
            'semantic_review':'Required; repetition metrics do not measure relevance.',
            'results':results}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
