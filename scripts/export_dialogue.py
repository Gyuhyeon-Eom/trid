"""Export local dialogue weights, removing the unused training-only intent head."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil

import torch
from safetensors.torch import save_file
from trid.model import Trid
from trid.runtime import load_run
from trid.text import prompt_ids


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--version',default='0.2-experimental')
    args=parser.parse_args()
    source,target=Path(args.run),Path(args.output)
    if target.exists():
        raise SystemExit('Export to a new directory; existing models are preserved.')
    torch.set_num_threads(4)
    original,tok,_=load_run(source,'cpu')
    exported=Trid(replace(original.cfg,num_intents=0)).eval()
    state={k:v for k,v in original.state_dict().items() if not k.startswith('intent_head.')}
    exported.load_state_dict(state,strict=True)
    # Head removal must be exactly neutral to the language-model logits.
    with torch.inference_mode():
        for text in ['안녕','네 이름은 뭐야?','오늘은 쉬고 싶어.']:
            x=torch.tensor([prompt_ids(tok,text)])
            torch.testing.assert_close(original(x)[0],exported(x)[0],rtol=0,atol=0)
    target.mkdir(parents=True)
    save_file({k:v.contiguous() for k,v in state.items()},str(target/'model.safetensors'))
    (target/'config.json').write_text(json.dumps(asdict(exported.cfg),indent=2))
    shutil.copy2(source/'tokenizer.json',target/'tokenizer.json')
    metadata={'version':args.version,'stage':'sft','status':'exported',
        'parameters':exported.parameter_count(),'weight_bytes':(target/'model.safetensors').stat().st_size,
        'source_run':str(source.resolve()),'source_weights_sha256':hashlib.sha256((source/'model.safetensors').read_bytes()).hexdigest(),
        'weights_sha256':hashlib.sha256((target/'model.safetensors').read_bytes()).hexdigest(),
        'tokenizer_sha256':hashlib.sha256((target/'tokenizer.json').read_bytes()).hexdigest(),
        'export':'Removed unused auxiliary head; exact CPU logit equality checked on three prompts.',
        'ancestry':'Own randomly initialized Korean pretraining -> own dialogue stages; no external pretrained weights.',
        'source_training':json.loads((source/'training.json').read_text())}
    (target/'training.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    print(json.dumps({k:metadata[k] for k in ['parameters','weight_bytes','weights_sha256']},indent=2))


if __name__=='__main__':
    main()
