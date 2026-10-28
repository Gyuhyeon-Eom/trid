"""Combine real MRC and hand-authored coherent conversations without repetition."""
import argparse
from collections import Counter
import json
from pathlib import Path
from tokenizers import Tokenizer
from trid.corpus import sha256, write_json
from trid.text import encode_example
from trid.prompts import korean_grounded_prompt, KOREAN_GROUNDED_STYLE


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='data/korean-sft-v1')
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise ValueError('Use a new output folder.')
    source=Path('data/korean-conversations-v1'); mrc=Path('data/korean-mrc-v1')
    tok=Tokenizer.from_file('runs/trid-pretrain-pilot/tokenizer.json')
    frozen=[json.loads(x) for x in Path('data/korean-eval-v1/cases.jsonl').read_text().splitlines()]
    frozen_docs={x['document'] for x in frozen}
    parts={'train':[], 'valid':[]}; documents={s:set() for s in parts}; dropped=Counter()
    groups=Counter()
    for name in ['scenarios.json','corrections.json','basic.json']:
        for scenario in json.loads((source/name).read_text()):
            split=scenario['split'];doc=scenario.get('document');history=list(scenario.get('initial_history',[]))
            if doc:
                if doc in frozen_docs:raise ValueError('Frozen evaluation document in training curriculum.')
                documents[split].add(doc)
            for i,(question,answer) in enumerate(scenario['turns']):
                prompt=korean_grounded_prompt(doc,question) if doc else question
                row={'prompt':prompt,'answer':answer,'history':list(history),'task':'korean_basic_dialogue' if not doc else 'korean_correction' if scenario.get('initial_history') else 'korean_grounded_dialogue', 'source_scenario':scenario['id'],'turn':i+1,'source':'Codex-authored original Korean examples; fictional documents','fictional':bool(doc)}
                while True:
                    try:
                        tokens,_=encode_example(tok,row,512);break
                    except ValueError:
                        if not row['history']:raise
                        row['history'].pop(0);dropped[split]+=1
                row['training_tokens']=len(tokens)
                parts[split].append(row)
                history.append({'user':question,'assistant':answer})
                groups[f'{split}_{row["task"]}']+=1
    if documents['train']&documents['valid']:raise ValueError('Conversation document leakage.')
    for split in parts:
        parts[split] += [json.loads(x) for x in (mrc/f'{split}.jsonl').read_text().splitlines()]
    def group(row):
        return 'mrc' if row['task'].startswith('korean_mrc') else 'basic' if row['task']=='korean_basic_dialogue' else 'grounded'
    proportions={'mrc':.65,'grounded':.25,'basic':.10}
    group_counts=Counter(group(r) for r in parts['train'])
    for row in parts['train']:
        g=group(row);row['weight']=proportions[g]/group_counts[g]*len(parts['train'])
    keys={s:{json.dumps([r.get('history',[]),r['prompt']],ensure_ascii=False,sort_keys=True) for r in items} for s,items in parts.items()}
    if keys['train']&keys['valid']:raise ValueError('Train/valid prompt overlap.')
    if any(len(keys[s])!=len(items) for s,items in parts.items()):raise ValueError('Repeated examples.')
    out.mkdir(parents=True)
    for split,items in parts.items():
        with (out/f'{split}.jsonl').open('w') as f:
            for r in items:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    write_json(out/'provenance.json',{'source_files':{str(source/n):sha256(source/n) for n in ['scenarios.json','corrections.json','basic.json']},
        'mrc_provenance':json.loads((mrc/'provenance.json').read_text()),'rows':{s:len(v) for s,v in parts.items()},'conversation_groups':dict(groups),'training_group_counts':dict(group_counts),'sampling_mass':proportions,
        'history':'Coherent gold histories for training only. Explicit bad histories for correction. No independently random previous turns. Oldest turns dropped only to fit 512 tokens.',
        'history_turns_dropped':dict(dropped),'context_length':512,'grounded_prompt_style':KOREAN_GROUNDED_STYLE,'tokenizer_sha256':sha256('runs/trid-pretrain-pilot/tokenizer.json'),
        'evaluation':'Frozen internal suite excluded; MRC official dev not included. No public benchmark performance claim.',
        'limitations':'Small authored conversation curriculum. Does not establish broad Korean proficiency. MRC is limited to short full passages.',
        'file_sha256':{s:sha256(out/f'{s}.jsonl') for s in parts}})
    print(json.dumps({'rows':{s:len(v) for s,v in parts.items()},'groups':dict(groups),'sampling_mass':proportions,'history_turns_dropped':dict(dropped)},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
