"""Train the generative backbone on dialogue and real document QA; never activates it."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def document_prompt(doc, question):
    return (f"다음은 {doc['date']}에 발표된 문서입니다. 문서 질문은 근거에 따라 답하고 [D1]을 인용하세요. "
            '인사나 사용 안내에는 자연스럽게 답하세요. 현재 시행 여부는 단정하지 마세요.\n'
            f"[D1] {doc['title']}\n{doc['text']}\n\n질문: {question}")


def prepare(output, init_run):
    from tokenizers import Tokenizer
    from trid.text import encode_example, prompt_ids
    if output.exists():
        raise ValueError('Use a new data directory')
    public = ROOT / 'data/public-document-sft-01'
    base = ROOT / 'data/base-dialogue-v1'
    rag = ROOT / 'data/korean-rag-curriculum-v2'
    docs = {d['id']: d for d in read_jsonl(ROOT / 'data/public-document-sft-source/documents.jsonl')}
    tok = Tokenizer.from_file(str(init_run / 'tokenizer.json'))
    rng = random.Random(271030)
    output_rows = {}
    dropped = Counter()
    for split in ('train', 'valid'):
        public_rows = read_jsonl(public / f'{split}.jsonl')
        source_ids = sorted({r['source_id'] for r in public_rows})
        dialogue = read_jsonl(base / f'{split}.jsonl')
        real = [r for r in read_jsonl(rag / f'{split}.jsonl') if r.get('group') == 'rag_real']
        components = {'dialogue': [], 'document_dialogue': [], 'public': [], 'real_qa': real}
        for row in dialogue:
            components['dialogue'].append(dict(row))
            if row['task'] in {'basic_greeting', 'basic_capability', 'basic_identity', 'basic_thanks', 'basic_clarify'}:
                doc = docs[rng.choice(source_ids)]
                components['document_dialogue'].append({**row, 'prompt': document_prompt(doc, row['prompt']), 'source_id': doc['id']})
        for row in public_rows:
            question = row['prompt'].rsplit('질문: ', 1)[1]
            if row['task'] == 'extractive_summary':
                # Original labels summarize two selected facts, not every fact in the full source.
                question = '문서에서 다음 두 내용을 간추려 주세요.\n' + '\n'.join(e['quote'] for e in row['evidence'])
            components['public'].append({**row, 'prompt': document_prompt(docs[row['source_id']], question)})
        kept = []
        mass = {'dialogue': .30, 'document_dialogue': .20, 'public': .30, 'real_qa': .20}
        for component, candidates in components.items():
            eligible = []
            for row in candidates:
                try:
                    encode_example(tok, row, 1024)
                    if len(prompt_ids(tok, row['prompt'], row.get('history'))) + 128 > 1024:
                        raise ValueError('generation reserve')
                except ValueError:
                    dropped[f'{split}:{component}'] += 1
                    continue
                eligible.append(row)
            if not eligible:
                raise ValueError(f'Empty component: {split}:{component}')
            tasks = Counter(r['task'] for r in eligible)
            for row in eligible:
                row = dict(row)
                row['weight'] = mass[component] / len(tasks) / tasks[row['task']]
                row['mixture_component'] = component
                kept.append(row)
        output_rows[split] = kept
    train_sources = {r['source_id'] for r in output_rows['train'] if 'source_id' in r}
    valid_sources = {r['source_id'] for r in output_rows['valid'] if 'source_id' in r}
    if train_sources & valid_sources:
        raise ValueError('Public source leakage')
    output.mkdir(parents=True)
    for split, data in output_rows.items():
        (output / f'{split}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in data))
    metadata = {'purpose': 'Generative model experiment, no automatic activation',
                'sources': ['base-dialogue-v1 authored dialogue', 'korean-rag-curriculum-v2 real_qa KLUE CC BY-SA', 'public-document-sft-01 KOGL-Type1'],
                'sampling_mass': mass, 'within_component': 'equal task probability',
                'public_context': 'full source text, never gold-only evidence selection',
                'dropped_for_context': dict(dropped),
                'rows': {s: dict(Counter(r['mixture_component'] for r in data)) for s, data in output_rows.items()},
                'limitations': 'Small public source set; developmental split, may overlap pretraining; user-reported public source is train-exposed regression, not heldout accuracy.'}
    (output / 'provenance.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


def prepare_qa_stage(output):
    """Correct the small-public-set sampling bias without changing existing artifacts."""
    from tokenizers import Tokenizer
    from trid.text import encode_example, prompt_ids
    from trid.prompts import RAG_PREFIX
    if output.exists():
        raise ValueError('Use a new data directory')
    previous = ROOT / 'data/public-dialogue-mix-01'
    tok = Tokenizer.from_file(str(ROOT / 'runs/trid-public-dialogue-01/tokenizer.json'))
    mass = {'dialogue': .10, 'document_dialogue': .05, 'public': .05, 'real_qa': .80}
    splits = {}
    dropped = Counter()
    for split in ('train', 'valid'):
        kept = []
        for original in read_jsonl(previous / f'{split}.jsonl'):
            row = dict(original)
            if row['mixture_component'] == 'public' and row['task'] in {'extractive_summary', 'followup', 'temporal_limit'}:
                continue
            if row['mixture_component'] == 'real_qa':
                if not row['prompt'].startswith(RAG_PREFIX):
                    raise ValueError('Unexpected real QA prompt')
                row['prompt'] = DOCUMENT_INSTRUCTION + row['prompt'][len(RAG_PREFIX):]
            try:
                encode_example(tok, row, 1024)
                if len(prompt_ids(tok, row['prompt'], row.get('history'))) + 128 > 1024:
                    raise ValueError('generation reserve')
            except ValueError:
                dropped[split] += 1
                continue
            kept.append(row)
        counts = Counter(r['mixture_component'] for r in kept)
        for row in kept:
            row['weight'] = mass[row['mixture_component']] / counts[row['mixture_component']]
        splits[split] = kept
    output.mkdir(parents=True)
    for split, data in splits.items():
        (output / f'{split}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in data))
    metadata = {'purpose': 'Second generative experiment: broader actual QA, common instruction format',
                'parent_data': str(previous), 'sampling_mass': mass, 'within_component': 'uniform rows',
                'removed_public_tasks': ['extractive_summary', 'followup', 'temporal_limit'],
                'dropped_for_context': dict(dropped),
                'parent_provenance': json.loads((previous / 'provenance.json').read_text()),
                'rows': {s: dict(Counter(r['mixture_component'] for r in data)) for s, data in splits.items()}}
    (output / 'provenance.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


ABSTAIN = '확인 불가'
DOCUMENT_INSTRUCTION = ('다음은 발표일 미상의 문서입니다. 문서 질문은 근거에 따라 답하고 출처 번호를 인용하세요. '
                        '인사나 사용 안내에는 자연스럽게 답하세요. 현재 시행 여부는 단정하지 마세요.\n')
REAL_TASKS = ('rag_klue', 'rag_klue_hard', 'rag_klue_missing', 'rag_tydi')


def prepare_abstain_stage(output, parent, abstain_share):
    """Copy a mix, changing only the training sampling share of real-QA abstention answers.

    Real-QA total mass and every other row weight stay identical; validation is copied unchanged so
    checkpoint selection is comparable with the parent experiment.
    """
    if output.exists():
        raise ValueError('Use a new data directory')
    if not 0 < abstain_share < 1:
        raise ValueError('abstain share must be between 0 and 1')
    train = read_jsonl(parent / 'train.jsonl')
    real = [r for r in train if r['mixture_component'] == 'real_qa']
    abstain = sum(r['weight'] for r in real if ABSTAIN in r['answer'])
    answer = sum(r['weight'] for r in real if ABSTAIN not in r['answer'])
    if not abstain or not answer:
        raise ValueError('Parent real QA must contain both abstentions and answers')
    total = abstain + answer
    scale = {True: abstain_share * total / abstain, False: (1 - abstain_share) * total / answer}
    for row in real:
        row['weight'] *= scale[ABSTAIN in row['answer']]
    output.mkdir(parents=True)
    (output / 'train.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in train))
    (output / 'valid.jsonl').write_bytes((parent / 'valid.jsonl').read_bytes())
    metadata = {'purpose': 'Single-variable experiment: real-QA abstention sampling share only',
                'parent_data': str(parent), 'real_qa_abstain_share': {'before': round(abstain / total, 4), 'after': abstain_share},
                'real_qa_total_mass_unchanged': round(total, 6), 'validation': 'byte-identical copy of parent',
                'parent_provenance': json.loads((parent / 'provenance.json').read_text())}
    (output / 'provenance.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in metadata.items() if k != 'parent_provenance'}, ensure_ascii=False), flush=True)


def prepare_real_stage(output, parent, source):
    """Replace the parent's real-QA rows with every real-passage row of `source` (KLUE train, TyDi train).

    Other components, their weights, the real-QA total mass and the validation file are unchanged.
    Synthetic authored rows are not imported. Rows that cannot fit the 1024 context with a 128-token
    generation reserve are dropped.
    """
    from tokenizers import Tokenizer
    from trid.prompts import RAG_PREFIX, rag_passages
    from trid.text import encode_example, prompt_ids
    if output.exists():
        raise ValueError('Use a new data directory')
    tok = Tokenizer.from_file(str(ROOT / 'runs/trid-public-dialogue-01/tokenizer.json'))
    parent_train = read_jsonl(parent / 'train.jsonl')
    kept = [r for r in parent_train if r['mixture_component'] != 'real_qa']
    real_mass = sum(r['weight'] for r in parent_train if r['mixture_component'] == 'real_qa')
    real, dropped, seen = [], 0, set()
    for original in read_jsonl(source / 'train.jsonl'):
        if original.get('task') not in REAL_TASKS:
            continue
        if not original['prompt'].startswith(RAG_PREFIX) or 'train' not in original['source']:
            raise ValueError(f"Unexpected real QA row {original['id']}")
        if original['id'] in seen:
            continue
        seen.add(original['id'])
        row = {k: v for k, v in original.items() if k != 'weight'}
        row.update(prompt=DOCUMENT_INSTRUCTION + original['prompt'][len(RAG_PREFIX):], mixture_component='real_qa')
        try:
            encode_example(tok, row, 1024)
            if len(prompt_ids(tok, row['prompt'], row.get('history'))) + 128 > 1024:
                raise ValueError('generation reserve')
        except ValueError:
            dropped += 1
            continue
        real.append(row)
    for row in real:
        row['weight'] = real_mass / len(real)
    # RAG310 rows carry no source ids, so compare the evidence passages themselves.
    eval_passages = {text.strip() for r in read_jsonl(ROOT / 'data/korean-rag-eval-v1/test.jsonl')
                     for _, text in rag_passages(r['prompt']) if len(text.strip()) >= 40}
    overlap = sum(text.strip() in eval_passages for r in real
                  for _, text in rag_passages(RAG_PREFIX + r['prompt'][len(DOCUMENT_INSTRUCTION):]))
    if overlap:
        raise ValueError(f'{overlap} training passages also appear in RAG310')
    output.mkdir(parents=True)
    (output / 'train.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in kept + real))
    (output / 'valid.jsonl').write_bytes((parent / 'valid.jsonl').read_bytes())
    tasks = Counter(r['task'] for r in real)
    metadata = {'purpose': 'Real-passage expansion: all KLUE train / KLUE hard-distractor / TyDi train rows, no synthetic rows',
                'parent_data': str(parent), 'source': str(source), 'real_qa_rows': {'before': len(parent_train) - len(kept), 'after': len(real)},
                'real_qa_tasks': dict(tasks), 'real_qa_abstain_share': round(sum(ABSTAIN in r['answer'] for r in real) / len(real), 4),
                'dropped_for_context': dropped, 'real_qa_total_mass_unchanged': round(real_mass, 6),
                'validation': 'byte-identical copy of parent', 'rag310_source_overlap': 0,
                'licenses': {'KLUE MRC v1.1 train': 'CC BY-SA 4.0', 'TyDi QA v1.1 GoldP train': 'Apache-2.0'},
                'parent_provenance': json.loads((parent / 'provenance.json').read_text())}
    (output / 'provenance.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in metadata.items() if k != 'parent_provenance'}, ensure_ascii=False), flush=True)


def prepare_teacher_stage(output, parent, teacher_dirs, teacher_mass):
    """Add extra source-verbatim QA sets (teacher QA from scripts/build_teacher_qa.py, task pilots from
    scripts/build_task_pilot.py) as one new component. The mix provenance carries each set's own purpose.

    Every parent weight is scaled by (1 - teacher_mass); teacher rows share teacher_mass uniformly.
    Parent validation is copied unchanged so checkpoint selection stays comparable.
    """
    if output.exists():
        raise ValueError('Use a new data directory')
    if not 0 < teacher_mass < 1:
        raise ValueError('teacher mass must be between 0 and 1')
    train = read_jsonl(parent / 'train.jsonl')
    for row in train:
        row['weight'] *= 1 - teacher_mass
    teacher = [r for d in teacher_dirs for r in read_jsonl(d / 'train.jsonl')]
    if not teacher or len({r['id'] for r in teacher}) != len(teacher):
        raise ValueError('Teacher rows missing or duplicated')
    for row in teacher:
        row['weight'] = teacher_mass / len(teacher)
    output.mkdir(parents=True)
    (output / 'train.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in train + teacher))
    (output / 'valid.jsonl').write_bytes((parent / 'valid.jsonl').read_bytes())
    added = [json.loads((d / 'provenance.json').read_text()) for d in teacher_dirs]
    metadata = {'purpose': 'Parent mix plus added QA sets: ' + ' | '.join(p['purpose'] for p in added),
                'parent_data': str(parent), 'teacher_data': [str(d) for d in teacher_dirs], 'teacher_mass': teacher_mass,
                'teacher_rows': dict(Counter(r['task'] for r in teacher)), 'validation': 'byte-identical copy of parent',
                'teacher_provenance': added,
                'parent_provenance': json.loads((parent / 'provenance.json').read_text())}
    (output / 'provenance.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in metadata.items() if not k.endswith('provenance')}, ensure_ascii=False), flush=True)



def evaluate(run, data, output):
    import torch
    from trid.runtime import load_run, generate
    torch.set_num_threads(2)
    model, tok, device = load_run(run, 'cpu')
    public = [r for r in read_jsonl(data / 'valid.jsonl') if r['mixture_component'] == 'public' and r['task'] == 'grounded_qa']
    docs = {d['id']: d for d in read_jsonl(ROOT / 'data/public-document-sft-source/documents.jsonl')}
    seen = docs['korea-kr-press-156612711']
    prompts = [{'id': r['id'], 'kind': 'heldout_public_qa', 'prompt': r['prompt'], 'reference': r['answer']} for r in public]
    for question in ('안녕', '너한테 할 수 있는 질문이 뭐야', '제작비 지원 비율과 최대 한도가 어떻게 돼?'):
        prompts.append({'id': question, 'kind': 'train_exposed_user_regression', 'prompt': document_prompt(seen, question)})
    for question in ('반가워, 처음 왔어.', '어떤 일을 부탁하면 되니?', '문서를 읽다가 모르는 게 있으면 물어봐도 돼?'):
        prompts.append({'id': question, 'kind': 'dialogue_probe', 'prompt': question})
    results = []
    for item in prompts:
        result = generate(model, tok, item['prompt'], device, max_tokens=128)
        row = {**item, **result}
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    report = {'run': str(run), 'weights_sha256': hashlib.sha256((run / 'model.safetensors').read_bytes()).hexdigest(),
              'protocol': 'Greedy free generation, 128 tokens; no constraints, answer lookup, or hardcoded response. No automatic quality score.', 'results': results}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'prepare-qa', 'prepare-abstain', 'prepare-real', 'prepare-teacher', 'train', 'evaluate'])
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--init-run', type=Path, default=ROOT / 'runs/trid-korean-medium-cpt-gpu-complete')
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='mps')
    parser.add_argument('--parent-data', type=Path, default=ROOT / 'data/public-dialogue-mix-02')
    parser.add_argument('--abstain-share', type=float, default=.10)
    parser.add_argument('--real-source', type=Path, default=ROOT / 'data/korean-rag-curriculum-v4')
    parser.add_argument('--teacher-data', type=Path, action='append', default=[])
    parser.add_argument('--teacher-mass', type=float, default=.25)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.data, args.init_run)
    elif args.action == 'prepare-qa':
        prepare_qa_stage(args.data)
    elif args.action == 'prepare-abstain':
        prepare_abstain_stage(args.data, args.parent_data, args.abstain_share)
    elif args.action == 'prepare-real':
        prepare_real_stage(args.data, args.parent_data, args.real_source)
    elif args.action == 'prepare-teacher':
        prepare_teacher_stage(args.data, args.parent_data, args.teacher_data, args.teacher_mass)
    elif args.action == 'evaluate':
        if args.report is None:
            parser.error('--report required')
        evaluate(args.run, args.data, args.report)
    else:
        import torch
        from trid.training import train
        torch.set_num_threads(2)
        train(args.data, args.run, init_run=args.init_run, steps=args.steps, batch_size=8,
              micro_batch_size=2, lr=0.0001, device=args.device, seed=271030,
              eval_every=100, validation_limit=64, context_length=1024, loss_normalization='example')


if __name__ == '__main__':
    main()
