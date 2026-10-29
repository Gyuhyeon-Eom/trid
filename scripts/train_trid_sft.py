"""trid SFT: LoRA on a base model, merged into one trid weight file (HF safetensors).

Isolated environment (Kanana's config needs transformers 4.x; the project venv pins 5.x):

  uv run --no-project --python 3.12 --with 'torch==2.9.*' --with 'transformers>=4.51,<5' --with 'peft==0.17.*' \
      --with safetensors --with numpy --with tokenizers python scripts/train_trid_sft.py --model <hf dir> ...

Grounded-QA rows are drawn from data/korean-rag-curriculum-v4/train.jsonl by the curriculum's own `weight`
(sampling mass), without replacement, with a fixed seed; `--rag-only` restricts the draw to `rag_*` tasks so the
curriculum's fixed conversational templates are not learned. `--chat` (repeatable) adds every row of distilled sets
(scripts/distill_trid_chat.py, scripts/distill_trid_document_qa.py), with dialogue history, and no system prompt. RAG rows use the same system
prompt as the `grammar_instructed` evaluation condition (scripts/evaluate_external_chat.SYSTEM). Each sample is the
model's native chat template with the generation prompt (Qwen3: no-thinking prefix) followed by the answer and the
template's end-of-turn token; loss covers only those answer tokens. No evaluation-set selection: fixed step count,
validation loss on a fixed slice of the curriculum valid split is recorded only.
"""
import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from evaluate_external_chat import SYSTEM  # noqa: E402

END_OF_TURN = {'qwen3': '<|im_end|>', 'kanana': '<|eot_id|>'}


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def messages_of(row):
    system = [{'role': 'system', 'content': SYSTEM}] if row['task'].startswith('rag') else []
    history = []
    for turn in row.get('history') or []:
        history += [{'role': 'user', 'content': turn['user']}, {'role': 'assistant', 'content': turn['assistant']}]
    return system + history + [{'role': 'user', 'content': row['prompt']}]


def encode(tok, family, row, max_tokens):
    """(input_ids, labels) with labels -100 outside the answer; None when longer than max_tokens."""
    prompt = tok.apply_chat_template(messages_of(row), tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
    if family == 'qwen3' and not prompt.endswith('<think>\n\n</think>\n\n'):
        raise ValueError('Qwen3 no-thinking generation prefix missing')
    prompt_ids = tok(prompt, add_special_tokens=False)['input_ids']
    answer_ids = tok(row['answer'] + END_OF_TURN[family], add_special_tokens=False)['input_ids']
    ids = prompt_ids + answer_ids
    if len(ids) > max_tokens:
        return None
    return ids, [-100] * len(prompt_ids) + answer_ids


def weighted_sample(rows, n, rng):
    """Efraimidis-Spirakis weighted sampling without replacement."""
    keyed = sorted(rows, key=lambda r: rng.random() ** (1.0 / float(r.get('weight', 1.0))), reverse=True)
    return keyed[:n]


def batches(examples, size, pad_id):
    import torch
    for i in range(0, len(examples), size):
        chunk = examples[i:i + size]
        width = max(len(ids) for ids, _ in chunk)
        ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        labels = torch.full((len(chunk), width), -100, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for j, (x, y) in enumerate(chunk):
            ids[j, :len(x)] = torch.tensor(x)
            labels[j, :len(y)] = torch.tensor(y)
            mask[j, :len(x)] = 1
        yield ids, labels, mask


def loss_of(model, ids, labels, mask, device):
    """Mean answer-token cross-entropy; the vocabulary projection runs only at answer positions, so memory does
    not scale with prompt length x vocabulary (128k for Kanana)."""
    import torch.nn.functional as F
    causal_lm = model.get_base_model()
    hidden = causal_lm.model(input_ids=ids.to(device), attention_mask=mask.to(device)).last_hidden_state
    target = labels[:, 1:].to(device)
    keep = target != -100
    logits = causal_lm.lm_head(hidden[:, :-1][keep]).float()
    return F.cross_entropy(logits, target[keep])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', type=Path, required=True, help='local HF model directory')
    p.add_argument('--model-id', required=True)
    p.add_argument('--revision', required=True)
    p.add_argument('--family', choices=sorted(END_OF_TURN), required=True)
    p.add_argument('--data', type=Path, default=ROOT / 'data/korean-rag-curriculum-v4')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--examples', type=int, default=2000)
    p.add_argument('--rag-only', action='store_true', help='draw only rag_* curriculum tasks')
    p.add_argument('--chat', type=Path, action='append', default=[],
                   help='distilled directory with train.jsonl (all rows used); repeatable')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--micro-batch-size', type=int, default=2)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--rank', type=int, default=16)
    p.add_argument('--max-tokens', type=int, default=1536)
    p.add_argument('--valid-examples', type=int, default=64)
    p.add_argument('--seed', type=int, default=291029)
    p.add_argument('--device', default='mps')
    args = p.parse_args()
    if args.output.exists():
        p.error('Use a new output directory')
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    train_rows = read_jsonl(args.data / 'train.jsonl')
    if args.rag_only:
        train_rows = [r for r in train_rows if r['task'].startswith('rag')]
    chosen = weighted_sample(train_rows, args.examples, rng)
    chat_rows = [row for path in args.chat for row in read_jsonl(path / 'train.jsonl')]
    chosen += chat_rows
    train, dropped = [], 0
    for row in chosen:
        encoded = encode(tok, args.family, row, args.max_tokens)
        if encoded is None:
            dropped += 1
        else:
            train.append(encoded)
    rng.shuffle(train)
    valid_rows = read_jsonl(args.data / 'valid.jsonl')[:args.valid_examples]
    valid = [e for e in (encode(tok, args.family, r, args.max_tokens) for r in valid_rows) if e is not None]

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(args.device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lora = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type='CAUSAL_LM',
                      target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
    model = get_peft_model(model, lora)
    params = [q for q in model.parameters() if q.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    accumulate = args.batch_size // args.micro_batch_size
    steps = math.ceil(len(train) / args.batch_size)
    warmup = max(1, steps // 20)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / warmup) *
                                                 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))

    def validate():
        model.eval()
        total = 0.0
        with torch.no_grad():
            for ids, labels, mask in batches(valid, args.micro_batch_size, pad_id):
                total += loss_of(model, ids, labels, mask, args.device).item()
        model.train()
        return total / max(1, math.ceil(len(valid) / args.micro_batch_size))

    metrics = [{'step': 0, 'valid_loss': validate()}]
    print(json.dumps(metrics[-1]), flush=True)
    started = time.time()
    model.train()
    micro = list(batches(train, args.micro_batch_size, pad_id))
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        group = micro[step * accumulate:(step + 1) * accumulate]
        running = 0.0
        for ids, labels, mask in group:
            loss = loss_of(model, ids, labels, mask, args.device) / len(group)
            loss.backward()
            running += loss.item()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        schedule.step()
        if (step + 1) % 25 == 0 or step + 1 == steps:
            entry = {'step': step + 1, 'train_loss': running, 'elapsed_seconds': round(time.time() - started, 1)}
            if (step + 1) % 100 == 0 or step + 1 == steps:
                entry['valid_loss'] = validate()
            metrics.append(entry)
            print(json.dumps(entry), flush=True)

    merged = model.merge_and_unload()
    args.output.mkdir(parents=True)
    merged.save_pretrained(args.output, safe_serialization=True)
    tok.save_pretrained(args.output)
    (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    (args.output / 'training.json').write_text(json.dumps({
        'protocol': __doc__, 'stage': 'trid_sft', 'base_model_id': args.model_id, 'base_revision': args.revision,
        'family': args.family, 'data': str(args.data.relative_to(ROOT)), 'train_sha256': digest(args.data / 'train.jsonl'),
        'valid_sha256': digest(args.data / 'valid.jsonl'), 'rag_only': args.rag_only,
        'chat': [str(path) for path in args.chat],
        'chat_sha256': {str(path): digest(path / 'train.jsonl') for path in args.chat}, 'chat_rows': len(chat_rows),
        'examples_drawn': len(chosen),
        'examples_trained': len(train), 'dropped_over_max_tokens': dropped,
        'task_counts': {t: sum(r['task'] == t for r in chosen) for t in sorted({r['task'] for r in chosen})},
        'drawn_ids_sha256': hashlib.sha256('\n'.join(r.get('id') or r['prompt'] for r in chosen).encode()).hexdigest(),
        'steps': steps, 'batch_size': args.batch_size, 'micro_batch_size': args.micro_batch_size, 'lr': args.lr,
        'lora': {'rank': args.rank, 'alpha': 2 * args.rank, 'dropout': 0.05, 'targets': lora.target_modules and sorted(lora.target_modules)},
        'max_tokens': args.max_tokens, 'seed': args.seed, 'device': args.device, 'system_prompt': SYSTEM,
        'elapsed_seconds': round(time.time() - started, 1), 'status': 'complete'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
