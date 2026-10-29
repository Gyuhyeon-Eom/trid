"""Full-parameter continued pretraining of trid's base weights on a token-budgeted public-sector sample.

  torchrun --nproc_per_node <gpus> scripts/train_trid_cpt.py --model <hf dir> --data data/trid-cpt-kanana-100m \
      --output runs/trid-cpt-01

Documents are tokenized as <bos> text <end_of_text>, concatenated in the sample's (seeded) order and cut into
fixed-length sequences; sequences are dealt round-robin to ranks. FP32 master weights, bf16 autocast, AdamW,
linear warmup then cosine decay to 10% of peak, gradient checkpointing, DDP. Validation loss on the sample's
valid split is measured before training and at the end. One pass over the sample; no early stopping.
"""
import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def packed(tok, path, seq_len, bos, eos):
    import torch
    ids = []
    for line in Path(path).read_text().splitlines():
        ids.append(bos)
        ids.extend(tok(json.loads(line)['text'], add_special_tokens=False)['input_ids'])
        ids.append(eos)
    usable = len(ids) // seq_len * seq_len
    return torch.tensor(ids[:usable], dtype=torch.long).view(-1, seq_len), len(ids)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--model-id', required=True)
    p.add_argument('--revision', required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seq-len', type=int, default=2048)
    p.add_argument('--micro-batch', type=int, default=1)
    p.add_argument('--global-tokens', type=int, default=262_144, help='tokens per optimizer step')
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--warmup-fraction', type=float, default=0.05)
    p.add_argument('--valid-sequences', type=int, default=256)
    p.add_argument('--seed', type=int, default=291029)
    args = p.parse_args()
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    torch.manual_seed(args.seed)
    if rank == 0 and args.output.exists():
        raise SystemExit('Use a new output directory')

    tok = AutoTokenizer.from_pretrained(args.model)
    bos = tok.bos_token_id
    eos = tok.convert_tokens_to_ids('<|end_of_text|>') if '<|end_of_text|>' in tok.get_vocab() else tok.eos_token_id
    train, train_tokens = packed(tok, args.data / 'train.jsonl', args.seq_len, bos, eos)
    valid, _ = packed(tok, args.data / 'valid.jsonl', args.seq_len, bos, eos)
    valid = valid[:args.valid_sequences]
    per_step = args.global_tokens // (args.seq_len * args.micro_batch * world)
    accumulate = max(1, per_step)
    sequences_per_step = accumulate * args.micro_batch * world
    steps = len(train) // sequences_per_step
    mine = train[rank::world]

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.config.use_cache = False
    ddp = DistributedDataParallel(model, device_ids=[local], gradient_as_bucket_view=True)
    decay = [q for n, q in model.named_parameters() if q.ndim >= 2]
    no_decay = [q for n, q in model.named_parameters() if q.ndim < 2]
    optimizer = torch.optim.AdamW([{'params': decay, 'weight_decay': 0.1}, {'params': no_decay, 'weight_decay': 0.0}],
                                  lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=True)
    warmup = max(1, int(steps * args.warmup_fraction))

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)

    def validate():
        model.eval()
        total, count = torch.zeros((), device=device), torch.zeros((), device=device)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for i in range(rank * args.micro_batch, len(valid), world * args.micro_batch):
                batch = valid[i:i + args.micro_batch].to(device)
                total += model(input_ids=batch, labels=batch).loss.float() * len(batch)
                count += len(batch)
        dist.all_reduce(total)
        dist.all_reduce(count)
        model.train()
        return (total / count).item()

    metrics = [{'step': 0, 'valid_loss': validate()}]
    if rank == 0:
        print(json.dumps({'steps': steps, 'accumulate': accumulate, 'sequences': len(train),
                          'train_tokens_packed': len(train) * args.seq_len, 'train_tokens_raw': train_tokens,
                          **metrics[0]}), flush=True)
    model.train()
    started = time.time()
    per_rank_step = accumulate * args.micro_batch
    for step in range(steps):
        running = torch.zeros((), device=device)
        for a in range(accumulate):
            start = step * per_rank_step + a * args.micro_batch
            batch = mine[start:start + args.micro_batch].to(device)
            sync = a == accumulate - 1
            ctx = ddp.no_sync() if not sync else torch.enable_grad()
            with ctx, torch.autocast('cuda', dtype=torch.bfloat16):
                loss = ddp(input_ids=batch, labels=batch).loss / accumulate
                loss.backward()
            running += loss.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        optimizer.zero_grad(set_to_none=False)  # keep DDP gradient_as_bucket_view storage
        if (step + 1) % 10 == 0 or step + 1 == steps:
            dist.all_reduce(running)
            entry = {'step': step + 1, 'train_loss': running.item() / world, 'lr': schedule.get_last_lr()[0],
                     'elapsed_seconds': round(time.time() - started, 1),
                     'tokens_per_second': round((step + 1) * sequences_per_step * args.seq_len / (time.time() - started))}
            metrics.append(entry)
            if rank == 0:
                print(json.dumps(entry), flush=True)
    metrics.append({'step': steps, 'valid_loss': validate()})
    if rank == 0:
        print(json.dumps(metrics[-1]), flush=True)
        args.output.mkdir(parents=True)
        model.to(torch.bfloat16).save_pretrained(args.output, safe_serialization=True)
        tok.save_pretrained(args.output)
        (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
        (args.output / 'training.json').write_text(json.dumps({
            'protocol': __doc__, 'stage': 'trid_continued_pretraining', 'base_model_id': args.model_id,
            'base_revision': args.revision, 'data': str(args.data), 'train_sha256': digest(args.data / 'train.jsonl'),
            'valid_sha256': digest(args.data / 'valid.jsonl'), 'train_tokens_raw': train_tokens,
            'train_tokens_used': steps * sequences_per_step * args.seq_len, 'steps': steps, 'world_size': world,
            'micro_batch': args.micro_batch, 'accumulate': accumulate, 'seq_len': args.seq_len, 'lr': args.lr,
            'warmup_steps': warmup, 'schedule': 'linear warmup, cosine to 10%', 'betas': [0.9, 0.95],
            'weight_decay': 0.1, 'precision': 'fp32 master, bf16 autocast', 'seed': args.seed,
            'elapsed_seconds': round(time.time() - started, 1), 'status': 'complete'}, ensure_ascii=False, indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
