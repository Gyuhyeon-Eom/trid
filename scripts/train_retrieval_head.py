"""Train the late-interaction retrieval head (trid.retrieval_head) from a base run.

Pairs (question -> gold passage) come from the RAG rows of the curriculum: rows whose answer is one `<value> [Dn]`
(rag_value/revision/noise/table and the KLUE-MRC rows rag_klue) use the cited passage; KLUE unanswerable rows
(rag_klue_missing) with exactly one passage use that passage (the KLUE question was written for it). Combine rows,
conflict/missing rows and rows with history are skipped and counted. Hard negatives per pair: the other passages of
the same prompt plus BM25 (bm25s + kiwi morphemes, scripts/benchmark_retrieval.py tokenisation) top-k non-gold
passages from the training passage corpus. Loss: in-batch negatives + one sampled hard negative per pair.
Validation: Recall@1/3 of the gold passage over all distinct passages of the valid split (gold + distractors).
Checkpoint selection by validation Recall@3 (ties -> lower loss).
"""
import argparse
import json
import math
import random
import shutil
import sys
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_retrieval import kiwi_tokenizer, bm25_retriever, KIWI_TAGS
from trid.corpus import sha256, write_json
from trid.extractive import parse_answer
from trid.prompts import rag_passages
from trid.retrieval_head import (RetrievalHead, contrastive_loss, encode_passages, encode_query, score,
                                 DIM, STAGE, WEIGHTS, QUERY_TOKENS, PASSAGE_TOKENS)
from trid.runtime import load_run, sync

GROUPS = ('rag', 'rag_real')


def load_pairs(data, split, limit, seed):
    """[{question, positive, prompt_negatives:[...], task}], skipped counts, per-task counts."""
    pairs, skipped = [], Counter()
    for line in (data / f'{split}.jsonl').read_text().splitlines():
        r = json.loads(line)
        if r.get('group') not in GROUPS or r.get('history'):
            skipped['other_group'] += 1
            continue
        if r['task'] == 'rag_combine':
            skipped['combine'] += 1
            continue
        passages = rag_passages(r['prompt'])
        question = r['prompt'].rsplit('\n질문: ', 1)[1].strip()
        parsed = parse_answer(r['answer'])
        if parsed is not None and parsed[0] == 0:
            gold = [t for s, t in passages if s == parsed[2]]
            if not gold:
                skipped['unknown_source'] += 1
                continue
            positive = gold[0]
        elif r['task'] == 'rag_klue_missing' and len(passages) == 1:
            positive = passages[0][1]
        else:
            skipped[r['task']] += 1
            continue
        pairs.append({'question': question, 'positive': positive, 'task': r['task'],
                      'prompt_negatives': [t for _, t in passages if t != positive]})
    if limit and len(pairs) > limit:
        pairs = random.Random(seed).sample(pairs, limit)
    if not pairs:
        raise ValueError(f'No usable pairs in {split}')
    return pairs, dict(skipped), dict(Counter(p['task'] for p in pairs))


def attach_bm25_negatives(pairs, corpus, k, tokenize):
    """pairs[i]['bm25_negatives'] = BM25 top-k passages (texts) that are not the gold passage."""
    if k < 1:
        for p in pairs:
            p['bm25_negatives'] = []
        return 0
    search = bm25_retriever(corpus, tokenize)
    total = 0
    for p in pairs:
        p['bm25_negatives'] = [corpus[i] for i in search(p['question'], k + 1) if corpus[i] != p['positive']][:k]
        total += len(p['bm25_negatives'])
    return total


def sample_batch(pairs, rng, size):
    """Batch of pairs with distinct positives (in-batch negatives must not be false negatives)."""
    batch, seen = [], set()
    for p in rng.sample(pairs, min(len(pairs), size * 4)):
        if p['positive'] not in seen:
            seen.add(p['positive']); batch.append(p)
        if len(batch) == size:
            break
    return batch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--init-run', required=True)
    p.add_argument('--data', default='data/korean-rag-curriculum-v2')
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=.0001)
    p.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    p.add_argument('--freeze-backbone', action='store_true', help='train only the projection')
    p.add_argument('--query-tokens', type=int, default=QUERY_TOKENS)
    p.add_argument('--passage-tokens', type=int, default=PASSAGE_TOKENS)
    p.add_argument('--temperature', type=float, default=.05)
    p.add_argument('--bm25-negatives', type=int, default=8, help='BM25 top-k non-gold hard negatives per pair (0 = off)')
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--train-limit', type=int, default=0, help='random subsample of usable training pairs (smoke tests)')
    p.add_argument('--validation-limit', type=int, default=0, help='random subsample of validation pairs')
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    if min(a.steps, a.batch_size, a.eval_every, a.query_tokens, a.passage_tokens) < 1 or not math.isfinite(a.lr) or a.lr <= 0 or a.temperature <= 0:
        p.error('Invalid training options')
    root, data = Path(a.output), Path(a.data)
    if root.exists():
        p.error('Use a new output directory')
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
    base, tok, device = load_run(a.init_run, a.device)
    if a.passage_tokens > base.cfg.context_length or a.query_tokens > base.cfg.context_length:
        p.error('--passage-tokens/--query-tokens exceed the base context length')
    cfg = replace(base.cfg)
    model = RetrievalHead(cfg, DIM)
    model.backbone.load_state_dict(base.state_dict()); del base
    model.to(device)
    if a.freeze_backbone:
        for q in model.backbone.parameters():
            q.requires_grad_(False)
    pairs, skipped, counts = {}, {}, {}
    for split, limit in (('train', a.train_limit), ('valid', a.validation_limit)):
        pairs[split], skipped[split], counts[split] = load_pairs(data, split, limit, a.seed)
    train_corpus = sorted({p['positive'] for p in pairs['train']})
    t = time.perf_counter()
    bm25_total = attach_bm25_negatives(pairs['train'], train_corpus, a.bm25_negatives, kiwi_tokenizer())
    bm25_seconds = time.perf_counter() - t
    for q in pairs['train']:
        q['hard'] = q['prompt_negatives'] + [n for n in q['bm25_negatives'] if n not in q['prompt_negatives']]
    valid_corpus = sorted({t for q in pairs['valid'] for t in [q['positive'], *q['prompt_negatives']]})
    valid_gold = [valid_corpus.index(q['positive']) for q in pairs['valid']]
    root.mkdir(parents=True)
    shutil.copy2(Path(a.init_run) / 'tokenizer.json', root / 'tokenizer.json')
    write_json(root / 'config.json', asdict(cfg))
    provenance_path = data / 'provenance.json'
    meta = {'stage': STAGE, 'status': 'training', 'initialization': a.init_run,
            'initial_weights_sha256': sha256(Path(a.init_run) / 'model.safetensors'),
            'tokenizer_sha256': sha256(root / 'tokenizer.json'), 'parameters': sum(q.numel() for q in model.parameters()),
            'trainable_parameters': sum(q.numel() for q in model.parameters() if q.requires_grad),
            'freeze_backbone': a.freeze_backbone, 'projection_dim': DIM, 'query_tokens': a.query_tokens,
            'passage_tokens': a.passage_tokens, 'markers': {'query': '질문: ', 'passage': '문단: '},
            'steps_requested': a.steps, 'batch_size': a.batch_size, 'learning_rate': a.lr, 'temperature': a.temperature,
            'seed': a.seed, 'eval_every': a.eval_every, 'device': device, 'threads': a.threads,
            'groups': list(GROUPS), 'pair_counts': counts, 'skipped_rows': skipped,
            'train_pairs': len(pairs['train']), 'valid_pairs': len(pairs['valid']),
            'train_passage_corpus': len(train_corpus), 'valid_passage_corpus': len(valid_corpus),
            'hard_negatives': {'bm25_top_k': a.bm25_negatives, 'bm25_tokenizer': 'kiwipiepy morphemes ' + ' '.join(sorted(KIWI_TAGS)),
                               'bm25_negatives_total': bm25_total, 'bm25_seconds': round(bm25_seconds, 2),
                               'prompt_negatives_total': sum(len(q['prompt_negatives']) for q in pairs['train']),
                               'pairs_without_hard_negative': sum(not q['hard'] for q in pairs['train'])},
            'train_limit': a.train_limit or None, 'validation_limit': a.validation_limit or None,
            'data': str(data), 'data_sha256': {s: sha256(data / f'{s}.jsonl') for s in pairs},
            'data_provenance': json.loads(provenance_path.read_text()) if provenance_path.exists() else None,
            'objective': 'softmax CE over MaxSim scores (query-length normalised, temperature); in-batch negatives + one hard negative per pair',
            'encoder': 'causal trid backbone, hidden_only; per-token linear projection + L2 norm; bos excluded',
            'checkpoint_selection': 'Maximum validation Recall@3 (ties: lower validation loss), including initial checkpoint',
            'weights_file': WEIGHTS, 'licences': {'bm25s': 'MIT (training-time hard negatives only)',
                                                    'kiwipiepy': 'Apache-2.0 (training-time hard negatives only)'}}
    write_json(root / 'training.json', meta)
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(.9, .95), weight_decay=.1)
    rng = random.Random(a.seed)

    @torch.inference_mode()
    def validate():
        model.eval()
        index = encode_passages(model, tok, valid_corpus, device, a.batch_size, a.passage_tokens)
        r1 = r3 = 0; loss_total = 0.
        for q, gold in zip(pairs['valid'], valid_gold):
            qv = encode_query(model, tok, q['question'], device, a.query_tokens)
            s = score(qv, index)
            ranked = np.argsort(-s, kind='stable')
            r1 += int(ranked[0] == gold); r3 += int(gold in ranked[:3])
            z = torch.from_numpy(s / len(qv) / a.temperature)
            loss_total += float(-torch.log_softmax(z, 0)[gold])
        model.train()
        n = len(pairs['valid'])
        return {'recall@1': r1 / n, 'recall@3': r3 / n, 'valid_loss': loss_total / n}

    def save(name):
        temp = root / 'checkpoint.tmp.safetensors'
        save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(temp))
        temp.replace(root / name)

    def better(s, best):
        return (s['recall@3'], -s['valid_loss']) > (best['recall@3'], -best['valid_loss'])

    sync(device); started = time.perf_counter()
    initial = validate(); best = initial; best_step = 0
    save(WEIGHTS)
    metrics = [{'step': 0, **initial, 'elapsed_seconds': round(time.perf_counter() - started, 2)}]
    write_json(root / 'metrics.json', metrics)
    print(f'{meta["parameters"]:,} parameters ({meta["trainable_parameters"]:,} trainable) | {device} | pairs {counts} | '
          f'skipped {skipped} | initial {initial}', flush=True)
    for step in range(1, a.steps + 1):
        warmup = min(50, max(1, a.steps // 10))
        ratio = step / warmup if step <= warmup else .1 + .9 * .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, a.steps - warmup)))
        for group in opt.param_groups:
            group['lr'] = a.lr * ratio
        opt.zero_grad(set_to_none=True)
        batch = sample_batch(pairs['train'], rng, a.batch_size)
        positives = [q['positive'] for q in batch]
        negatives = []
        for q in batch:
            pool = [n for n in q['hard'] if n not in positives and n not in negatives]
            if pool:
                negatives.append(rng.choice(pool))
        error, _ = contrastive_loss(model, tok, [q['question'] for q in batch], positives, negatives, device,
                                    a.query_tokens, a.passage_tokens, a.temperature, a.freeze_backbone)
        if not torch.isfinite(error):
            raise RuntimeError('Nonfinite loss')
        error.backward(); torch.nn.utils.clip_grad_norm_(params, 1.)
        opt.step()
        if step % a.eval_every == 0 or step == a.steps:
            result = validate(); sync(device)
            if better(result, best):
                best, best_step = result, step
                save(WEIGHTS)
            save('last-' + WEIGHTS)
            entry = {'step': step, 'train_loss': float(error.detach()), 'hard_negatives_in_batch': len(negatives), **result,
                     'elapsed_seconds': round(time.perf_counter() - started, 2)}
            metrics.append(entry); write_json(root / 'metrics.json', metrics)
            meta.update(completed_steps=step, best_step=best_step, best_validation=best)
            write_json(root / 'training.json', meta)
            print(json.dumps(entry), flush=True)
    meta.update(status='complete', initial_validation=initial, best_step=best_step, best_validation=best,
                training_seconds=round(time.perf_counter() - started, 2),
                weight_bytes=(root / WEIGHTS).stat().st_size, optimizer_resume_supported=False)
    write_json(root / 'training.json', meta)


if __name__ == '__main__':
    main()
