"""KoBEST 5과제 로그우도 선택형 평가 (CC BY-SA 4.0 데이터; 생성 없이 선택지 조건부 로그우도 argmax).

프롬프트 템플릿은 lm-evaluation-harness의 kobest 과제 정의를 그대로 따른다:
  https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/kobest/kobest_boolq.yaml
  https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/kobest/kobest_copa.yaml
  https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/kobest/kobest_hellaswag.yaml
  https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/kobest/kobest_sentineg.yaml
  https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/kobest/kobest_wic.yaml
  https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/kobest/utils.py
harness와 같이 문맥과 선택지 사이에 공백 1개(target_delimiter)를 두고, acc(로그우도 합)·acc_norm(선택지 UTF-8 바이트 수로 나눔)·macro F1을 기록한다.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from trid.corpus import sha256, write_json
from trid.runtime import load_run

REPO = 'skt/kobest_v1'
REVISION = 'a5ea15e3ac77ed694b79f6204eb31889a2ba989f'
LICENSE = 'CC BY-SA 4.0'
TASKS = ['boolq', 'copa', 'hellaswag', 'sentineg', 'wic']
HARNESS = 'https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/kobest/'


def to_doc(task, row):
    """(문맥, 선택지 목록, 정답 index) — harness doc_to_text/doc_to_choice/doc_to_target과 동일."""
    if task == 'boolq':
        return f"{row['paragraph']} 질문: {row['question']} 답변: ", ['아니오', '예'], int(row['label'])
    if task == 'copa':
        connector = {'원인': ' 왜냐하면', '결과': ' 그래서'}[row['question'].strip()]
        return f"{row['premise']}{connector}", [row['alternative_1'], row['alternative_2']], int(row['label'])
    if task == 'hellaswag':
        return f"문장: {row['context']}", [row[f'ending_{i}'] for i in range(1, 5)], int(row['label'])
    if task == 'sentineg':
        return f"문장: {row['sentence']} 긍부정:", ['부정', '긍정'], int(row['label'])
    if task == 'wic':
        return f"문장1: {row['context_1']} 문장2: {row['context_2']} 두 문장에서 {row['word']}가 같은 뜻으로 쓰였나?", ['아니오', '예'], int(row['label'])
    raise ValueError(task)


def fetch(root, split):
    """datasets로 고정 revision을 받아 과제별 JSONL로 저장하고 provenance.json을 기록한다. 이미 있으면 건너뛴다."""
    files = {t: root / t / f'{split}.jsonl' for t in TASKS}
    if all(f.exists() for f in files.values()) and (root / 'provenance.json').exists():
        return
    from datasets import load_dataset
    for task, path in files.items():
        if path.exists():
            continue
        ds = load_dataset(REPO, task, split=split, revision=REVISION)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            for row in ds:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        print('fetched', task, split, len(ds), flush=True)
    prov = json.loads((root / 'provenance.json').read_text()) if (root / 'provenance.json').exists() else {
        'repository': f'https://huggingface.co/datasets/{REPO}', 'revision': REVISION, 'license': LICENSE,
        'license_url': 'https://creativecommons.org/licenses/by-sa/4.0/',
        'source_urls': {t: f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{t}/{split}.jsonl' for t in TASKS},
        'downloaded_with': 'datasets.load_dataset(revision pinned); rows re-serialised as JSONL without modification',
        'sha256': {}, 'rows': {}}
    for task, path in files.items():
        prov['sha256'][f'{task}/{split}.jsonl'] = sha256(path)
        prov['rows'][f'{task}/{split}.jsonl'] = sum(1 for _ in path.open(encoding='utf-8'))
    prov['fetched_at'] = datetime.now(timezone.utc).isoformat()
    write_json(root / 'provenance.json', prov)


def encode_pair(tok, bos, context, continuation, limit):
    """harness _encode_pair: 전체를 함께 인코딩한 뒤 문맥 토큰 수만큼 잘라 선택지 토큰을 얻는다. 문맥은 왼쪽부터 자른다."""
    ctx = tok.encode(context).ids
    whole = tok.encode(context + continuation).ids
    cont = whole[len(ctx):]
    ids = ([bos] + ctx + cont)[-limit:]
    return ids, len(cont)


@torch.inference_mode()
def score(model, device, items, batch_size):
    """items: [(ids, n_cont)] → 각 항목의 선택지 로그우도 합."""
    out = [0.0] * len(items)
    order = sorted(range(len(items)), key=lambda i: -len(items[i][0]))
    for offset in range(0, len(order), batch_size):
        idx = order[offset:offset + batch_size]
        width = max(len(items[i][0]) for i in idx)
        x = torch.zeros((len(idx), width), dtype=torch.long)
        for r, i in enumerate(idx):
            x[r, :len(items[i][0])] = torch.tensor(items[i][0])
        logp = F.log_softmax(model(x.to(device))[0].float(), dim=-1)
        for r, i in enumerate(idx):
            ids, n = items[i]
            targets = torch.tensor(ids[len(ids) - n:], device=device)
            out[i] = logp[r, len(ids) - n - 1:len(ids) - 1].gather(1, targets[:, None]).sum().item()
    return out


def macro_f1(golds, preds, n):
    scores = []
    for c in range(n):
        tp = sum(g == c and p == c for g, p in zip(golds, preds))
        fp = sum(g != c and p == c for g, p in zip(golds, preds))
        fn = sum(g == c and p != c for g, p in zip(golds, preds))
        scores.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(scores) / n


def evaluate_task(model, tok, device, task, rows, batch_size):
    bos, limit = tok.token_to_id('<|bos|>'), model.cfg.context_length
    docs = [to_doc(task, r) for r in rows]
    items, spans = [], []
    for context, choices, _ in docs:
        spans.append((len(items), len(choices)))
        items += [encode_pair(tok, bos, context, ' ' + c, limit) for c in choices]
    start = time.perf_counter()
    ll = score(model, device, items, batch_size)
    golds, preds, preds_norm = [], [], []
    for (context, choices, gold), (s, n) in zip(docs, spans):
        lls = ll[s:s + n]
        golds.append(gold)
        preds.append(max(range(n), key=lambda i: lls[i]))
        preds_norm.append(max(range(n), key=lambda i: lls[i] / len((' ' + choices[i]).encode('utf-8'))))
    n_choice = len(docs[0][1])
    counts = [golds.count(c) for c in range(n_choice)]
    return {'docs': len(docs), 'choices': n_choice,
            'acc': sum(g == p for g, p in zip(golds, preds)) / len(docs),
            'acc_norm': sum(g == p for g, p in zip(golds, preds_norm)) / len(docs),
            'macro_f1': macro_f1(golds, preds, n_choice),
            'random_baseline_acc': 1 / n_choice, 'majority_baseline_acc': max(counts) / len(docs),
            'gold_counts': counts, 'pred_counts': [preds.count(c) for c in range(n_choice)],
            'truncated_contexts': sum(len(ids) >= limit for ids, _ in items),
            'seconds': round(time.perf_counter() - start, 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--source', default='data/kobest-source')
    p.add_argument('--split', choices=['validation', 'test'], default='test')
    p.add_argument('--tasks', nargs='+', default=TASKS, choices=TASKS)
    p.add_argument('--limit', type=int, default=0, help='과제당 앞에서부터 N문항만 (0=전체)')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    out, root = Path(a.output), Path(a.source)
    if out.exists():
        raise ValueError('Use a new report path.')
    fetch(root, a.split)
    prov = json.loads((root / 'provenance.json').read_text())
    torch.set_num_threads(a.threads)
    model, tok, device = load_run(a.run, a.device)
    results = {}
    for task in a.tasks:
        path = root / task / f'{a.split}.jsonl'
        if sha256(path) != prov['sha256'][f'{task}/{a.split}.jsonl']:
            raise ValueError(f'{path} changed since fetch.')
        rows = [json.loads(l) for l in path.open(encoding='utf-8')]
        if a.limit:
            rows = rows[:a.limit]
        results[task] = evaluate_task(model, tok, device, task, rows, a.batch_size)
        print(task, json.dumps(results[task], ensure_ascii=False), flush=True)
    mean = lambda k: sum(r[k] for r in results.values()) / len(results)
    state = Path(a.run) / 'state.json'
    write_json(out, {
        'run': a.run, 'weights_sha256': sha256(Path(a.run) / 'model.safetensors'),
        'step': json.loads(state.read_text()).get('step') if state.exists() else None,
        'dataset': {'repository': prov['repository'], 'revision': prov['revision'], 'license': prov['license'], 'split': a.split,
                    'sha256': {k: v for k, v in prov['sha256'].items() if k.endswith(f'{a.split}.jsonl')}},
        'prompt_templates': HARNESS, 'limit': a.limit, 'device': str(device), 'context_length': model.cfg.context_length,
        'tasks': results, 'mean_acc': mean('acc'), 'mean_acc_norm': mean('acc_norm'), 'mean_random_baseline_acc': mean('random_baseline_acc'),
        'evaluated_at': datetime.now(timezone.utc).isoformat(),
        'protocol': '0-shot. 각 선택지를 문맥 뒤에 공백 1개로 이어 붙여 BOS부터 인코딩하고, 선택지 토큰의 조건부 로그우도 합(acc) 또는 바이트 수로 나눈 값(acc_norm)이 가장 큰 선택지를 고른다. 문맥이 context_length를 넘으면 왼쪽을 자른다. 무작위 기준선은 1/선택지 수, 다수 클래스 기준선은 정답 분포의 최빈값 비율이다.'})
    print(out.read_text(), flush=True)


if __name__ == '__main__':
    main()
