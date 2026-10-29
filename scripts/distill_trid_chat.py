"""Self-distilled conversational SFT rows for trid: the base instruct model answers trid's own dialogue prompts
under a trid persona system prompt; training later omits the system prompt, so trid keeps the base model's
conversational quality and learns its own identity (context distillation).

Prompts come from trid-authored dialogue sources (base-dialogue-v1, dialogue-v5, the conversational rows of
korean-rag-curriculum-v4); answers are never taken from those sources. Prompts that normalize to any evaluation
prompt (public-dialogue probes/regressions, dialogue-eval-dev/final, base-dialogue-v1 evaluation) are excluded.
Greedy decoding; rows that hit the token limit or name the base model/vendor are dropped and counted.

  python scripts/distill_trid_chat.py --model <hf dir> --output data/trid-chat-distill-v1 --device cuda
"""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PERSONA = ('당신은 trid입니다. 대한민국 공공기관 내부망에서 동작하는 한국어 AI로, 기관 문서를 근거로 질문에 답하고 '
           '업무 대화와 글쓰기를 돕습니다. 인터넷 검색, 실시간 정보 확인, 파일 저장은 하지 않습니다. '
           '모르는 것은 모른다고 말하고, 근거 없는 사실을 지어내지 않습니다. 자연스럽고 간결한 존댓말로 답하세요.')
SOURCES = (('base-dialogue-v1', ROOT / 'data/base-dialogue-v1/train.jsonl'),
           ('dialogue-v5', ROOT / 'data/dialogue-v5/train.jsonl'),
           ('rag-curriculum-v4-conversation', ROOT / 'data/korean-rag-curriculum-v4/train.jsonl'))
EVAL_PROMPTS = (ROOT / 'data/dialogue-eval-dev.jsonl', ROOT / 'data/dialogue-eval-final.jsonl',
                ROOT / 'data/base-dialogue-v1/evaluation.jsonl')
PUBLIC_PROBES = ('안녕', '너한테 할 수 있는 질문이 뭐야', '제작비 지원 비율과 최대 한도가 어떻게 돼?',
                 '반가워, 처음 왔어.', '어떤 일을 부탁하면 되니?', '문서를 읽다가 모르는 게 있으면 물어봐도 돼?')
VENDOR = re.compile(r'(?i)kanana|카나나|kakao|카카오')


def key(text):
    return re.sub(r'[\W_]+', '', text).lower()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def prompts(limit, rng):
    banned = {key(p) for p in PUBLIC_PROBES}
    for path in EVAL_PROMPTS:
        for r in read_jsonl(path):
            banned |= {key(t['prompt']) for t in r.get('turns', [r])}
    seen, rows = set(), []
    for source, path in SOURCES:
        for r in read_jsonl(path):
            if source.startswith('rag') and not re.match(r'^(basic|work|followup)', r.get('task', '')):
                continue
            history = r.get('history') or []
            k = key(' '.join([h['user'] for h in history] + [r['prompt']]))
            if not k or k in seen or key(r['prompt']) in banned:
                continue
            seen.add(k)
            rows.append({'prompt': r['prompt'], 'history': history, 'source': source, 'source_task': r.get('task')})
    rng.shuffle(rows)
    return rows[:limit], len(rows)


QUANTITY = re.compile(r'\d+(?:[.,]\d+)*\s*(?:년|월|일|시|분|초|원|%|퍼센트|명|개|건|세|살|만|억|천|km|kg|도)')
NAME_VARIANTS = re.compile(r'트리디\s*\(\s*TRID\s*\)|TRID|트리디|트리드')


def screen(row):
    """(row, None) or (None, reason). Drops answers naming the base model/vendor and answers asserting a quantity
    (date, amount, count, duration...) absent from the conversation, i.e. teacher hallucinations such as today's date
    or an institution's settlement deadline. Normalizes the teacher's spellings of the assistant name to `trid`."""
    context = row['prompt'] + ' '.join(h['user'] + ' ' + h['assistant'] for h in row.get('history') or [])
    answer = row['answer']
    if not answer.strip():
        return None, 'empty'
    if VENDOR.search(answer):
        return None, 'vendor_name'
    if any(m.group(0) not in context for m in QUANTITY.finditer(answer)):
        return None, 'unsupported_quantity'
    return {**row, 'answer': NAME_VARIANTS.sub('trid', answer)}, None


def write(output, rows, provenance):
    output.mkdir(parents=True)
    body = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
    (output / 'train.jsonl').write_text(body)
    provenance.update(rows=len(rows), train_sha256=hashlib.sha256(body.encode()).hexdigest())
    (output / 'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n')


def refilter(source, output):
    """Re-screen an existing distilled set with the current screen(); records the parent hash."""
    parent = json.loads((source / 'provenance.json').read_text())
    rows, dropped = [], {}
    for row in read_jsonl(source / 'train.jsonl'):
        kept, reason = screen(row)
        if kept is None:
            dropped[reason] = dropped.get(reason, 0) + 1
        else:
            rows.append(kept)
    write(output, rows, {**parent, 'parent': str(source), 'parent_train_sha256': parent['train_sha256'],
                         'screen': screen.__doc__, 'screen_dropped': dropped})
    print(json.dumps({'rows': len(rows), 'screen_dropped': dropped}))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', type=Path)
    p.add_argument('--model-id')
    p.add_argument('--revision')
    p.add_argument('--refilter', type=Path, help='re-screen this distilled directory instead of generating')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prompts', type=int, default=1500)
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--seed', type=int, default=291029)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.output.exists():
        p.error('Use a new output directory')
    if args.refilter:
        return refilter(args.refilter, args.output)
    if not (args.model and args.model_id and args.revision):
        p.error('--model, --model-id and --revision are required for generation')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rng = random.Random(args.seed)
    chosen, available = prompts(args.prompts, rng)
    tok = AutoTokenizer.from_pretrained(args.model, padding_side='left')
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(args.device).eval()
    eot = tok.convert_tokens_to_ids('<|eot_id|>') if '<|eot_id|>' in tok.get_vocab() else tok.eos_token_id
    rows, dropped = [], {'truncated': 0}
    for i in range(0, len(chosen), args.batch_size):
        batch = chosen[i:i + args.batch_size]
        texts = []
        for r in batch:
            messages = [{'role': 'system', 'content': PERSONA}]
            for h in r['history']:
                messages += [{'role': 'user', 'content': h['user']}, {'role': 'assistant', 'content': h['assistant']}]
            messages.append({'role': 'user', 'content': r['prompt']})
            texts.append(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        enc = tok(texts, return_tensors='pt', padding=True, add_special_tokens=False).to(args.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, eos_token_id=eot,
                                 pad_token_id=tok.pad_token_id)
        for r, seq in zip(batch, out[:, enc['input_ids'].shape[1]:].tolist()):
            if eot not in seq:
                dropped['truncated'] += 1
                continue
            kept, reason = screen({**r, 'answer': tok.decode(seq[:seq.index(eot)], skip_special_tokens=True).strip(),
                                   'task': 'chat_distilled'})
            if kept is None:
                dropped[reason] = dropped.get(reason, 0) + 1
            else:
                rows.append(kept)
        print(f'{min(i + args.batch_size, len(chosen))}/{len(chosen)} kept {len(rows)}', flush=True)
    write(args.output, rows, {
        'protocol': __doc__, 'teacher': args.model_id, 'teacher_revision': args.revision, 'persona': PERSONA,
        'prompt_sources': [s for s, _ in SOURCES], 'excluded_eval_prompts': [str(q.relative_to(ROOT)) for q in EVAL_PROMPTS] + list(PUBLIC_PROBES),
        'available_unique_prompts': available, 'prompts': len(chosen), 'dropped': dropped,
        'screen': screen.__doc__, 'max_new_tokens': args.max_new_tokens, 'seed': args.seed})
    print(json.dumps(dropped))


if __name__ == '__main__':
    main()
