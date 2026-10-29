"""Shared-context real QA: the same question/answer items as the parent mix, regrouped so every context has several
questions with different answers.

Diagnosis (2026-10-29): in public-dialogue-mix-04, 12,831 of 13,115 real-QA contexts occur with exactly one question,
so `context -> answer` minimizes the loss without reading the question; gpu-05-mix04-s3000 gives the same answer to
every question about a fixed passage. This builder changes only the context composition:

- Items: every real-QA row of the parent (KLUE answerable/missing/hard, TyDi), reduced to
  (own passage, question, answer span or abstention). Questions and answer spans are copied unchanged.
- Groups: passages are packed (seeded shuffle) into contexts of up to `--max-passages` passages that fit the
  1,024-token context with a 128-token generation reserve; each item becomes one row whose context is its group.
  Within a group every row shares the byte-identical context, so only the question decides the answer and source.
- Groups whose rows would all have the same answer are dissolved and re-packed; leftovers are dropped and counted.
- Non-real components, their weights, the real-QA total mass and the validation file are unchanged.

  python scripts/prepare_contrastive_qa.py --parent data/public-dialogue-mix-04 --output data/public-dialogue-mix-05
"""
import argparse
import hashlib
import json
import random
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCUMENT_INSTRUCTION = ('다음은 발표일 미상의 문서입니다. 문서 질문은 근거에 따라 답하고 출처 번호를 인용하세요. '
                        '인사나 사용 안내에는 자연스럽게 답하세요. 현재 시행 여부는 단정하지 마세요.\n')
ABSTAIN = '확인 불가'
CONTEXT, RESERVE = 1024, 128


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def norm(text):
    return unicodedata.normalize('NFC', text).strip()


def klue_passages():
    """guid -> NFC passage text, as prepare_rag_curriculum_v2 normalizes KLUE contexts."""
    data = json.loads((ROOT / 'data/klue-mrc-source/train.json').read_text())['data']
    return {qa['guid']: norm(p['context']) for rec in data for p in rec['paragraphs'] for qa in p['qas']}


def items_of(parent_rows):
    """(passage, question, answer_span|None, row) for every parent real-QA row; the passage is the row's own."""
    from trid.prompts import RAG_PREFIX, rag_passages
    klue = klue_passages()
    items, counts = [], Counter()
    for row in parent_rows:
        if not row['prompt'].startswith(DOCUMENT_INSTRUCTION):
            raise ValueError(f"Unexpected real QA prompt {row['id']}")
        body = RAG_PREFIX + row['prompt'][len(DOCUMENT_INSTRUCTION):]
        passages = dict(rag_passages(body))
        question = body.rsplit('\n질문: ', 1)[1]
        if row['answer'] == ABSTAIN:
            own = klue.get(row['source_guid'])
            if own is None or own not in passages.values():
                counts['missing_without_own_passage'] += 1
                continue
            items.append((own, question, None, row))
            continue
        span, _, cite = row['answer'].rpartition(' [')
        source = cite.rstrip(']')
        if not span or source not in passages or span not in passages[source]:
            counts['unparsed_answer'] += 1
            continue
        items.append((passages[source], question, span, row))
    return items, counts


def render(group, ids):
    """group: [(passage, [items])] -> (body, {passage: id}); passage order is the group order."""
    label = {passage: sid for (passage, _), sid in zip(group, ids)}
    return '\n'.join(f'[{label[p]}] {p}' for p, _ in group), label


def build(items, tok, max_passages, seed):
    from trid.text import prompt_ids
    rng = random.Random(seed)
    by_passage = {}
    for item in items:
        by_passage.setdefault(item[0], []).append(item)
    passages = list(by_passage)
    rng.shuffle(passages)
    length = {p: len(tok.encode(f'[D1] {p}\n').ids) for p in passages}
    longest_q = {p: max(len(tok.encode(i[1]).ids) for i in by_passage[p]) for p in passages}
    fixed = len(prompt_ids(tok, DOCUMENT_INSTRUCTION + '\n질문: '))
    budget = CONTEXT - RESERVE - fixed

    def fits(group):
        return sum(length[p] for p in group) + max(longest_q[p] for p in group) <= budget

    def distinct(group):
        return len({(p, i[2]) if i[2] else ABSTAIN for p in group for i in by_passage[p]}) >= 2

    # First-fit packing in seeded random order; a passage joins the first earlier group it fits into.
    groups, open_groups, counts = [], [], Counter()
    for passage in passages:
        if not fits([passage]):
            counts['passage_too_long'] += 1
            continue
        for group in open_groups:
            if fits(group + [passage]):
                group.append(passage)
                if len(group) == max_passages:
                    open_groups.remove(group)
                break
        else:
            groups.append([passage])
            open_groups.append(groups[-1])
    # A lone passage whose questions share one answer is not contrastive: move it into another group or drop it.
    rows, lone = [g for g in groups if distinct(g)], [p for g in groups if not distinct(g) for p in g]
    dropped = 0
    for passage in lone:
        target = next((g for g in rows if len(g) < max_passages and fits(g + [passage])), None)
        if target is None:
            dropped += 1
        else:
            target.append(passage)
    counts['dropped_single_answer_passages'] = dropped
    out = []
    for g, group in enumerate(rows):
        rng.shuffle(group)
        ids = rng.sample([f'D{k}' for k in range(1, 10)], len(group))
        body, label = render([(p, by_passage[p]) for p in group], ids)
        for passage in group:
            for _, question, span, row in by_passage[passage]:
                answer = ABSTAIN if span is None else f'{span} [{label[passage]}]'
                new = {k: v for k, v in row.items() if k not in ('prompt', 'answer', 'weight', 'distractor')}
                new.update(id=f"{row['id']}-shared", prompt=f'{DOCUMENT_INSTRUCTION}{body}\n질문: {question}', answer=answer,
                           context_group=f'shared-{g:05d}', passages_in_context=len(group))
                out.append(new)
    return out, counts


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--parent', type=Path, default=ROOT / 'data/public-dialogue-mix-04')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, default=ROOT / 'runs/trid-public-dialogue-01/tokenizer.json')
    p.add_argument('--max-passages', type=int, default=3)
    p.add_argument('--seed', type=int, default=291029)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Use a new data directory')
    import sys
    sys.path.insert(0, str(ROOT))
    from tokenizers import Tokenizer
    from trid.prompts import RAG_PREFIX, rag_passages
    from trid.text import encode_example, prompt_ids
    tok = Tokenizer.from_file(str(a.tokenizer))
    parent = read_jsonl(a.parent / 'train.jsonl')
    kept = [r for r in parent if r['mixture_component'] != 'real_qa']
    real = [r for r in parent if r['mixture_component'] == 'real_qa']
    real_mass = sum(r['weight'] for r in real)
    items, counts = items_of(real)
    rows, build_counts = build(items, tok, a.max_passages, a.seed)
    counts.update(build_counts)
    for row in rows:
        encode_example(tok, row, CONTEXT)
        if len(prompt_ids(tok, row['prompt'])) + RESERVE > CONTEXT:
            raise ValueError(f"generation reserve violated: {row['id']}")
        row['weight'] = real_mass / len(rows)
    eval_passages = {text.strip() for r in read_jsonl(ROOT / 'data/korean-rag-eval-v1/test.jsonl')
                     for _, text in rag_passages(r['prompt']) if len(text.strip()) >= 40}
    overlap = sum(text.strip() in eval_passages for r in rows
                  for _, text in rag_passages(RAG_PREFIX + r['prompt'][len(DOCUMENT_INSTRUCTION):]))
    if overlap:
        raise ValueError(f'{overlap} training passages also appear in RAG310')
    groups = Counter(r['context_group'] for r in rows)
    answers = {}
    for r in rows:
        answers.setdefault(r['context_group'], set()).add(r['answer'])
    a.output.mkdir(parents=True)
    (a.output / 'train.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in kept + rows))
    (a.output / 'valid.jsonl').write_bytes((a.parent / 'valid.jsonl').read_bytes())
    metadata = {
        'purpose': 'Single-variable experiment: real-QA context composition only (shared contexts, several questions each)',
        'parent_data': str(a.parent), 'seed': a.seed, 'max_passages': a.max_passages,
        'real_qa_rows': {'parent': len(real), 'items_parsed': len(items), 'after': len(rows)},
        'real_qa_tasks': dict(Counter(r['task'] for r in rows)),
        'contexts': len(groups), 'questions_per_context': dict(sorted(Counter(groups.values()).items())),
        'contexts_with_single_answer': sum(len(v) < 2 for v in answers.values()),
        'passages_per_context': dict(sorted(Counter(r['passages_in_context'] for r in rows).items())),
        'real_qa_abstain_share': round(sum(r['answer'] == ABSTAIN for r in rows) / len(rows), 4),
        'counts': dict(counts), 'real_qa_total_mass_unchanged': round(real_mass, 6),
        'validation': 'byte-identical copy of parent', 'rag310_source_overlap': 0,
        'tokenizer_sha256': hashlib.sha256(a.tokenizer.read_bytes()).hexdigest(),
        'limitations': 'Hard same-title distractors of the parent are replaced by other grouped passages; abstention items '
                       'keep their own passage only. Questions and answer spans are unchanged.',
        'parent_provenance': json.loads((a.parent / 'provenance.json').read_text())}
    (a.output / 'provenance.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in metadata.items() if k != 'parent_provenance'}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
