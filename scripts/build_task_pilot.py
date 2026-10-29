"""Build the C-002 task pilot (condition, exception, table, combine, numeric change, explanation) from a spec of
real korea.kr KOGL Type 1 passages, then check it against the existing SFT loader and value-copy decoder.

Every answer value, quoted evidence and condition must be a verbatim span of the cited passage, and table rows
must occur in order in the raw document text; the build fails otherwise. Numeric-change rows are artificial
edits of a real passage and keep both the original and edited strings. Rows use the RAG prompt format of
scripts/train_public_dialogue.py (DOCUMENT_INSTRUCTION, [Dn] passages) with one or more distractor passages.
Split is by document. Questions and explanation framing are assistant-authored and not human-reviewed.

  python scripts/build_task_pilot.py build --spec reports/coordination/C-002/pilot-spec.json --output data/qa-task-pilot-01
  python scripts/build_task_pilot.py check --data data/qa-task-pilot-01 --report reports/coordination/C-002/compat.json
"""
import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from build_teacher_qa import EXCLUDE, passages, read_jsonl  # noqa: E402
from train_public_dialogue import DOCUMENT_INSTRUCTION  # noqa: E402

TOKENIZER = ROOT / 'runs/trid-korean-medium-cpt-gpu-complete/tokenizer.json'


def load_docs(source, wanted):
    docs = {}
    with Path(source).open() as f:
        for line in f:
            doc = json.loads(line)
            if doc['id'] in wanted:
                docs[doc['id']] = doc
                if len(docs) == len(wanted):
                    break
    missing = wanted - set(docs)
    if missing:
        raise ValueError(f'Documents not found: {sorted(missing)}')
    return docs


def table_text(spec, raw):
    if 'raw_span' in spec:
        start, end = spec['raw_span']
        i, j = raw.find(start), raw.find(end)
        if i < 0 or j < i:
            raise ValueError(f"Raw table span not found in {spec['doc']}")
        return raw[i:j + len(end)]
    for row in spec['rows']:
        cells = [re.escape(c) for c in row if c]
        if not re.search(r'\s+'.join(cells), raw):
            raise ValueError(f"Table row not found in order in {spec['doc']}: {row}")
    lines = ([spec['caption']] if spec['caption'] else []) + [' | '.join(spec['header'])] + [' | '.join(r) for r in spec['rows']]
    return '\n'.join(lines)


def require(span, text, what):
    if not span or span not in text:
        raise ValueError(f'{what} is not a verbatim span of its passage: {span!r}')


def assemble(item, kind, targets, distractor, split, docs, spec, rng):
    order = targets + ([distractor] if distractor else [])
    rng.shuffle(order)
    ids = {id(t): f'D{n}' for n, t in enumerate(order, 1)}
    body = '\n'.join(f'[{ids[id(t)]}] {t[1]}' for t in order)

    def sid(t):
        return ids[id(t)]
    if kind == 'combine':
        values = []
        for (doc, index, value), t in zip(item['parts'], targets):
            require(value, t[1], 'Combined value')
            if any(value in o[1] for o in order if o is not t):
                raise ValueError(f"{item['id']}: value {value!r} also appears in another passage")
            values.append(f'{value} [{sid(t)}]')
        answer = ', '.join(values)
    elif kind == 'explanation':
        t = targets[0]
        for field in ('conclusion', 'evidence'):
            require(item[field], t[1], field)
        if item['condition']:
            require(item['condition'], t[1], 'condition')
        condition = f"{item['condition']} [{sid(t)}]" if item['condition'] else '원문에 명시된 조건 없음'
        answer = f"결론: {item['conclusion']} [{sid(t)}]\n근거: “{item['evidence']}” [{sid(t)}]\n적용 조건: {condition}"
    else:
        t = targets[0]
        require(item['answer'], t[1], 'Answer')
        if any(item['answer'] in o[1] for o in order if o is not t):
            raise ValueError(f"{item['id']}: answer also appears in another passage")
        answer = f"{item['answer']} [{sid(t)}]"
    prov = [{'doc': d, 'source': docs[d]['source'], 'title': docs[d]['title'], 'date': docs[d].get('date'),
             'domain': spec['domains'].get(d), 'license': 'KOGL Type 1', 'passage_id': sid(t),
             'passage_sha256': hashlib.sha256(text.encode()).hexdigest(),
             **({'artificial_edit': True, 'original_sha256': hashlib.sha256(orig.encode()).hexdigest(),
                 'edits': item['edits']} if orig else {})}
            for t in order for d, text, orig in [t]]
    return dict(id=f"pilot-{item['id']}", prompt=f"{DOCUMENT_INSTRUCTION}{body}\n질문: {item['question']}", answer=answer,
                task=f'pilot_{kind}', group='qa_task_pilot', mixture_component='qa_task_pilot', split=split,
                provenance=prov, authored='question and explanation framing by AI assistant; spans verbatim; not human-reviewed',
                **({'note': item['note']} if 'note' in item else {}))


def build(spec_path, output):
    if output.exists():
        raise ValueError('Use a new output directory')
    from tokenizers import Tokenizer
    from trid.text import encode_example, prompt_ids
    spec = json.loads(Path(spec_path).read_text())
    excluded = {d['id'] for p in EXCLUDE for d in read_jsonl(p)}
    wanted = {i['doc'] for i in spec['items'] if 'doc' in i} | {p[0] for i in spec['items'] for p in i.get('parts', [])}
    wanted |= {t['doc'] for t in spec['tables'].values()} | {d for s in spec['distractors'].values() for d, _ in s}
    if wanted & excluded:
        raise ValueError(f'Evaluation documents used: {sorted(wanted & excluded)}')
    docs = load_docs(ROOT / spec['source'], wanted)
    cut = {d: passages(doc['text']) for d, doc in docs.items()}
    split_of = {d: 'valid' if d in spec['dev_docs'] else 'train' for d in docs}
    tok = Tokenizer.from_file(str(TOKENIZER))

    def fits(tok, row):
        try:
            encode_example(tok, row, 1024)
        except ValueError:
            return False
        return len(prompt_ids(tok, row['prompt'], None)) + 128 <= 1024

    rows, out = [], {'train': [], 'valid': []}
    for item in spec['items']:
        kind, targets = item['type'], []   # targets: (doc, text, original_text or None)
        if kind == 'table':
            table = spec['tables'][item['table']]
            targets.append((table['doc'], table_text(table, docs[table['doc']]['text']), None))
        elif kind == 'combine':
            for doc, index, _ in item['parts']:
                targets.append((doc, cut[doc][index], None))
        else:
            original = cut[item['doc']][item['passage']]
            text = original
            for old, new in item.get('edits', []):
                require(old, original, 'Edited original')
                text = text.replace(old, new)
            targets.append((item['doc'], text, original if item.get('edits') else None))
        splits = {split_of[d] for d, _, _ in targets}
        if len(splits) != 1:
            raise ValueError(f"{item['id']} mixes train and dev documents")
        split = splits.pop()
        rng = random.Random(item['id'])
        pool = [(d, cut[d][i]) for d, i in spec['distractors'][split] if d not in {t[0] for t in targets}]
        rng.shuffle(pool)
        row = None
        # One distractor passage, tried in seeded order then by length. Combine rows already carry two real
        # passages, so for them alone a row without a distractor is the last option; anything else fails loudly.
        options = [(d, text, None) for d, text in sorted(pool, key=lambda p: (p is not pool[0], len(p[1])))]
        for distractor in options + ([None] if kind == 'combine' else []):
            row = assemble(item, kind, targets, distractor, split, docs, spec, random.Random(item['id']))
            if fits(tok, row):
                break
            row = None
        if row is None:
            raise ValueError(f"{item['id']}: the prompt does not fit 1,024 tokens with a 128-token reserve")
        out[split].append(row)
    output.mkdir(parents=True)
    for split, data in out.items():
        (output / f'{split}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in data))
    meta = dict(purpose=spec['purpose'], spec=str(spec_path), spec_sha256=hashlib.sha256(Path(spec_path).read_bytes()).hexdigest(),
                rows={s: dict(Counter(r['task'] for r in d)) for s, d in out.items()},
                documents={s: sorted({p['doc'] for r in d for p in r['provenance']}) for s, d in out.items()},
                dev_rule=spec['dev_rule'], excluded_sets=[str(p) for p in EXCLUDE], tokenizer=str(TOKENIZER))
    if set(meta['documents']['train']) & set(meta['documents']['valid']):
        raise ValueError('Document appears in both splits')
    (output / 'provenance.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps(meta['rows'], ensure_ascii=False))


def check(data, report):
    """Existing loader, tokenizer fit and value-copy decoder acceptance of every gold answer. No training."""
    from tokenizers import Tokenizer
    from trid.constrained import EvidenceConstraint, token_bytes
    from trid.prompts import RAG_PREFIX, rag_passages
    from trid.text import encode_example, prompt_ids, read_rows
    tok = Tokenizer.from_file(str(TOKENIZER))
    vocab = token_bytes(tok)
    results = []
    for split in ('train', 'valid'):
        rows = read_rows(data / f'{split}.jsonl')
        for row in rows:
            body = row['prompt'][len(DOCUMENT_INSTRUCTION):]
            ps = rag_passages(RAG_PREFIX + body)
            ids = encode_example(tok, row, 1024)
            constraint = EvidenceConstraint(ps, vocab)
            accepted = constraint.complete(row['answer'].encode())
            results.append(dict(id=row['id'], task=row['task'], split=split, loader='ok', tokens=len(ids),
                                prompt_tokens=len(prompt_ids(tok, row['prompt'], None)),
                                answer_tokens=len(tok.encode(row['answer']).ids),
                                value_copy_decoder_accepts_gold=accepted))
    summary = {}
    for r in results:
        s = summary.setdefault(r['task'], Counter())
        s['rows'] += 1
        s['decoder_accepts'] += r['value_copy_decoder_accepts_gold']
        s['max_answer_tokens'] = max(s['max_answer_tokens'], r['answer_tokens'])
    report.write_text(json.dumps(dict(summary={k: dict(v) for k, v in summary.items()}, rows=results), ensure_ascii=False, indent=2))
    print(json.dumps({k: dict(v) for k, v in summary.items()}, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=['build', 'check'])
    p.add_argument('--spec', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--data', type=Path)
    p.add_argument('--report', type=Path)
    a = p.parse_args()
    if a.stage == 'build':
        build(a.spec, a.output)
    else:
        check(a.data, a.report)


if __name__ == '__main__':
    main()
