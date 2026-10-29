"""Question sensitivity: does the answer change when only the question changes?

Held-out KLUE MRC v1.1 official dev (never used for training): every paragraph with at least two answerable
questions whose gold answers differ. Each question is asked with the byte-identical single-passage RAG310-format
prompt (`trid.prompts.rag_prompt`, passage id fixed per paragraph), greedy free generation, 64 tokens.

Per paragraph: `all_same` = every question got the identical output (the shortcut this measures);
`distinct_ratio` = distinct outputs / questions. Per question: `correct` = output cites the passage id and its
value contains a gold answer span; `abstained` = output contains 확인 불가.

  python scripts/evaluate_question_sensitivity.py --run runs/<run> --output reports/<name>.json
"""
import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / 'data/klue-mrc-source/dev.json'
CITATION = re.compile(r'\[(D[1-9])\]')


def norm(text):
    return unicodedata.normalize('NFC', text).strip()


def paragraphs(limit=0):
    """[(passage_id, passage, [(question, [gold spans])])], deterministic."""
    rng = random.Random(291029)
    out = []
    for rec in json.loads(SOURCE.read_text())['data']:
        for p in rec['paragraphs']:
            qas = [(norm(q['question']), sorted({norm(a['text']) for a in q['answers'] if a.get('text')}))
                   for q in p['qas'] if not q['is_impossible']]
            qas = [(q, g) for q, g in qas if g]
            if len({g[0] for _, g in qas}) >= 2:
                out.append((f'D{rng.randint(1, 9)}', norm(p['context']), qas))
    return out[:limit] if limit else out


def score(generated, passage_id, golds):
    cites = set(CITATION.findall(generated))
    value = CITATION.sub('', generated).strip()
    return {'correct': cites == {passage_id} and any(g in value for g in golds),
            'abstained': '확인 불가' in generated}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Use a new output file')
    import torch
    from trid.prompts import rag_prompt
    from trid.runtime import generate, load_run
    from trid.text import prompt_ids
    torch.set_num_threads(4)
    model, tok, device = load_run(a.run, a.device)
    weights = hashlib.sha256((a.run / 'model.safetensors').read_bytes()).hexdigest()
    results, skipped = [], 0
    for passage_id, passage, qas in paragraphs(a.limit):
        prompts = [rag_prompt([(passage_id, passage)], q) for q, _ in qas]
        if max(len(prompt_ids(tok, x)) for x in prompts) + 64 > model.cfg.context_length:
            skipped += 1
            continue
        answers = [generate(model, tok, x, device, max_tokens=64)['answer'] for x in prompts]
        questions = [{'question': q, 'gold': g, 'generated': out, **score(out, passage_id, g)}
                     for (q, g), out in zip(qas, answers)]
        results.append({'passage_id': passage_id, 'passage': passage, 'all_same': len(set(answers)) == 1,
                        'distinct_ratio': round(len(set(answers)) / len(answers), 4), 'questions': questions})
    n_q = sum(len(r['questions']) for r in results)
    summary = {'paragraphs': len(results), 'questions': n_q, 'skipped_context': skipped,
               'all_same_rate': round(sum(r['all_same'] for r in results) / len(results), 4),
               'mean_distinct_ratio': round(sum(r['distinct_ratio'] for r in results) / len(results), 4),
               'correct': sum(q['correct'] for r in results for q in r['questions']),
               'abstained': sum(q['abstained'] for r in results for q in r['questions'])}
    summary['correct_rate'] = round(summary['correct'] / n_q, 4)
    report = {'run': str(a.run), 'model_sha256': weights, 'source': str(SOURCE.relative_to(ROOT)),
              'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(), 'protocol': __doc__.strip(),
              'summary': summary, 'results': results}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
