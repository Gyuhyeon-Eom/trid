import hashlib
import importlib.util
import json
import random
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
spec = importlib.util.spec_from_file_location('build_pretraining_mix', SCRIPTS / 'build_pretraining_mix.py')
mix = importlib.util.module_from_spec(spec); spec.loader.exec_module(mix)

WORDS = '지원금 신청 기간 담당 부서 예산 집행 계획 공고 위원회 심의 결과 보고 기준 조례 시행 규칙 별표 부칙 개정'.split()


def article(seed, n=120):
    rng = random.Random(seed)
    return ' '.join(rng.choice(WORDS) + str(rng.randrange(1000)) for _ in range(n))


def write_source(root, name, docs):
    path = Path(root) / name
    path.mkdir()
    with (path / 'docs.jsonl').open('w', encoding='utf-8') as f:
        for i, (group, text) in enumerate(docs):
            f.write(json.dumps({'id': f'{name}-{i}', 'title': '', 'source': '', 'text': text,
                                'sha256': hashlib.sha256(text.encode()).hexdigest(), 'license': name,
                                'attribution': name, 'group': group, 'date': None, 'kind': name},
                               ensure_ascii=False) + '\n')
    (path / 'ATTRIBUTION.txt').write_text(name)
    return path


def read_out(out):
    return {s: [json.loads(l) for l in (Path(out) / f'{s}.jsonl').read_text().splitlines()] for s in ('train', 'valid', 'test')}


class MixTests(unittest.TestCase):
    def test_duplicates_keep_the_higher_priority_source(self):
        with tempfile.TemporaryDirectory() as d:
            base = [article(i) for i in range(300)]
            near = base[0].split(); near[5] = '변경됨'; near = ' '.join(near)  # one word changed
            law = write_source(d, 'law', [(f'law:{i}', t) for i, t in enumerate(base)])
            web = write_source(d, 'web', [('web:a', near), ('web:b', base[1].replace(' ', '  ')), ('web:c', article(999))])
            mix.build([(law, 10**9), (web, 10**9)], Path(d) / 'out', 10, quality=None)
            rows = [r for part in read_out(Path(d) / 'out').values() for r in part]
            ids = {r['id'] for r in rows}
            self.assertEqual({i for i in ids if i.startswith('web')}, {'web-2'})
            self.assertEqual(len([i for i in ids if i.startswith('law')]), 300)
            prov = json.loads((Path(d) / 'out/provenance.json').read_text())
            self.assertEqual(prov['filter_counts']['web:exact_duplicate'], 1)
            self.assertEqual(prov['filter_counts']['web:near_duplicate'], 1)

    def near(self, texts):
        import numpy as np
        sig = mix.Signer()
        return mix.near_duplicates(np.vstack([sig(t) for t in texts]), lambda i, k: mix.jaccard(texts[i], texts[k]))

    def test_distinct_documents_are_not_merged(self):
        self.assertEqual(self.near([article(i) for i in range(400)]), {})

    def test_similar_but_below_threshold_is_kept(self):
        # Same template, different facts: exact Jaccard ~0.7, which the MinHash estimate alone can push over 0.8.
        base = article(7, 300).split()
        variant = ' '.join('다른값' + str(i) if i % 30 == 0 else w for i, w in enumerate(base))
        texts = [' '.join(base), variant]
        self.assertTrue(0.6 < mix.jaccard(*texts) < 0.8)
        self.assertEqual(self.near(texts), {})
        texts.append(' '.join(base[:-1]))  # true near-duplicate of row 0
        self.assertEqual(set(self.near(texts)), {2})

    def test_groups_never_straddle_splits(self):
        with tempfile.TemporaryDirectory() as d:
            docs = [(f'law:{g}', article(g * 10 + v)) for g in range(400) for v in range(3)]
            src = write_source(d, 'law', docs)
            mix.build([(src, 10**9)], Path(d) / 'out', 10, quality=None)
            where = defaultdict(set)
            for split, rows in read_out(Path(d) / 'out').items():
                for r in rows:
                    where[r['group']].add(split)
            self.assertTrue(all(len(s) == 1 for s in where.values()))
            self.assertTrue({'train', 'valid', 'test'} <= set().union(*where.values()))

    def test_token_budget_is_a_ceiling(self):
        with tempfile.TemporaryDirectory() as d:
            docs = [(f'web:{i}', article(i)) for i in range(500)]
            src = write_source(d, 'web', docs)
            (src / 'pilot-stats.json').write_text(json.dumps({'chars_per_token_sample': 2.0}))
            total = sum(len(t) for _, t in docs) / 2.0
            mix.build([(src, int(total * 0.4))], Path(d) / 'out', 10, quality=None)
            kept = sum(len(r['text']) for part in read_out(Path(d) / 'out').values() for r in part) / 2.0
            self.assertLessEqual(kept, total * 0.4)
            self.assertGreater(kept, total * 0.35)

    def test_budget_suffixes(self):
        self.assertEqual(mix.budget('350M'), 350_000_000)
        self.assertEqual(mix.budget('1.2B'), 1_200_000_000)
        self.assertEqual(mix.budget('5000'), 5000)


if __name__ == '__main__':
    unittest.main()
