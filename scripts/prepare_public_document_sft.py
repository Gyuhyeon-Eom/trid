"""Build dated, source-grounded generative SFT; never modifies reader experiments."""
import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer
from trid.text import encode_example, read_rows

ROOT = Path(__file__).resolve().parents[1]


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_quote(doc, fact):
    start, end = fact['start'], fact['end']
    if not (0 <= start < end <= len(doc['text'])) or doc['text'][start:end] != fact['quote']:
        raise ValueError('Evidence offset/quote mismatch')


def build(source, output, tokenizer, frozen, context=1024):
    if output.exists():
        raise ValueError('Use a new output directory; existing datasets are immutable')
    docs = rows(source / 'documents.jsonl')
    annotations = rows(source / 'annotations.jsonl')
    excluded = rows(frozen)
    excluded_ids = {d['id'] for d in excluded}
    excluded_hashes = {digest(d['text']) for d in excluded}
    if len({d['id'] for d in docs}) != len(docs) or len({digest(d['text']) for d in docs}) != len(docs):
        raise ValueError('Duplicate source document')
    by_id = {d['id']: d for d in docs}
    if any(a['source_id'] not in by_id for a in annotations):
        raise ValueError('Unknown annotation source')
    tok = Tokenizer.from_file(str(tokenizer))
    splits = {'train': [], 'valid': []}
    # Fixed source-ID ordering, not row-level random splitting. Not an unseen-pretraining benchmark.
    valid_ids = set(sorted(by_id)[::5])
    max_tokens = 0
    for doc in docs:
        if doc['id'] in excluded_ids or digest(doc['text']) in excluded_hashes:
            raise ValueError('Frozen source overlap')
        if doc['license'] != 'KOGL-Type1' or not doc['attribution'] or not doc['source'].startswith('https://www.korea.kr/'):
            raise ValueError('Missing text reuse attribution')
        if digest(doc['text']) != doc['sha256']:
            raise ValueError('Source checksum mismatch')
        facts = [a for a in annotations if a['source_id'] == doc['id']]
        if len(facts) != 2:
            raise ValueError('Expected two reviewed facts per document')
        for fact in facts:
            validate_quote(doc, fact)
        evidence = '\n\n'.join(a['quote'] for a in facts)
        prefix = (f"다음은 {doc['date']}에 발표된 문서의 발췌입니다. 당시 문서에 근거해 답하고 [D1]을 인용하세요. "
                  '현재 시행 여부나 문서 밖의 사실은 단정하지 마세요.\n'
                  f"[D1] {doc['title']}\n발표일: {doc['date']}\n{evidence}\n\n질문: ")
        examples = [
            ('extractive_summary', '이 발췌문의 두 핵심 내용을 정리해 주세요.',
             '문서 발췌의 핵심 내용은 다음과 같습니다.\n' + '\n'.join('- ' + a['quote'] + ' [D1]' for a in facts), []),
            *[('grounded_qa', a['question'], a['quote'] + ' [D1]', []) for a in facts],
            ('followup', '방금 답한 내용의 근거 문장을 그대로 보여 주세요.', facts[0]['quote'] + ' [D1]',
             [{'user': facts[0]['question'], 'assistant': facts[0]['quote'] + ' [D1]'}]),
            ('temporal_limit', '이 내용이 지금도 그대로 적용된다고 보면 될까요?',
             f"이 자료는 {doc['date']} 발표 문서입니다. [D1] 이 발췌만으로 현재도 동일하게 적용되는지는 확인할 수 없습니다. 최신 공고나 담당 기관 안내를 확인해야 합니다.", []),
        ]
        # Explicit source-specific arithmetic annotations, never inferred from matching numbers.
        if doc['id'] == 'korea-kr-press-156612711':
            for cost in ('3000', '10000'):
                result = min(Decimal(cost) * Decimal('0.70'), Decimal('4000'))
                examples.append(('calculation', f'선정 기업의 인정되는 총제작비가 {cost}만원이면 문서 기준 정부지원금 상한은 얼마인가요?',
                    f'총제작비의 70%와 4천만원 한도 중 작은 금액입니다. {cost}만원 × 70% = {Decimal(cost) * Decimal("0.70"):g}만원이므로 지원금 상한은 {result:g}만원입니다. 실제 지급액을 보장하는 계산은 아닙니다. [D1]', []))
            examples.append(('clarification', '우리 회사는 지원금을 얼마 받을 수 있나요?',
                '이 문서에서는 선정 기업에 총제작비의 70%까지, 최대 4천만원을 지원한다고 안내합니다. [D1] 계산하려면 인정되는 총제작비가 얼마인지 알아야 합니다. 총제작비는 얼마이며 지원 대상으로 선정되셨나요?', []))
        for i, (task, question, answer, history) in enumerate(examples):
            row = {'id': f"{doc['id']}-{i}", 'prompt': prefix + question, 'answer': answer,
                   'history': history, 'task': task, 'group': 'public_document_sft',
                   'source_id': doc['id'], 'source_url': doc['source'], 'source_date': doc['date'],
                   'source_sha256': doc['sha256'], 'license': doc['license'], 'attribution': doc['attribution'],
                   'evidence': [{'start': a['start'], 'end': a['end'], 'quote': a['quote']} for a in facts]}
            ids, _ = encode_example(tok, row, context)
            max_tokens = max(max_tokens, len(ids))
            splits['valid' if doc['id'] in valid_ids else 'train'].append(row)
    output.mkdir(parents=True)
    for split, data in splits.items():
        (output / f'{split}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in data))
        read_rows(output / f'{split}.jsonl')
    provenance = {
        'source': '대한민국 정책브리핑 보도자료; KOGL-Type1 text only',
        'license_reference': 'https://www.kogl.or.kr/info/license.do',
        'annotation_method': 'Assistant-authored questions with manually selected verbatim source evidence; summaries are extractive, not independently human reviewed.',
        'scope': '20 historical Jan 2024 sources. Generative SFT only, not extractive-reader training. No claim of model improvement. No fabricated source documents.',
        'limitations': 'Small, narrow dated dataset; repeated task wording. Source-ID/exact-text frozen exclusion only, not semantic near-duplicate exclusion. Sources may have appeared in pretraining; valid is a development split, not an independent benchmark. Temporal-limit examples are not missing-evidence labels.',
        'documents': len(docs), 'source_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (source / 'documents.jsonl', source / 'annotations.jsonl')},
        'frozen_source_manifest_sha256': hashlib.sha256(frozen.read_bytes()).hexdigest(),
        'tokenizer_sha256': hashlib.sha256(tokenizer.read_bytes()).hexdigest(), 'context_length': context,
        'max_encoded_tokens': max_tokens, 'split_documents': {'train': len(docs) - len(valid_ids), 'valid': len(valid_ids)},
        'rows': {s: len(r) for s, r in splits.items()}, 'tasks': dict(Counter(r['task'] for data in splits.values() for r in data)),
        'split_hashes': {s: hashlib.sha256((output / f'{s}.jsonl').read_bytes()).hexdigest() for s in splits},
    }
    (output / 'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n')
    (output / 'ATTRIBUTION.txt').write_text('\n\n'.join(f"{d['title']} ({d['date']})\n{d['source']}\n{d['attribution']}\nKOGL-Type1; 텍스트만 이용. 질문·답변 재구성: trid." for d in docs) + '\n')
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/public-document-sft-source')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, default=ROOT / 'runs/trid-korean-medium-base-v1/tokenizer.json')
    parser.add_argument('--frozen-sources', type=Path, default=ROOT / 'data/institution-qa-frozen-01/source-docs.jsonl')
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output, args.tokenizer, args.frozen_sources), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
