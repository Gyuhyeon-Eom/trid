"""Build reading-QA training rows from real KOGL Type 1 policy briefing passages with open-model drafts.

Stages (each resumable, writes under --work):
  select    seeded sample of korea.kr documents, minus frozen evaluation and public-document SFT documents,
            split into clean passages.
  generate  a generator model (llama.cpp server) drafts short-answer questions per passage as JSON.
  verify    a second, different model answers each question from the passage alone. A draft is kept only if
            its answer is a verbatim passage substring and the verifier's answer agrees with it.
  assemble  RAG-format rows `값 [Dn]` on shared contexts: two passages from different documents, with every question
            of both asked against the byte-identical context; some rows ask a third document's question and expect
            `확인 불가`. Train/valid split by document.

Open models only draft and check questions; every answer is copied from the source text. Generated questions
and the selection are machine-made and not human-reviewed.
"""
import argparse
import hashlib
import json
import random
import re
import sys
import time
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SOURCE = ROOT / 'data/korea-kr-briefing-source/docs.jsonl'
EXCLUDE = (ROOT / 'data/institution-qa-frozen-01/source-docs.jsonl', ROOT / 'data/public-document-sft-source/documents.jsonl')
ABSTAIN = '확인 불가'
MIN_CHARS, MAX_CHARS, MAX_ANSWER = 220, 650, 30
HANGUL = re.compile(r'[가-힣]')
# Release headers, embargo notes and contact blocks are not reading material.
BOILERPLATE = re.compile(r'보\s*도\s*일\s*시|보도\s*가능|엠바고|배\s*포\s*일\s*시|담당\s*부서|문의|☎|\d{2,4}-\d{3,4}-\d{4}|총\s*\d+\s*매')
# Answers must be values or noun phrases, not sentence endings.
CLAUSE = re.compile(r'(다|음|함|됨|임|요|니다)$')
# PDF extraction sometimes drops digits ("년 월부터", "약 억"); a unit with no number before it is unusable.
DANGLING_UNIT = re.compile(r'(^|[\s(~])(년|월|일|억|조|만|원|%|명|개|회)(?=[\s,)]|부터|까지|$)')
GEN_SCHEMA = {'type': 'array', 'minItems': 1, 'maxItems': 3, 'items': {
    'type': 'object', 'required': ['question', 'answer'],
    'properties': {'question': {'type': 'string'}, 'answer': {'type': 'string'}}}}
GEN_PROMPT = ('아래 문단만 읽고 답할 수 있는 한국어 질문을 최대 3개 만드세요. 답은 문단에 글자 그대로 있는 짧은 구절(30자 이하)이어야 합니다. '
              '숫자, 금액, 날짜, 기관, 이름, 대상, 장소 같은 구체적인 값을 묻고, 질문에 답을 넣지 마세요. '
              'JSON 배열 [{{"question": ..., "answer": ...}}]로만 출력하세요.\n\n문단:\n{passage}')
VERIFY_PROMPT = ('다음 문단만 근거로 질문에 답하세요. 문단에 있는 그대로의 짧은 구절만 출력하고 설명하지 마세요. '
                 '문단에 답이 없으면 없음이라고 쓰세요.\n\n문단:\n{passage}\n\n질문: {question}')


def norm(text):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text)).rstrip('.')


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def append_jsonl(path, rows):
    with Path(path).open('a') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def passages(text):
    """Merge blank-line paragraphs into 220-650 char passages; drop headers, tables and contact blocks."""
    out, current = [], ''
    for para in re.split(r'\n\s*\n', text):
        para = re.sub(r'\s+', ' ', para).strip()
        if len(para) < 25 or len(HANGUL.findall(para)) < .5 * len(para) or '[D' in para or BOILERPLATE.search(para):
            continue
        if current and len(current) + len(para) + 1 > MAX_CHARS:
            out.append(current)
            current = ''
        current = f'{current} {para}'.strip() if current else para
        if len(current) > MAX_CHARS:
            out.append(current[:MAX_CHARS].rsplit(' ', 1)[0])
            current = ''
    if current:
        out.append(current)
    return [p for p in out if MIN_CHARS <= len(p) <= MAX_CHARS]


def select(work, count, seed, exclude_selected=()):
    """`exclude_selected`: earlier work dirs whose documents must not be selected again."""
    excluded_ids, excluded_hashes = set(), set()
    for path in EXCLUDE:
        for doc in read_jsonl(path):
            excluded_ids.add(doc['id'])
            excluded_hashes.add(hashlib.sha256(doc['text'].encode()).hexdigest())
    for previous in exclude_selected:
        excluded_ids.update(d['id'] for d in read_jsonl(Path(previous) / 'selected.jsonl'))
    with SOURCE.open() as f:
        total = sum(1 for _ in f)
    chosen = set(random.Random(seed).sample(range(total), min(total, count * 3)))
    selected, skipped = [], Counter()
    with SOURCE.open() as f:
        for i, line in enumerate(f):
            if i not in chosen:
                continue
            doc = json.loads(line)
            if doc['id'] in excluded_ids or hashlib.sha256(doc['text'].encode()).hexdigest() in excluded_hashes:
                skipped['excluded'] += 1
                continue
            parts = passages(doc['text'])
            if not parts:
                skipped['no_passage'] += 1
                continue
            selected.append(dict(id=doc['id'], title=doc['title'], date=doc.get('date'), source=doc['source'],
                                 license=doc.get('license', 'KOGL-Type1'), passages=parts))
    random.Random(seed).shuffle(selected)
    selected = selected[:count]
    work.mkdir(parents=True, exist_ok=True)
    (work / 'selected.jsonl').write_text(''.join(json.dumps(d, ensure_ascii=False) + '\n' for d in selected))
    meta = dict(source=str(SOURCE), source_sha256_note='docs.jsonl is append-only crawl output; ids are stable',
                seed=seed, requested=count, selected=len(selected), skipped=dict(skipped),
                excluded_sets=[str(p) for p in EXCLUDE], excluded_previous_selections=[str(p) for p in exclude_selected],
                excluded_ids=len(excluded_ids))
    (work / 'select.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps(meta, ensure_ascii=False))


def chat(url, prompt, max_tokens, schema=None):
    def post(path, payload):
        req = urllib.request.Request(url + path, json.dumps(payload).encode(), {'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.load(r)
    rendered = post('/apply-template', {'messages': [{'role': 'user', 'content': prompt}],
                                        'chat_template_kwargs': {'enable_thinking': False}})['prompt']
    payload = dict(prompt=rendered, n_predict=max_tokens, temperature=0, seed=42, cache_prompt=False)
    if schema:
        payload['json_schema'] = schema
    return post('/completion', payload)['content'].strip()


def model_id(url):
    with urllib.request.urlopen(url + '/props', timeout=30) as r:
        return Path(json.load(r)['model_path']).name


def generate(work, url, passages_per_doc, workers):
    out = work / 'drafts.jsonl'
    done = {(r['doc_id'], r['passage_index']) for r in read_jsonl(out)} if out.exists() else set()
    teacher = model_id(url)
    jobs = []
    for doc in read_jsonl(work / 'selected.jsonl'):
        ranked = sorted(range(len(doc['passages'])), key=lambda i: -len(re.findall(r'\d', doc['passages'][i])))
        jobs += [(doc['id'], i, doc['passages'][i]) for i in ranked[:passages_per_doc] if (doc['id'], i) not in done]

    def draft(job):
        doc_id, index, passage = job
        try:
            return dict(doc_id=doc_id, passage_index=index, generator=teacher, error=None,
                        items=json.loads(chat(url, GEN_PROMPT.format(passage=passage), 256, GEN_SCHEMA)))
        except (ValueError, OSError) as exc:
            return dict(doc_id=doc_id, passage_index=index, generator=teacher, items=[], error=f'{type(exc).__name__}: {exc}')

    started = time.time()
    with ThreadPoolExecutor(workers) as pool:
        for n, row in enumerate(pool.map(draft, jobs), 1):
            append_jsonl(out, [row])
            if n % 50 == 0:
                print(f'generated {n}/{len(jobs)} passages, {n / (time.time() - started):.2f}/s', flush=True)


def digits_missing(answer):
    """A unit whose preceding token carries no number, e.g. '년 월부터', '약 억' (not '3천억 원')."""
    for match in DANGLING_UNIT.finditer(answer):
        before = answer[:match.start(2)].split()
        token = before[-1] if before else ''
        if not re.search(r'[\d천만억조백십]$', token.rstrip('(~')):
            return True
    return False


def screen(question, answer, passage):
    reasons = []
    if not answer or len(answer) > MAX_ANSWER:
        reasons.append('answer_length')
    if answer not in passage:
        reasons.append('answer_not_verbatim')
    if CLAUSE.search(answer):
        reasons.append('answer_is_clause')
    if digits_missing(answer):
        reasons.append('digits_missing')
    if not question.endswith('?') and not question.endswith('요'):
        reasons.append('not_a_question')
    if answer and norm(answer) in norm(question):
        reasons.append('answer_in_question')
    return reasons


def verify(work, url, workers):
    out = work / 'verified.jsonl'
    done = {(r['doc_id'], r['passage_index'], r['question']) for r in read_jsonl(out)} if out.exists() else set()
    docs = {d['id']: d for d in read_jsonl(work / 'selected.jsonl')}
    checker = model_id(url)
    jobs = []
    for draft in read_jsonl(work / 'drafts.jsonl'):
        passage = docs[draft['doc_id']]['passages'][draft['passage_index']]
        for item in draft['items']:
            question, answer = item['question'].strip(), item['answer'].strip().rstrip('.')
            if (draft['doc_id'], draft['passage_index'], question) not in done:
                jobs.append((draft, passage, question, answer))

    def check(job):
        draft, passage, question, answer = job
        reasons, reply = screen(question, answer, passage), None
        if not reasons:
            reply = chat(url, VERIFY_PROMPT.format(passage=passage, question=question), 48)
            a, c = norm(answer), norm(reply.split('\n')[0])
            if not c or (a not in c and c not in a):
                reasons.append('verifier_disagrees')
        return dict(doc_id=draft['doc_id'], passage_index=draft['passage_index'], question=question, answer=answer,
                    generator=draft['generator'], verifier=checker, verifier_answer=reply, kept=not reasons, reasons=reasons)

    with ThreadPoolExecutor(workers) as pool:
        for n, row in enumerate(pool.map(check, jobs), 1):
            append_jsonl(out, [row])
            if n % 200 == 0:
                print(f'verified {n}/{len(jobs)}', flush=True)


def assemble(work, output, tokenizer, missing_share, seed):
    """Shared contexts: two passages from different documents, and every verified question of both passages is
    asked against the byte-identical context, so only the question decides the answer and its source.

    A question is dropped when its answer string also occurs in the other passage (ambiguous source), or when the
    same question text is also asked of the other passage. Abstention rows ask a question from a third document's
    passage against the same context. Split by document.

    Conflicts are decided over all verified rows and quarantine the whole question key (written to
    quarantine.jsonl with every row): the same question on the same passage with different answers or verdicts,
    and the same question on different passages of one document with different kept answers. Inputs are sorted
    before any seeded shuffle, so the output does not depend on the order of verified.jsonl.
    """
    from tokenizers import Tokenizer
    from trid.text import encode_example, prompt_ids
    from scripts.train_public_dialogue import DOCUMENT_INSTRUCTION
    if output.exists():
        raise ValueError('Use a new data directory')
    tok = Tokenizer.from_file(str(tokenizer))
    rng = random.Random(seed)
    docs = {d['id']: d for d in read_jsonl(work / 'selected.jsonl')}
    verified = read_jsonl(work / 'verified.jsonl')
    per_key, per_doc_question = defaultdict(list), defaultdict(set)
    for r in verified:
        per_key[(r['doc_id'], r['passage_index'], norm(r['question']))].append(r)
        if r['kept']:
            per_doc_question[(r['doc_id'], norm(r['question']))].add(r['answer'])
    same_passage = {k for k, rs in per_key.items() if len({(r['answer'], r['kept']) for r in rs}) > 1}
    cross_passage = {k for k, answers in per_doc_question.items() if len(answers) > 1}
    quarantine = [dict(category='same_passage_conflict' if k in same_passage else 'cross_passage_conflict',
                       doc_id=k[0], passage_index=k[1], question=rs[0]['question'], rows=rs)
                  for k, rs in ((k, sorted(rs, key=lambda r: json.dumps(r, ensure_ascii=False, sort_keys=True)))
                                for k, rs in sorted(per_key.items()))
                  if k in same_passage or (k[0], k[2]) in cross_passage]
    kept = sorted((r for r in verified if r['kept']),
                  key=lambda r: (r['doc_id'], r['passage_index'], norm(r['question']), r['answer']))
    by_passage, dropped = {}, Counter()
    for item in kept:
        key = (item['doc_id'], item['passage_index'], norm(item['question']))
        if key in same_passage:
            dropped['quarantine_same_passage_conflict'] += 1
            continue
        if (key[0], key[2]) in cross_passage:
            dropped['quarantine_cross_passage_conflict'] += 1
            continue
        questions = by_passage.setdefault(key[:2], {})
        if key[2] in questions:
            dropped['duplicate'] += 1
            continue
        questions[key[2]] = item
    doc_ids = sorted({doc_id for doc_id, _ in by_passage})
    rng.shuffle(doc_ids)
    valid_docs = set(doc_ids[:max(1, len(doc_ids) // 20)])
    splits = {'train': [], 'valid': []}
    contexts = Counter()

    def fits(row):
        try:
            encode_example(tok, row, 1024)
            return len(prompt_ids(tok, row['prompt'], None)) + 128 <= 1024
        except ValueError:
            return False

    for split in splits:
        keys = [k for k in by_passage if (k[0] in valid_docs) == (split == 'valid')]
        rng.shuffle(keys)
        while len(keys) >= 2:
            a = keys.pop()
            partner = next((i for i in range(len(keys) - 1, max(-1, len(keys) - 8), -1) if keys[i][0] != a[0]), None)
            if partner is None:
                dropped['no_partner'] += 1
                continue
            b = keys.pop(partner)
            order = [a, b]
            rng.shuffle(order)
            texts = [docs[d]['passages'][i] for d, i in order]
            body = '\n'.join(f'[D{n}] {t}' for n, t in enumerate(texts, 1))
            rows = []
            for n, key in enumerate(order):
                other = norm(texts[1 - n])
                for question, item in by_passage[key].items():
                    if norm(item['answer']) in other:
                        dropped['answer_in_both_passages'] += 1
                        continue
                    if question in by_passage[order[1 - n]]:
                        dropped['question_in_both_passages'] += 1
                        continue
                    rows.append((item, f"{item['answer']} [D{n + 1}]", 'teacher_qa'))
            if len({answer for _, answer, _ in rows}) < 2:
                dropped['single_answer_context'] += len(rows)
                continue
            pool = [k for k in keys if k[0] not in (a[0], b[0])]
            n_missing = round(len(rows) * missing_share / (1 - missing_share))
            asked = set(by_passage[a]) | set(by_passage[b])
            for key in rng.sample(pool, min(n_missing, len(pool))):
                question, item = next(iter(by_passage[key].items()))
                if question not in asked and all(norm(item['answer']) not in norm(t) for t in texts):
                    rows.append((item, ABSTAIN, 'teacher_qa_missing'))
            built = []
            for item, answer, task in rows:
                doc = docs[item['doc_id']]
                built.append(dict(
                    id=f"teacher-{a[0]}-{a[1]}-{b[0]}-{b[1]}-{hashlib.sha1(item['question'].encode()).hexdigest()[:8]}",
                    prompt=f"{DOCUMENT_INSTRUCTION}{body}\n질문: {item['question']}", answer=answer, task=task,
                    group='teacher_qa', mixture_component='teacher_qa',
                    source=f"korea.kr policy briefing, KOGL Type 1 ({doc['source']}); question drafted by "
                           f"{item['generator']}, checked by {item['verifier']}; answer copied verbatim from the source",
                    source_guid=item['doc_id'], source_title=doc['title'],
                    context_id=hashlib.sha1(body.encode()).hexdigest()[:12]))
            if not all(fits(r) for r in built):
                dropped.update(f"context_too_long_{r['task']}" for r in built)
                continue
            splits[split] += built
            contexts[len(built)] += 1
        dropped['unpaired_passage_questions'] += sum(len(by_passage[k]) for k in keys)
    output.mkdir(parents=True)
    for split, rows in splits.items():
        (output / f'{split}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    (output / 'quarantine.jsonl').write_text(''.join(json.dumps(q, ensure_ascii=False) + '\n' for q in quarantine))
    meta = dict(purpose='Open-model-drafted, source-verbatim reading QA on real KOGL Type 1 policy briefings, shared contexts',
                method=__doc__, work=str(work), select=json.loads((work / 'select.json').read_text()),
                drafts=len(read_jsonl(work / 'drafts.jsonl')), questions=len(verified), kept=len(kept),
                rejected=dict(Counter(r for v in verified for r in v['reasons'])),
                rows={s: dict(Counter(r['task'] for r in rows)) for s, rows in splits.items()},
                rows_per_context=dict(sorted(contexts.items())), dropped=dict(dropped),
                quarantined_keys=dict(Counter(q['category'] for q in quarantine)),
                missing_share=missing_share, seed=seed, split='by document, 5% valid',
                license='KOGL Type 1 text; attribution kept per row',
                limitations='Questions are machine-drafted and machine-checked, not human-reviewed. Source documents may '
                            'have appeared in trid pretraining. Paired passages come from random other documents.')
    (output / 'provenance.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != 'method'}, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=['select', 'generate', 'verify', 'assemble'])
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--count', type=int, default=2000)
    p.add_argument('--seed', type=int, default=271031)
    p.add_argument('--url', default='http://127.0.0.1:18473')
    p.add_argument('--passages-per-doc', type=int, default=2)
    p.add_argument('--output', type=Path)
    p.add_argument('--exclude-selected', type=Path, action='append', default=[],
                   help='select: earlier work dir whose documents are skipped (repeatable)')
    p.add_argument('--tokenizer', type=Path, default=ROOT / 'runs/trid-korean-medium-cpt-gpu-complete/tokenizer.json')
    p.add_argument('--missing-share', type=float, default=.2)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    if a.stage == 'select':
        select(a.work, a.count, a.seed, a.exclude_selected)
    elif a.stage == 'generate':
        generate(a.work, a.url.rstrip('/'), a.passages_per_doc, a.workers)
    elif a.stage == 'verify':
        verify(a.work, a.url.rstrip('/'), a.workers)
    else:
        if a.output is None:
            p.error('--output required')
        assemble(a.work, a.output, a.tokenizer, a.missing_share, a.seed)


if __name__ == '__main__':
    main()
