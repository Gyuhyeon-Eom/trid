"""Free-form document QA rows for the open-weight-base track: the base instruct model writes questions about real
policy-briefing press releases and answers them from the document, in the product's document prompt format
(scripts/train_public_dialogue.document_prompt). Targets multi-fact answers ("비율과 한도") that single-span RAG rows
never teach.

  select  (local)  pick documents, excluding every evaluation source, normalize title/masthead -> <out>/docs.jsonl
  generate (GPU)   per document: 2 questions (one single fact, one combining two or more facts), then answers

Answers are screened like distilled chat (scripts/distill_trid_chat.screen): any quantity (date, amount, count...)
must appear in the document; vendor names dropped; the answer must cite [D1] and must not continue with a new
'질문:' line. Rows record document id, source URL and licence (KOGL Type 1) for attribution.
"""
import argparse
import hashlib
import html
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from collect_korea_briefing import clean_text  # noqa: E402
from distill_trid_chat import PERSONA, VENDOR, NAME_VARIANTS, QUANTITY, read_jsonl, write  # noqa: E402
from train_public_dialogue import document_prompt  # noqa: E402

EXCLUDE = (ROOT / 'data/institution-qa-frozen-01/source-docs.jsonl', ROOT / 'data/public-document-sft-source/documents.jsonl')
QUESTION_REQUEST = ('다음 보도자료를 읽고 문서 내용만으로 답할 수 있는 질문 2개를 만드세요.\n'
                    '1번은 금액·비율·기간·날짜·대상 같은 구체적 사실 하나를 묻습니다.\n'
                    '2번은 두 가지 이상 사실을 함께 묻습니다(예: 지원 비율과 최대 한도, 대상과 신청 방법).\n'
                    '질문만 한 줄에 하나씩, 번호 없이 쓰세요.\n\n[문서]\n{title}\n{text}')
ANSWER_SYSTEM = PERSONA + ' 문서 질문에는 문서에 적힌 내용만 사용하고, 질문이 묻는 사실을 빠짐없이 짧게 답한 뒤 끝에 [D1]을 붙이세요.'


def unescape(text):
    while (plain := html.unescape(text)) != text:
        text = plain
    return text


def select(source, output, count, seed, min_chars=600, max_chars=2400):
    skip_ids, skip_hashes = set(), set()
    for path in EXCLUDE:
        for doc in read_jsonl(path):
            skip_ids.add(doc['id'])
            skip_hashes.add(hashlib.sha256(doc['text'].encode()).hexdigest())
    pool = []
    with open(source) as f:
        for line in f:
            doc = json.loads(line)
            if doc['id'] in skip_ids or doc['sha256'] in skip_hashes or not (doc.get('date') or '') >= '2021':
                continue
            text = clean_text(doc['text'])
            if min_chars <= len(text) <= max_chars and '[D' not in text:
                pool.append({'id': doc['id'], 'title': unescape(doc['title']), 'date': doc['date'], 'text': text,
                             'source': doc['source'], 'attribution': doc['attribution'], 'license': doc['license']})
    random.Random(seed).shuffle(pool)
    rows = pool[:count]
    output.mkdir(parents=True)
    body = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
    (output / 'docs.jsonl').write_text(body)
    (output / 'provenance.json').write_text(json.dumps({
        'source': str(source.relative_to(ROOT)), 'eligible': len(pool), 'selected': len(rows), 'seed': seed,
        'filters': {'date': '>=2021', 'chars': [min_chars, max_chars]}, 'excluded_sources': [str(p.relative_to(ROOT)) for p in EXCLUDE],
        'docs_sha256': hashlib.sha256(body.encode()).hexdigest(), 'license': 'KOGL Type 1 (text only), attribution per row'},
        ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'eligible': len(pool), 'selected': len(rows)}))


def screen_question(question):
    """Teacher questions that echo the generation instructions are dropped: an unclosed '(' or a parenthetical
    note such as '(… 포함되어야 함', '(… 중 선택'."""
    if question.count('(') > question.count(')') or re.search(r'\([^)]*(포함되어야|명시된|찾아야|선택|예:)[^)]*\)?\s*$', question):
        return 'instruction_echo'
    return None


def screen_answer(doc, answer):
    if VENDOR.search(answer):
        return None, 'vendor_name'
    if '[D1]' not in answer:
        return None, 'no_citation'
    if answer.lstrip().startswith('[D1]'):
        return None, 'citation_first'
    if re.search(r'(?m)^\s*질문\s*[:：]', answer):
        return None, 'continued_question'
    if any(m.group(0) not in doc['text'] and m.group(0) not in doc['title'] for m in QUANTITY.finditer(answer)):
        return None, 'unsupported_quantity'
    return NAME_VARIANTS.sub('trid', answer), None


def rescreen(source, docs_dir, output):
    """Re-apply screen_question/screen_answer to an existing generated set; records the parent hash."""
    docs = {d['id']: d for d in read_jsonl(docs_dir / 'docs.jsonl')}
    parent = json.loads((source / 'provenance.json').read_text())
    rows, dropped = [], {}
    for row in read_jsonl(source / 'train.jsonl'):
        question = row['prompt'].rsplit('\n질문: ', 1)[1]
        reason = screen_question(question)
        if reason is None:
            kept, reason = screen_answer(docs[row['doc_id']], row['answer'])
        if reason:
            dropped[reason] = dropped.get(reason, 0) + 1
        else:
            rows.append({**row, 'answer': kept})
    write(output, rows, {**parent, 'parent': str(source), 'parent_train_sha256': parent['train_sha256'],
                         'rescreen_dropped': dropped})
    print(json.dumps({'rows': len(rows), 'rescreen_dropped': dropped}))


def generate(model_dir, model_id, revision, docs_dir, output, batch_size, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    docs = read_jsonl(docs_dir / 'docs.jsonl')
    tok = AutoTokenizer.from_pretrained(model_dir, padding_side='left')
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).to(device).eval()
    eot = tok.convert_tokens_to_ids('<|eot_id|>') if '<|eot_id|>' in tok.get_vocab() else tok.eos_token_id

    def run(conversations, max_new_tokens):
        out = []
        for i in range(0, len(conversations), batch_size):
            texts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in conversations[i:i + batch_size]]
            enc = tok(texts, return_tensors='pt', padding=True, add_special_tokens=False).to(device)
            with torch.no_grad():
                seqs = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=eot,
                                      pad_token_id=tok.pad_token_id)
            for seq in seqs[:, enc['input_ids'].shape[1]:].tolist():
                out.append(tok.decode(seq[:seq.index(eot)], skip_special_tokens=True).strip() if eot in seq else None)
            print(f'{min(i + batch_size, len(conversations))}/{len(conversations)}', flush=True)
        return out

    asked = run([[{'role': 'user', 'content': QUESTION_REQUEST.format(title=d['title'], text=d['text'])}] for d in docs], 128)
    pairs, dropped = [], {}
    for doc, text in zip(docs, asked):
        questions = [q.strip(' -*0123456789.)') for q in (text or '').splitlines() if q.strip()][:2]
        if len(questions) < 2:
            dropped['question_parse'] = dropped.get('question_parse', 0) + 1
        for kind, q in zip(('single_fact', 'multi_fact'), questions):
            reason = screen_question(q)
            if reason:
                dropped[reason] = dropped.get(reason, 0) + 1
            else:
                pairs.append((doc, kind, q))
    answers = run([[{'role': 'system', 'content': ANSWER_SYSTEM}, {'role': 'user', 'content': document_prompt(d, q)}]
                   for d, _, q in pairs], 256)
    rows = []
    for (doc, kind, q), answer in zip(pairs, answers):
        if answer is None:
            dropped['truncated'] = dropped.get('truncated', 0) + 1
            continue
        kept, reason = screen_answer(doc, answer)
        if kept is None:
            dropped[reason] = dropped.get(reason, 0) + 1
            continue
        rows.append({'prompt': document_prompt(doc, q), 'answer': kept, 'task': f'docqa_{kind}', 'history': [],
                     'doc_id': doc['id'], 'source': doc['source'], 'attribution': doc['attribution'], 'license': doc['license']})
    write(output, rows, {'protocol': __doc__, 'teacher': model_id, 'teacher_revision': revision,
                         'documents': len(docs), 'questions': len(pairs), 'dropped': dropped,
                         'docs_sha256': hashlib.sha256((docs_dir / 'docs.jsonl').read_bytes()).hexdigest()})
    print(json.dumps({'rows': len(rows), 'dropped': dropped}))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['select', 'generate', 'rescreen'])
    p.add_argument('--generated', type=Path, help='generate output to re-screen (rescreen)')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source', type=Path, default=ROOT / 'data/korea-kr-briefing-source/docs.jsonl')
    p.add_argument('--docs', type=Path, help='select output directory (generate)')
    p.add_argument('--count', type=int, default=1200)
    p.add_argument('--model', type=Path)
    p.add_argument('--model-id')
    p.add_argument('--revision')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--seed', type=int, default=291029)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.output.exists():
        p.error('Use a new output directory')
    if args.action == 'select':
        select(args.source, args.output, args.count, args.seed)
    elif args.action == 'rescreen':
        if not (args.generated and args.docs):
            p.error('rescreen needs --generated and --docs')
        rescreen(args.generated, args.docs, args.output)
    else:
        if not (args.docs and args.model and args.model_id and args.revision):
            p.error('generate needs --docs, --model, --model-id, --revision')
        generate(args.model, args.model_id, args.revision, args.docs, args.output, args.batch_size, args.device)


if __name__ == '__main__':
    main()
