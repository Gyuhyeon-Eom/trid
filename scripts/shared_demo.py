"""Temporary multi-visitor demo. Only accepts requests from the authenticated edge proxy."""
import argparse
from dataclasses import dataclass, field
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import html
import json
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit, unquote

import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_external_chat import GRAMMAR, MAX_TOKENS, SYSTEM, complete, digest, evidence_grammar, request as llama_request
from trid.case_prompts import CASE_SYSTEM
from trid.extractive import load_reader, read as read_extractive, passage_spans
from trid.prompts import rag_prompt
from trid.pdf_documents import Document, DocumentError, make_chunks, answer_document, retrieve

ROOT = Path(__file__).resolve().parents[1]
TTL = 15 * 60
MAX_BODY = 2 * 1024 * 1024
MAX_SESSIONS = 32
# Source-first answers, one or two pairs like the value-first GRAMMAR: "[D1] 값" or "[D1] 값, [D2] 값".
SOURCE_FIRST_ANSWER = re.compile(r'\[(D[1-9])\]\s*([^\[\],\n]+?)(?:\s*,\s*\[(D[1-9])\]\s*([^\[\],\n]+?))?\.?')

def read_chunk(model, _presence, tok, text, question, device):
    """Adapt prompt-relative extractive spans to the document reader's chunk offsets."""
    prompt = rag_prompt([('D1', text)], question)
    result = read_extractive(model, tok, prompt, device)
    abstains = result['presence'] != 0
    span = None
    answer = result['answer']
    if not abstains:
        sid, lo, hi = passage_spans(prompt)[0]
        start, end = result['span']
        if result['source'] != sid or not lo <= start < end <= hi:
            raise DocumentError('모델의 원문 위치 검증에 실패했습니다.')
        span = [start - lo, end - lo]
        answer = result['span_text']
        if text[span[0]:span[1]] != answer:
            raise DocumentError('모델의 원문 위치 검증에 실패했습니다.')
    return {**result, 'answer': answer, 'abstains': abstains, 'source_character_span': span}

def answer_document_chat(document, question, base, family, context, constrained=False, selected=None):
    """One completion over the retrieved chunks (as D1..Dn), with the evaluation's `grammar_instructed` system prompt
    and optionally its `evidence_constrained` grammar. `value [Dn]` and source-first `[Dn] value` are accepted (the
    evaluation's content_* reading); an answer is shown only when every value is found verbatim in the chunk it
    cites, otherwise the reply is a non-answer and the raw output is not displayed as an answer."""
    question = question.strip()
    if not question:
        raise DocumentError('질문을 입력해 주세요.')
    selected = [(chunk, score) for chunk, score in (retrieve(document, question) if selected is None else selected)
                if '[D' not in chunk.text]
    result = {'file': document.name, 'candidates': [], 'checked_chunks': len(selected),
              'total_chunks': len(document.chunks), 'warnings': document.warnings, 'raw_output': None,
              'scope': '관련도 상위 최대 6개 구간을 확인한 결과입니다. 미검색 구간까지 정보가 없다고 보장하지 않습니다.'}
    if not selected:
        return {**result, 'status': 'not_found', 'answer': '질문과 관련된 문서 구간을 찾지 못했습니다. 질문을 바꾸거나 추출 내용을 확인해 주세요.'}
    prompt = rag_prompt([(f'D{i}', chunk.text) for i, (chunk, _) in enumerate(selected, 1)], question)
    messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}]
    r = complete(base, messages, family, MAX_TOKENS, context, evidence_grammar(prompt) if constrained else None)
    if r.get('skipped_context'):
        raise DocumentError('검색된 문서 구간이 모델 문맥 한도를 넘었습니다. 질문을 더 구체적으로 입력해 주세요.')
    output = r['generated']
    result['raw_output'] = output
    if output.removesuffix('.') == '확인 불가':
        return {**result, 'status': 'not_found', 'answer': '확인한 문서 구간에서 답을 찾지 못했습니다(모델 보류).'}
    if output.removesuffix('.') == '근거 충돌':
        return {**result, 'status': 'review', 'answer': '모델이 근거 충돌로 판단했습니다. /source로 원문을 확인해 주세요.'}
    match = GRAMMAR.fullmatch(output)
    source_first = None if match else SOURCE_FIRST_ANSWER.fullmatch(output)
    pairs = ([(match[1], match[2]), (match[3], match[4])] if match
             else [(source_first[2], source_first[1]), (source_first[4], source_first[3])] if source_first else [])
    candidates = []
    for value, sid in pairs:
        if value is None:
            continue
        value, index = value.strip(), int(sid[1:]) - 1
        chunk = selected[index][0] if index < len(selected) else None
        start = chunk.text.find(value) if chunk and value else -1
        if start < 0:
            return {**result, 'status': 'review', 'answer': '모델 출력이 원문 위치 검증을 통과하지 못해 답으로 표시하지 않습니다.'}
        candidates.append({'page': chunk.page, 'text': chunk.text, 'page_text_start': chunk.start,
                           'retrieval_score': round(selected[index][1], 3),
                           'page_character_span': [chunk.start + start, chunk.start + start + len(value)],
                           'prediction': {'answer': value, 'source': sid, 'abstains': False, 'presence': 0,
                                          'source_character_span': [start, start + len(value)]}})
    if not candidates:
        return {**result, 'status': 'review', 'answer': '모델 출력이 형식 검증을 통과하지 못해 답으로 표시하지 않습니다.'}
    return {**result, 'status': 'answer', 'candidates': candidates,
            'answer': ', '.join(f"{c['prediction']['answer']} [{c['prediction']['source']}]" for c in candidates)}


LEADING_TAGS = re.compile(r'^(?:\s*(?:\[[^\]\n]{1,12}\]|D[1-9]\s*:)\s*)+')
NUMBER = re.compile(r'\d[\d,.]*')
MIN_QUOTE_CHARS = 4


def _numbers(text):
    return {n.rstrip('.,').replace(',', '') for n in NUMBER.findall(text)}


def answer_document_case(document, question, base, family, context, max_tokens=256, selected=None):
    """Serving contract of CASE-trained versions (explanatory answers, trained with CASE_SYSTEM). The model is asked
    with its training instruction; its output is never shown as a verified answer. Only output sentences that occur
    verbatim in a retrieved passage are shown as verified quotes, highlighted in the source. The rest is returned as
    `commentary` explicitly marked unverified, with numbers that appear in no retrieved passage nor the question listed
    as warnings (a warning list, not a truth check). Repeated sentences are collapsed."""
    question = question.strip()
    if not question:
        raise DocumentError('질문을 입력해 주세요.')
    selected = [(chunk, score) for chunk, score in (retrieve(document, question) if selected is None else selected)
                if '[D' not in chunk.text]
    result = {'file': document.name, 'candidates': [], 'checked_chunks': len(selected), 'answer_mode': 'case',
              'total_chunks': len(document.chunks), 'warnings': document.warnings, 'raw_output': None,
              'commentary': None, 'commentary_verified': False, 'unsupported_numbers': [],
              'scope': '관련도 상위 최대 6개 구간을 확인한 결과입니다. 미검색 구간까지 정보가 없다고 보장하지 않습니다.'}
    if not selected:
        return {**result, 'status': 'not_found', 'answer': '질문과 관련된 문서 구간을 찾지 못했습니다. 질문을 바꾸거나 추출 내용을 확인해 주세요.'}
    prompt = rag_prompt([(f'D{i}', chunk.text) for i, (chunk, _) in enumerate(selected, 1)], question)
    messages = [{'role': 'system', 'content': CASE_SYSTEM}, {'role': 'user', 'content': prompt}]
    r = complete(base, messages, family, max_tokens, context)
    if r.get('skipped_context'):
        raise DocumentError('검색된 문서 구간이 모델 문맥 한도를 넘었습니다. 질문을 더 구체적으로 입력해 주세요.')
    output = r['generated']
    # Echoing the question or the instructions back is a failed answer (a known defect of the pilot), not a quote.
    flat = _norm(output)
    echoed = (len(_norm(question)) >= 8 and _norm(question) in flat) or any(
        _norm(text) in flat for text in ('제공된 근거로만 답하세요', '근거에 있는 결론과 그 이유', '근거에 없는 내용은 추측하지'))
    if echoed or output.lstrip().startswith('질문:'):
        return {**result, 'raw_output': output, 'status': 'model_failure',
                'answer': '모델이 질문이나 지시문을 그대로 옮겨 적어 답변에 실패했습니다. 이 출력은 표시하지 않습니다 '
                          '(TRID 1.7B 실험판에서 확인된 결함).'}
    segments, seen, repeats = [], set(), 0
    for part in re.split(r'(?<=[.!?])\s+|\n+', output):
        part = part.strip()
        if not part:
            continue
        if part in seen:
            repeats += 1
            continue
        seen.add(part)
        segments.append(part)
    candidates, quoted = [], set()
    # A length-truncated last sentence is a fragment; it can match the source verbatim but is not a full quote.
    for part in segments[:-1] if r.get('truncated_output') and segments else segments:
        quote = LEADING_TAGS.sub('', part).strip()
        for text in (quote, quote.rstrip('.')):
            if len(text) < MIN_QUOTE_CHARS or text in quoted:
                continue
            hit = next(((i, chunk, chunk.text.find(text)) for i, (chunk, _) in enumerate(selected)
                        if text in chunk.text), None)
            if hit:
                i, chunk, start = hit
                quoted.add(text)
                candidates.append({'page': chunk.page, 'text': chunk.text, 'page_text_start': chunk.start,
                                   'retrieval_score': round(selected[i][1], 3),
                                   'page_character_span': [chunk.start + start, chunk.start + start + len(text)],
                                   'prediction': {'answer': text, 'source': f'D{i + 1}', 'abstains': False, 'presence': 0,
                                                  'source_character_span': [start, start + len(text)]}})
                break
    support = _numbers('\n'.join(chunk.text for chunk, _ in selected) + '\n' + question)
    notes, shown, withheld = [], [], 0
    for part in segments:
        # A sentence carrying a number found in no retrieved passage nor the question is withheld, not just flagged.
        if _numbers(part) - support:
            withheld += 1
        else:
            shown.append(part)
    if withheld:
        notes.append(f'검색된 원문과 질문에 없는 숫자가 들어 있는 해설 문장 {withheld}개를 숨겼습니다.')
    if repeats:
        notes.append(f'같은 문장 반복 {repeats}회를 생략했습니다.')
    if r.get('truncated_output'):
        notes.append(f'모델 출력이 길이 한도({max_tokens}토큰)에서 잘렸습니다.')
    result.update(raw_output=output, commentary='\n'.join(shown) or None, commentary_notes=notes,
                  unsupported_numbers=sorted(_numbers(output) - support), candidates=candidates)
    if candidates:
        return {**result, 'status': 'quoted',
                'answer': f'원문과 글자 그대로 일치하는 인용 {len(candidates)}건을 아래에 표시합니다. 모델 해설은 검증되지 않았습니다.'}
    return {**result, 'status': 'unverified',
            'answer': '원문과 글자 그대로 일치하는 인용이 없어 검증된 답이 없습니다. 아래 모델 해설은 검증되지 않았습니다.'}



@dataclass
class Session:
    csrf: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    touched: float = field(default_factory=time.monotonic)
    document: object = None
    document_id: str | None = None
    calls: list = field(default_factory=list)
    pinned_doc: str | None = None
    offered_docs: list = field(default_factory=list)


@dataclass(frozen=True)
class PassageChunk:
    """A workspace passage in the shape the answer functions expect (text/page/start), plus its provenance."""
    text: str
    page: int
    start: int
    doc_id: str
    path: tuple


NORMALIZE = re.compile(r'[\s·ㆍ・「」『』()（）\[\]〈〉<>,.\-]')


def _norm(text):
    return NORMALIZE.sub('', text or '')


def _longest_shared(a, b):
    """Length of the longest substring of `a` that also occurs in `b` (both normalised)."""
    best = 0
    for i in range(len(a)):
        for j in range(len(a), i + best, -1):
            if a[i:j] in b:
                best = j - i
                break
    return best


class Workspace:
    """Questions without an uploaded document search the shared index of approved public sources
    (scripts/build_workspace_index.py). Selection uses the question's own metadata before ranking:
      1. an institution named in the question restricts the search to that institution's documents; when the corpus
         has none, the reply says so (no fallback to another institution);
      2. a document title (or a long part of one) named in the question restricts it to those documents;
      3. otherwise, when the best matches belong to different institutions with comparable scores, the reply asks in
         plain language which institution's rule is meant instead of picking one.
    Each answer uses passages of ONE document, so rules of different institutions are never mixed."""
    MAX_PASSAGES = 4  # CPU prompt processing dominates latency; four 550-char passages keep answers short
    TITLE_MIN = 8      # normalised characters shared with a title to count as naming that document
    CLOSE = 0.85       # another institution's best score within this ratio of the top makes the question ambiguous

    def __init__(self, directory):
        from trid.service.index import Index
        root = Path(directory)
        self.index = Index.load(root)
        self.meta = json.loads((root / 'meta.json').read_text())
        self.manifest = json.loads((root / 'manifest.json').read_text())
        self.name = root.name
        self.passage_doc = np.array([p.doc_id for p in self.index.passages])
        self.by_institution = {}
        for doc_id, m in self.meta.items():
            name = (m['institution'] or '').split('(질의')[0]
            if len(_norm(name)) >= 3:
                self.by_institution.setdefault(_norm(name), set()).add(doc_id)
        self.norm_titles = {doc_id: _norm(m['title']) for doc_id, m in self.meta.items()}

    def scope(self):
        kinds = ' · '.join(f'{k} {n:,}건' for k, n in self.manifest['by_kind'].items())
        lo, hi = self.manifest['date_range']
        return (f"검색 자료 {self.manifest['documents']:,}건({kinds}; 기준일 {lo}~{hi}): 법제처 국가법령정보센터 "
                f"행정규칙·법령해석례와 정책브리핑 보도자료(공공누리 1유형). 기관 내부 비공개 규정·편람은 들어 있지 않습니다. "
                f"현행 여부는 수집 시점(2026-10-27~29) 표시 기준이며 다시 확인하지 않았습니다.")

    def describe(self, doc_id, score=None):
        m = self.meta[doc_id]
        return {'doc_id': doc_id, 'title': m['title'], 'institution': m['institution'], 'kind': m['kind'],
                'date': m['date'], 'date_type': m['date_type'], 'effective_date': m['effective_date'],
                'currency': m['currency'], 'source': m['source'], 'license': m['license'],
                **({'score': round(score, 2)} if score is not None else {})}

    INSTITUTION_LIKE = re.compile(r'[가-힣]{2,}(?:공사|공단|위원회|재단|진흥원|연구원|은행|센터|특별시|광역시|특별자치시|특별자치도|청|처|부)(?![가-힣])')
    WEAK = 0.6  # best score inside the named institution below this share of the corpus-wide best: no relevant rule

    def constraints(self, question):
        """(named institutions, unknown institution-like names, allowed doc ids or None, named document ids or None).
        An institution's documents are those it issues (meta) or whose title carries its name (regulations issued by
        a department of the institution)."""
        q = _norm(question)
        named = [name for name in self.by_institution if name in q]
        named = [n for n in named if not any(n != m and n in m for m in named)]  # keep the longest names
        # An institution-like name is known only as a corpus institution or as the leading word of a rule's title
        # ("해양경찰청 공무국외출장 …"); a mention inside another body's title (an interpretation about it) is not its rule.
        unknown = [w for w in self.INSTITUTION_LIKE.findall(question)
                   if not any(_norm(w) in n or n in _norm(w) for n in named) and _norm(w) not in self.by_institution
                   and not any(t.startswith(_norm(w)) for t in self.norm_titles.values())]
        allowed = None
        if named:
            allowed = set().union(*(self.by_institution[n] for n in named))
            allowed |= {d for d, t in self.norm_titles.items() if any(t.startswith(n) for n in named)}
        # Title matching ignores institution names, so naming an institution alone never counts as naming a document.
        q_title = q
        for name in named + [_norm(w) for w in self.INSTITUTION_LIKE.findall(question)]:
            q_title = q_title.replace(name, '|')
        pool = allowed if allowed is not None else self.norm_titles.keys()
        shared = {d: _longest_shared(self.norm_titles[d], q_title) for d in pool}
        top = max(shared.values(), default=0)
        titled = {d for d, n in shared.items() if n == top} if top >= self.TITLE_MIN else None
        return named, unknown, allowed, titled

    def select(self, question, pinned=None):
        """dict(status, applied, chunks, others, institutions, ...). status: 'answer' | 'clarify' | 'unknown_institution'
        | 'institution_no_match' | 'not_found'."""
        from trid.service.index import tokenize_all
        named, unknown, allowed, titled = self.constraints(question)
        result = {'applied': None, 'chunks': [], 'others': [], 'institutions': named, 'unknown_institutions': unknown,
                  'title_matched': bool(titled)}
        if unknown and not named and not pinned:
            return {**result, 'status': 'unknown_institution'}
        restrict = {pinned} if pinned else (titled or allowed)
        tokens = tokenize_all([question])[0]
        if not tokens:
            return {**result, 'status': 'not_found'}
        everywhere = self.index.bm25.get_scores(tokens)
        scores = everywhere
        if restrict is not None:
            scores = np.where(np.isin(self.passage_doc, list(restrict)), everywhere, -np.inf)
            if named and not titled and not pinned and scores.max() < self.WEAK * everywhere.max():
                return {**result, 'status': 'institution_no_match'}
        order = [int(i) for i in np.argsort(-scores, kind='stable')[:60] if scores[i] > 0]
        if not order:
            return {**result, 'status': 'not_found'}
        best = {}
        for i in order:
            best.setdefault(str(self.passage_doc[i]), float(scores[i]))
        applied = str(self.passage_doc[order[0]])
        others = [self.describe(d, s) for d, s in best.items() if d != applied][:5]
        result.update(others=others)
        if not (pinned or titled or named):
            inst = lambda d: (self.meta[d]['institution'] or self.meta[d]['title']).split('(질의')[0]
            top = best[applied]
            rivals = {}
            for d, s in best.items():
                if inst(d) != inst(applied) and s >= self.CLOSE * top:
                    rivals.setdefault(inst(d), d)
            if rivals:
                options = [inst(applied)] + list(rivals)[:3]
                return {**result, 'status': 'clarify', 'institutions': options,
                        'candidates': [self.describe(applied, top)] + [self.describe(d, best[d]) for d in rivals.values()][:3]}
        chunks = [(PassageChunk(self.index.passages[i].text, 1, self.index.passages[i].char_start,
                                self.passage_doc[i], self.index.passages[i].path), float(scores[i]))
                  for i in order if self.passage_doc[i] == applied][:self.MAX_PASSAGES]
        return {**result, 'status': 'answer', 'applied': str(applied), 'chunks': chunks}



def workspace_reply(workspace, selection):
    """Reply without a model call when the question cannot be tied to one institution's rule in the corpus."""
    status = selection['status']
    if status == 'clarify':
        options = '\n'.join(f"- {c['institution'] or '기관 표기 없음'}: {c['title']}" for c in selection['candidates'])
        answer = ('이 질문은 여러 기관의 규정에서 비슷하게 검색됩니다. 기관마다 기준이 다를 수 있어 임의로 한 곳을 고르지 '
                  f'않았습니다.\n{options}\n어느 기관(또는 어떤 규정) 기준인지 질문에 기관명이나 규정명을 넣어 다시 물어봐 주세요.')
    elif status == 'unknown_institution':
        names = ', '.join(selection['unknown_institutions'])
        answer = (f"질문에 적힌 '{names}'의 규정은 현재 검색 자료에 없습니다. 다른 기관의 규정으로 대신 답하지 않습니다. "
                  '해당 기관 규정을 /pdf 또는 /paste로 넣으면 그 문서에서 찾을 수 있습니다.')
    elif status == 'institution_no_match':
        names = ', '.join(selection['institutions'])
        answer = (f"'{names}' 자료에서 이 질문과 관련된 규정을 찾지 못했습니다. 다른 기관의 규정으로 대신 답하지 않습니다.")
    else:
        answer = '검색 자료에서 질문과 관련된 구간을 찾지 못했습니다. 질문을 바꾸거나 기관명·규정명을 넣어 주세요.'
    return {'status': status, 'answer': answer, 'candidates': [], 'checked_chunks': 0, 'warnings': [],
            'scope': '모델을 실행하지 않았습니다. 검색 자료와 질문의 기관·규정명을 확인한 결과입니다.'}


PUBLIC_FIELDS = ('id', 'label', 'kind', 'base_model', 'license', 'version', 'status', 'evaluation', 'limits')


class State:
    """Serves the versions listed in a versions file. The first `extractive` entry also supplies the chunking
    tokenizer; `llama` entries talk to a loopback llama.cpp server whose model file hash must match the file."""
    def __init__(self, versions):
        self.versions, self.children = {}, []
        if not versions or versions[0]['kind'] != 'extractive':
            raise ValueError('첫 버전은 추출 독해기여야 합니다(문서 구간 분할 토크나이저).')
        try:
            for meta in versions:
                if meta['kind'] == 'extractive':
                    metadata = json.loads((ROOT / meta['run'] / 'training.json').read_text())
                    if metadata.get('status') != 'complete':
                        raise ValueError('학습 완료 모델만 공유할 수 있습니다.')
                    model, tok, device = load_reader(ROOT / meta['run'], device='cpu')
                    self.versions[meta['id']] = dict(meta, model=model, tok=tok, device=device)
                elif meta['kind'] == 'llama':
                    self.versions[meta['id']] = dict(meta, **self.start_llama(meta))
                else:
                    raise ValueError(f"unknown version kind {meta['kind']}")
        except BaseException:
            self.close()
            raise
        self.tok = self.versions[versions[0]['id']]['tok']
        self.default = versions[0]['id']
        self.run = self.versions[self.default]['label']
        self.sessions = {}
        self.lock = threading.RLock()
        self.work = threading.Lock()

    def start_llama(self, meta):
        model_file = ROOT / meta['model_file']
        if digest(model_file) != meta['model_sha256']:
            raise ValueError(f"{meta['id']}: model file hash mismatch")
        port = meta['port']
        try:
            llama_request(f'http://127.0.0.1:{port}', '/health')
            raise RuntimeError(f"{meta['id']}: port {port} is already serving; refusing to adopt an unmanaged server")
        except OSError:
            pass
        log = open(ROOT / '.shared-demo' / f"llama-{meta['id']}.log", 'wb') if (ROOT / '.shared-demo').is_dir() else subprocess.DEVNULL
        child = subprocess.Popen([str(ROOT / meta['llama_server']), '-m', str(model_file), '--host', '127.0.0.1', '--port', str(port),
                                  '-c', str(meta['context']), '-np', '1', '-t', str(meta.get('threads', 4)),
                                  '-tb', str(meta.get('threads', 4)), '--jinja'],
                                 stdout=log, stderr=log)
        self.children.append(child)
        base = f'http://127.0.0.1:{port}'
        for _ in range(240):
            if child.poll() is not None:
                raise RuntimeError(f"{meta['id']}: llama-server exited")
            try:
                props = llama_request(base, '/props')
                if Path(props['model_path']).resolve() == model_file.resolve():
                    return {'base': base, 'context': props['default_generation_settings']['n_ctx']}
                raise RuntimeError(f"{meta['id']}: another model is served on port {port}")
            except OSError:
                time.sleep(.5)
        raise RuntimeError(f"{meta['id']}: llama-server readiness timeout")

    def public_versions(self):
        return [{k: v[k] for k in PUBLIC_FIELDS} for v in self.versions.values()]

    def close(self):
        for child in self.children:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()

    def prune(self):
        with self.lock:
            now = time.monotonic()
            for sid in list(self.sessions):
                if now - self.sessions[sid].touched > TTL:
                    del self.sessions[sid]


def parse_pdf(body, name, tok):
    # Parsing untrusted PDFs happens in a separate process with a wall-clock timeout.
    try:
        process = subprocess.Popen(
            [sys.executable, str(ROOT / 'scripts/shared_pdf_extract.py')],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            cwd=ROOT,
        )
        output = []
        def communicate():
            output.append(process.communicate(body)[0])
        communication = threading.Thread(target=communicate, daemon=True)
        communication.start()
        deadline = time.monotonic() + 15
        try:
            while process.poll() is None:
                if time.monotonic() > deadline:
                    raise DocumentError('PDF 처리 시간이 초과됐습니다. 더 작은 문서를 넣어 주세요.')
                if sys.platform == 'darwin':
                    memory = subprocess.run(['/bin/ps', '-o', 'rss=', '-p', str(process.pid)], capture_output=True, timeout=2).stdout.strip()
                    if memory and int(memory) > 512 * 1024:
                        raise DocumentError('PDF 처리 메모리 한도를 넘었습니다. 더 작은 문서를 넣어 주세요.')
                communication.join(.1)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=3)
            communication.join(3)
        if process.returncode or not output:
            raise DocumentError('PDF를 읽지 못했습니다. 작은 텍스트 PDF로 다시 시도해 주세요.')
        result = json.loads(output[0])
    except (subprocess.SubprocessError, ValueError):
        raise DocumentError('PDF를 읽지 못했습니다. 텍스트가 있는 작은 PDF로 다시 시도해 주세요.')
    if 'error' in result:
        raise DocumentError(result['error'])
    name = name.replace('\\', '/').rsplit('/', 1)[-1][:180] or 'document.pdf'
    return Document(name, result['pages'], make_chunks(result['pages'], tok), result['warnings'])


class BoundedServer(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, *args):
        self.slots = threading.BoundedSemaphore(8)
        super().__init__(*args)

    def process_request(self, request, address):
        if not self.slots.acquire(False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, address)
        except Exception:
            self.slots.release()
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self.slots.release()


def make_server(state, secret, port=0, hostname='test.tridatum.co'):
    if len(secret) < 32:
        raise ValueError('Proxy secret must contain at least 32 characters')
    assets = ROOT / 'trid/shared_ui'
    origin = 'https://' + hostname
    workspace = getattr(state, 'workspace', None)
    workspace_scope = html.escape(workspace.scope()) if workspace else '공유 검색 자료가 없습니다. 문서를 직접 넣어 주세요.'
    version_options = ''.join(
        f'<option value="{html.escape(v["id"], quote=True)}">{html.escape(v["label"])}</option>'
        for v in state.public_versions()
    )
    version_cards = ''.join(
        f'<pre class="muted version-card" data-version="{html.escape(v["id"], quote=True)}"'
        f'{"" if v["id"] == state.default else " hidden"}>'
        + html.escape(f'{v["label"]} · {v["version"]} · {v["status"]}\n기반: {v["base_model"]} · 라이선스: {v["license"]}\n'
                      f'실측: {v["evaluation"]}\n한계: {v["limits"]}') + '</pre>'
        for v in state.public_versions()
    )

    class Handler(BaseHTTPRequestHandler):
        server_version = 'trid'
        sys_version = ''

        def setup(self):
            super().setup()
            self.connection.settimeout(20)

        def log_message(self, *args):
            pass

        def authenticated(self):
            supplied = self.headers.get('X-Trid-Proxy', '')
            return (supplied.isascii() and hmac.compare_digest(supplied, secret)
                    and self.headers.get('X-Trid-Host') == hostname)

        def respond(self, status, value, mime='application/json; charset=utf-8', cookie=None):
            body = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            for key, val in {
                'Content-Type': mime, 'Content-Length': str(len(body)), 'Cache-Control': 'no-store',
                'X-Content-Type-Options': 'nosniff', 'X-Frame-Options': 'DENY',
                'Referrer-Policy': 'no-referrer', 'Connection': 'close',
                'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'; object-src 'none'; base-uri 'none'; form-action 'self'",
            }.items():
                self.send_header(key, val)
            if cookie:
                self.send_header('Set-Cookie', f'__Host-trid={cookie}; Path=/; Secure; HttpOnly; SameSite=Strict; Max-Age={TTL}')
            self.end_headers()
            self.wfile.write(body)
            self.close_connection = True

        def session(self, create=False):
            jar = SimpleCookie()
            try:
                jar.load(self.headers.get('Cookie', ''))
                sid = jar['__Host-trid'].value if '__Host-trid' in jar else ''
            except Exception:
                sid = ''
            with state.lock:
                state.prune()
                session = state.sessions.get(sid)
                if session is None and create and len(state.sessions) < MAX_SESSIONS:
                    sid = secrets.token_urlsafe(32)
                    session = Session()
                    state.sessions[sid] = session
                if session:
                    session.touched = time.monotonic()
                return sid, session

        def do_GET(self):
            if not self.authenticated():
                self.respond(403, {'error': '허용되지 않은 연결입니다.'}); return
            path = urlsplit(self.path).path
            if path == '/healthz':
                self.respond(200, {'ok': True, 'model': state.run, 'versions': list(state.versions)}); return
            if path == '/app.js':
                self.respond(200, (assets / 'app.js').read_bytes(), 'text/javascript; charset=utf-8'); return
            if path != '/':
                self.respond(404, {'error': '없는 페이지입니다.'}); return
            sid, session = self.session(create=True)
            if session is None:
                self.respond(503, {'error': '현재 테스트 인원이 많습니다. 잠시 후 다시 접속해 주세요.'}); return
            page = ((assets / 'index.html').read_text()
                    .replace('TOKEN_PLACEHOLDER', session.csrf)
                    .replace('MODEL_PLACEHOLDER', html.escape(state.run))
                    .replace('VERSION_OPTIONS_PLACEHOLDER', version_options)
                    .replace('VERSION_CARDS_PLACEHOLDER', version_cards)
                    .replace('WORKSPACE_SCOPE_PLACEHOLDER', workspace_scope))
            self.respond(200, page.encode(), 'text/html; charset=utf-8', cookie=sid)

        def do_POST(self):
            if not self.authenticated() or self.headers.get('Origin') != origin:
                self.respond(403, {'error': '허용되지 않은 요청입니다.'}); return
            sid, session = self.session()
            token = self.headers.get('X-Trid-Token', '')
            if session is None or not token.isascii() or not hmac.compare_digest(token, session.csrf):
                self.respond(403, {'error': '세션이 만료됐습니다. 페이지를 새로고침해 주세요.'}); return
            with state.lock:
                now = time.monotonic()
                session.calls = [t for t in session.calls if now - t < 60]
                if len(session.calls) >= 20:
                    self.respond(429, {'error': '요청이 많습니다. 1분 뒤 다시 시도해 주세요.'}); return
                session.calls.append(now)
            try:
                size = int(self.headers.get('Content-Length', '-1'))
                if self.headers.get('Transfer-Encoding') or not 0 <= size <= MAX_BODY:
                    self.respond(413, {'error': '입력은 2MB 이하로 넣어 주세요.'}); return
                body = self.rfile.read(size)
                if len(body) != size:
                    raise DocumentError('입력 전송이 완료되지 않았습니다.')
                path = urlsplit(self.path).path
                if not state.work.acquire(timeout=2):
                    self.respond(503, {'error': '다른 질문을 처리 중입니다. 잠시 후 다시 시도해 주세요.'}); return
                try:
                    if path in {'/api/load', '/api/sample', '/api/text'}:
                        if path == '/api/load':
                            doc = parse_pdf(body, unquote(self.headers.get('X-File-Name', 'document.pdf')), state.tok)
                        elif path == '/api/sample':
                            content = (ROOT / 'data/reading-demo-v3/travel-named.txt').read_text()
                            doc = Document('가상 출장 규정 · 샘플', [content], make_chunks([content], state.tok), ['시연용 가상 자료입니다. 실제 기관 규정이 아닙니다.'])
                        else:
                            request = json.loads(body)
                            content = request.get('text') if isinstance(request, dict) else None
                            if not isinstance(content, str) or not 10 <= len(content.strip()) <= 5000:
                                raise DocumentError('문서는 10~5,000자로 입력해 주세요.')
                            import unicodedata
                            content = unicodedata.normalize('NFC', content.strip())
                            doc = Document('직접 입력한 문서', [content], make_chunks([content], state.tok), [])
                        with state.lock:
                            session.document = doc
                            session.document_id = secrets.token_urlsafe(24)
                        result = {'document_id': session.document_id, 'name': doc.name, 'pages': doc.pages, 'warnings': doc.warnings}
                    elif path == '/api/pin':
                        if workspace is None:
                            raise DocumentError('공유 검색 자료가 준비되지 않았습니다.')
                        request = json.loads(body)
                        choice = request.get('choice') if isinstance(request, dict) else None
                        with state.lock:
                            if choice is None:
                                session.pinned_doc = None
                            elif isinstance(choice, int) and 1 <= choice <= len(session.offered_docs):
                                session.pinned_doc = session.offered_docs[choice - 1]
                            else:
                                raise DocumentError('직전 답변에 표시된 다른 후보 문서 번호를 골라 주세요.')
                        result = {'pinned': workspace.describe(session.pinned_doc) if session.pinned_doc else None}
                    elif path == '/api/clear':
                        with state.lock:
                            session.document = session.document_id = None
                        result = {'cleared': True}
                    elif path == '/api/ask':
                        request = json.loads(body)
                        if not isinstance(request, dict):
                            raise DocumentError('질문 형식을 확인해 주세요.')
                        workspace_mode = request.get('scope') == 'workspace'
                        if workspace_mode and workspace is None:
                            raise DocumentError('공유 검색 자료가 준비되지 않았습니다. 문서를 직접 넣어 주세요.')
                        if not workspace_mode and (session.document is None or request.get('document_id') != session.document_id):
                            self.respond(409, {'error': '이 브라우저에서 문서를 먼저 넣어 주세요.'}); return
                        question = request.get('question')
                        if not isinstance(question, str) or len(question) > 500:
                            raise DocumentError('질문은 500자 이하로 입력해 주세요.')
                        version_id = request.get('version', state.default)
                        if not isinstance(version_id, str) or version_id not in state.versions:
                            raise DocumentError('목록에 있는 모델 버전을 선택해 주세요.')
                        version = state.versions[version_id]
                        started = time.monotonic()
                        selected, others, selection = None, [], None
                        document = session.document
                        if workspace_mode:
                            selection = workspace.select(question, session.pinned_doc)
                            others = selection['others']
                        if selection and selection['status'] != 'answer':
                            result = workspace_reply(workspace, selection)
                        else:
                            if selection:
                                selected = selection['chunks']
                                document = Document(workspace.meta[selection['applied']]['title'], [],
                                                    [c for c, _ in selected], [])
                            if version['kind'] == 'extractive':
                                result = answer_document(document, question, version['model'], None, version['tok'],
                                                         version['device'], reader=read_chunk, selected=selected)
                                if any(item['prediction']['presence'] == 2 for item in result['checked']):
                                    result.update(status='review', answer='모델이 근거 충돌로 판단한 구간이 있습니다. /source로 원문을 확인해 주세요.')
                            elif version.get('answer_mode') == 'case':
                                result = answer_document_case(document, question, version['base'], version['family'],
                                                              version['context'], selected=selected)
                            else:
                                result = answer_document_chat(document, question, version['base'], version['family'],
                                                              version['context'], version['decoding'] == 'evidence_constrained',
                                                              selected=selected)
                            result.pop('checked', None)
                        if selection:
                            with state.lock:
                                session.offered_docs = [o['doc_id'] for o in others]
                            for c in result.get('candidates', []):
                                chunk = next(ch for ch, _ in selected if ch.text == c['text'])
                                c['source_label'] = ' · '.join(x for x in (
                                    workspace.meta[chunk.doc_id]['institution'], workspace.meta[chunk.doc_id]['title'],
                                    ' '.join(chunk.path[1:]) or None) if x)
                                c['page_character_span'] = [chunk.start + c['prediction']['source_character_span'][0],
                                                            chunk.start + c['prediction']['source_character_span'][1]]
                            applied = selection['applied']
                            result['workspace'] = {'applied': workspace.describe(applied) if applied else None,
                                                   'pinned': bool(session.pinned_doc), 'others': others,
                                                   'selection': selection['status'], 'institutions': selection['institutions'],
                                                   'title_matched': selection['title_matched'], 'scope': workspace.scope()}
                            result['total_chunks'] = len(workspace.index.passages)
                            if selection['status'] == 'answer':
                                how = ('질문에 적힌 문서명' if selection['title_matched'] else
                                       f"질문에 적힌 기관({', '.join(selection['institutions'])})" if selection['institutions'] else
                                       '관련도')
                                result['scope'] = (f'{how}으로 고른 문서 1건의 구간만 모델에 넣었습니다. 다른 기관·문서의 규정을 '
                                                   '섞지 않습니다. 검색되지 않은 자료까지 정보가 없다고 보장하지 않습니다.')
                        result['model'] = version['label']
                        result['version'] = version_id
                        result['elapsed_ms'] = round((time.monotonic() - started) * 1000)
                    else:
                        self.respond(404, {'error': '없는 기능입니다.'}); return
                finally:
                    state.work.release()
                self.respond(200, result, cookie=sid)
            except DocumentError as error:
                self.respond(422, {'error': str(error)})
            except (ValueError, UnicodeError):
                self.respond(422, {'error': '입력 형식을 확인해 주세요.'})
            except (ConnectionError, TimeoutError):
                self.close_connection = True
            except Exception:
                self.respond(500, {'error': '처리에 실패했습니다. 작은 문서로 다시 시도해 주세요.'})

    return BoundedServer(('127.0.0.1', port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--versions', type=Path, default=ROOT / 'deploy/shared-demo/versions.json',
                        help='JSON list of served versions; the first is the default and must be extractive')
    parser.add_argument('--secret-file', required=True)
    parser.add_argument('--port', type=int, default=51873)
    parser.add_argument('--workspace', type=Path, default=ROOT / 'runs/workspace-index/v1',
                        help='shared search index of approved public sources (scripts/build_workspace_index.py)')
    args = parser.parse_args()
    torch.set_num_threads(4)
    def stop_signal(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop_signal)
    state = None
    try:
        state = State(json.loads(args.versions.read_text()))
        state.workspace = Workspace(args.workspace) if (args.workspace / 'manifest.json').exists() else None
        server = make_server(state, Path(args.secret_file).read_text().strip(), args.port)
        stop = threading.Event()
        def prune():
            while not stop.wait(30):
                state.prune()
        threading.Thread(target=prune, daemon=True).start()
        print(f'trid temporary demo ready on 127.0.0.1:{server.server_port} versions={list(state.versions)}', flush=True)
        try:
            server.serve_forever()
        finally:
            stop.set()
            server.server_close()
            state.sessions.clear()
    except KeyboardInterrupt:
        pass
    finally:
        if state is not None:
            state.close()


if __name__ == '__main__':
    main()
