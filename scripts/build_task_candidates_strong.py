"""Strong-teacher stages for scripts/build_task_candidates.py (C-009): a >= 32B generator drafts finished C-002 items and
a different >= 32B model verifies them independently. select / check / status / export stay in build_task_candidates.

  draft   generator (Qwen3-32B, non-thinking, JSON schema) reads the passage(s) and returns one item or skips.
          It fixes the answer span first (verbatim), then writes a self-contained question that does not expose it.
          combine: one conclusion that needs a fact from passage A and a fact from passage B (stated in `why_both`).
          numeric_change: every number that depends on the edited one (repeats, totals, ratios, units) is edited
          together or the passage is skipped; the script applies the edits and keeps original/edited hashes.
          table: the raw table block with its full header hierarchy is shown and kept verbatim (C-002 raw_span); the
          question names the full row and column path.
  verify  verifier (gpt-oss-120b, reasoning) first answers from the built documents and question alone, then judges
          the candidate against the documents: target, condition, unit, completeness, other valid answers, logical
          support, question clarity; numeric_change: internal consistency of the edited passage; table: a unique cell;
          combine: both facts necessary. Any 'no' rejects, any 'unsure' or truncation quarantines.

Calls go through /v1/chat/completions; finish_reason and token usage are kept for every call.
"""
import hashlib
import json
import re
import urllib.request

ROLE = ('당신은 한국 공공기관 문서로 독해 학습 문제를 만드는 출제자입니다. 문서에 실제로 있는 내용만 사용합니다. '
        '답 필드는 문단의 연속된 구간을 한 글자도 바꾸지 않고 복사합니다(띄어쓰기·기호 포함). '
        '먼저 정답 구간과 그 범위를 확정한 뒤, 그 범위만 정확히 묻는 질문을 씁니다. 질문은 기관·사업·제도·행사·표 이름을 넣어 '
        '문단 없이도 무엇을 묻는지 알 수 있어야 하며, "이 사업", "해당 기관"처럼 지시 대상이 없는 표현과 답 노출을 피합니다. '
        '조건 없이 정답이 하나로 정해지지 않으면 질문에 필요한 조건(연도·대상·구분)을 넣습니다. '
        '좋은 문제를 만들 수 없으면 skip을 true로 하고 이유를 reason에 씁니다.')
TYPE_DEF = {
    'condition': ('조건: 어떤 혜택·의무·자격·절차·제재가 적용되기 위한 요건(대상·자격·기준·한도·기한 등)을 묻는 문제. '
                  '정답은 그 요건 구간만이다. 포부·소감, 근거 법령, 일시, 경위, 정의 문장 전체는 조건 문제가 아니다.'),
    'exception': ('예외: 문단에 원칙과 그 원칙이 적용되지 않는 경우(다만·제외·면제·예외·단서)가 함께 있을 때, 원칙이 적용되지 않는 대상·경우를 묻는 문제. '
                  '질문에 원칙을 밝히고, 정답은 예외 대상·경우 구간만이다(위반 여부·결과 서술 등 덧붙은 부분 제외). 문단에 없는 예외 전제를 만들면 안 된다.'),
    'explanation': ('설명: 문서가 명시한 결론(정책·조치의 목적·이유, 또는 법령해석의 결론)과 그 결론을 뒷받침하는 근거를 함께 설명하게 하는 문제. '
                    '근거는 결론과 다른 문장으로, 그것만 읽어도 왜 그런 결론이 나오는지 알 수 있는 사실·규정·논리다. '
                    '근거가 행사 내용·참석자·일정처럼 결론을 뒷받침하지 않거나, 질문이 결론과 반대되는 전제를 깔면 설명 문제가 아니다.'),
    'numeric_change': ('수치변화: 문단의 수치 하나를 바꾼 반사실 문단에서 바뀐 값을 묻는 문제. 편집된 문단 안에 원래 값이 남거나, '
                       '제목·요약줄·합계·비율·증감이 바뀐 값과 모순되면 안 된다.'),
    'combine': ('다중근거: 질문 하나에 대한 답 하나를 내려면 두 문단의 사실을 결합해야 하는 문제. 허용 형태는 둘이다. '
                '(가) 연결형: 질문이 문단 A의 사실로 대상을 특정하고(예: "만 39세 이하만 신청할 수 있는 사업"), 그 대상에 관한 값을 문단 B에서 묻는다. '
                '(나) 비교·계산형: A의 값과 B의 값을 비교하거나 합쳐야 답이 나온다(예: A의 목표 대비 B의 실적). '
                '"A는 무엇이고 B는 무엇인가", "목적과 방향은?"처럼 하위 질문 두 개로 나뉘는 질문, 서로 다른 사례 두 개의 나열은 다중근거가 아니다.'),
    'table': '표: 표 제목, 행 이름, 열 머리글 경로로 정해지는 셀 하나의 값을 묻는 문제.',
}
TASK = {
    'condition': f"유형 정의 — {TYPE_DEF['condition']} 이런 요건이 문단에 없으면 skip합니다.",
    'exception': f"유형 정의 — {TYPE_DEF['exception']} 원칙-예외 구조가 문단에 없으면 skip합니다.",
    'explanation': (f"유형 정의 — {TYPE_DEF['explanation']} 질문은 \"근거와 함께 설명해 주세요.\"로 끝냅니다. conclusion은 목적·이유 구간(…위해, …목적, …때문 등)만, "
                    'evidence는 conclusion과 다른 문장, condition은 문단에 적용 조건·대상·범위가 실제로 있을 때만 그 구간이고 없으면 빈 문자열입니다.'),
    'numeric_change': (f"유형 정의 — {TYPE_DEF['numeric_change']} edits는 [원문 구간, 바뀐 구간] 목록입니다. 첫 항목이 질문이 묻는 수치이고, "
                       '같은 수치가 문단 첫 부분의 제목·부제·요약줄이나 다른 문장에 다시 나오면 그 구간도 모두 함께 바꿉니다. 바꾼 수치에서 계산되는 '
                       '비율·합계·증감이 문단에 있으면 함께 고치고, 계산 관계가 복잡하거나 일관되게 바꿀 수 없으면 skip합니다. '
                       '각 원문 구간은 숫자 앞뒤 단어를 포함해 문단에서 한 번만 나오게 하고, 구간끼리 겹치지 않게 합니다. 날짜·연도·차수는 바꾸지 않습니다. '
                       'answer는 첫 항목의 바뀐 구간 안에 있는 바뀐 값입니다(원래 값이 아님).'),
    'combine': (f"유형 정의 — {TYPE_DEF['combine']} value_a는 문단 A의 구간, value_b는 문단 B의 구간, "
                'answer_sentence는 두 값을 함께 써서 질문에 답하는 한 문장, why_both는 두 사실을 어떻게 결합해 하나의 답이 되는지 한 문장으로 씁니다. '
                '질문에 "각각"을 쓰거나 "…와 …은 무엇인가"처럼 두 가지를 따로 묻지 않습니다. 그런 결합이 없으면 skip합니다.'),
    'table': (f"유형 정의 — {TYPE_DEF['table']} 아래 표 원문(머리글 계층 포함)에서 셀 하나를 묻습니다. 질문에는 표 제목(주제), 행 이름, "
              '열의 전체 머리글 경로(예: 2019년 > 수입)를 넣어 정답 셀이 하나로 정해지게 합니다. answer는 공백 없는 셀 값 하나 그대로입니다. '
              '머리글이 모호해 셀을 하나로 정할 수 없으면 skip합니다.'),
}
CHECK = ('check에는 먼저 고른 정답 구간이 유형 정의에 맞는지, 질문이 그 구간만을 하나의 정답으로 가리키는지, 문단에 없는 전제를 넣지 않았는지를 '
         '두세 문장으로 점검해 씁니다. 점검에서 문제가 있으면 다른 구간을 고르거나 skip합니다.')
FIELDS = {
    'condition': {'question': 'string', 'answer': 'string'},
    'exception': {'question': 'string', 'answer': 'string'},
    'explanation': {'question': 'string', 'conclusion': 'string', 'evidence': 'string', 'condition': 'string'},
    'numeric_change': {'question': 'string', 'answer': 'string',
                       'edits': {'type': 'array', 'items': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 2, 'maxItems': 2}}},
    'combine': {'question': 'string', 'value_a': 'string', 'value_b': 'string', 'answer_sentence': 'string', 'why_both': 'string'},
    'table': {'question': 'string', 'answer': 'string', 'row': 'string', 'column_path': 'string'},
}
ANSWER = ('다음 문서만 근거로 질문에 답하세요. 문서에 답이 없으면 "확인 불가"라고만 쓰세요. 답에 필요한 조건·대상·단위를 포함하고, '
          '정답이 여러 개면 모두 쓰세요. 근거가 된 문장을 짧게 함께 쓰세요.\n\n{context}\n\n질문: {question}')
JUDGE = ('문서, 질문, 답안 A(검토 대상), 답안 B(다른 검토자가 문서만 보고 먼저 쓴 답)가 있습니다. 문서를 기준으로 엄격하게 판정해 JSON으로 출력하세요. '
         '각 항목은 yes/no/unsure입니다.\n'
         'type_ok: 이 문제가 다음 유형 정의에 맞는다 — {type_def}\n'
         'supported: 문서가 A를 직접 뒷받침한다\n'
         'complete: A가 질문이 묻는 범위를 빠짐없이 답하고, 질문과 무관한 내용(덧붙은 다른 사실·결과 서술)을 담고 있지 않다\n'
         'condition_ok: 정답에 필요한 조건·대상·단위가 A에 빠지지 않았다\n'
         'target_ok: A가 질문이 묻는 대상의 종류(사람·기관·날짜·금액·조건·이유 등)와 같다\n'
         'unique: 문서 안에 A와 다른 정답이 없다(표라면 질문의 행·열 경로로 셀이 하나로 정해진다)\n'
         'question_ok: 질문이 지시 대상 없이도 분명하고, 답을 노출하지 않으며, 문서에 없는 전제를 깔지 않는다\n'
         'consistent: 문서 안의 숫자·제목·요약줄·합계·비율이 서로 모순되지 않는다{edits}\n'
         'agrees: A와 B가 같은 내용이다(표현 차이는 무시)\n'
         'span_tight: A가 질문이 묻는 부분만 담고, 결과·배경·다른 사실 등 묻지 않은 내용을 덧붙이지 않았다'
         '(다중근거의 "대상 특정 근거", 설명의 "근거"는 형식상 요구되는 부분이므로 군더더기로 보지 않는다)\n'
         '{extra}\n\n{context}\n\n질문: {question}\n답안 A: {candidate}\n답안 B: {independent}')
EXTRA = {
    'combine': ('both_needed: A의 "답"은 질문에 대한 답이고 "대상 특정 근거"는 질문이 가리키는 대상을 정하는 다른 문단의 사실이다. '
                '그 근거 없이는 질문의 대상을 정할 수 없고, 답은 다른 문단에서만 나온다(하나는 대상을 특정하고 다른 하나가 값을 주거나, 두 값을 비교·계산). '
                '질문이 조문 번호만으로 대상을 가리켜 근거 문단이 필요 없거나, 서로 무관한 두 질문·두 사례의 나열이면 no'),
    'explanation': ('both_needed: A의 결론이 문서에 명시되어 있고(목적·이유 또는 해석 결론), 결론과 다른 근거 문장만 읽어도 '
                    '그 결론이 왜 나오는지 알 수 있다. 근거가 결론을 뒷받침하지 않거나 질문의 전제가 결론과 반대이면 no'),
}
JUDGE_KEYS = ('type_ok', 'supported', 'complete', 'condition_ok', 'target_ok', 'unique', 'question_ok', 'consistent', 'agrees',
              'span_tight')
EDIT_NOTE = (' (이 문서는 반사실 편집본이다. 편집: {edits}. 편집된 값이 같은 문장·목록의 비율(%)·합계·전체 개수(예: 전국 17개 시도)·증감률과 '
             '산술적으로 맞는지 실제로 계산해 확인하고, 편집된 값과 원래 값이 문서 안에 함께 남아 있거나 제목·요약줄·합계·비율·증감이 '
             '편집된 값과 맞지 않으면 no)')


def chat(url, messages, max_tokens, schema=None, template=None):
    payload = dict(messages=messages, max_tokens=max_tokens, temperature=0, seed=42)
    if schema:
        payload['response_format'] = {'type': 'json_schema', 'json_schema': {'name': 'out', 'schema': schema}}
    if template:
        payload['chat_template_kwargs'] = template
    req = urllib.request.Request(url + '/v1/chat/completions', json.dumps(payload).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.load(r)
    choice = out['choices'][0]
    msg = choice['message']
    # completion_tokens counts the hidden reasoning and the visible answer together; their character lengths are kept
    # separately so a 'length' stop can be attributed to the reasoning or to the answer.
    meta = dict(finish_reason=choice.get('finish_reason'), prompt_tokens=out.get('usage', {}).get('prompt_tokens'),
                completion_tokens=out.get('usage', {}).get('completion_tokens'), max_tokens=max_tokens,
                reasoning_chars=len(msg.get('reasoning_content') or ''), answer_chars=len(msg.get('content') or ''))
    return (msg.get('content') or '').strip(), meta


def draft_schema(kind, ids):
    props = {'check': {'type': 'string'}, 'skip': {'type': 'boolean'}, 'reason': {'type': 'string'}}
    picks = ['passage_a', 'passage_b'] if kind == 'combine' else [] if kind == 'table' else ['passage']
    for k in picks:
        props[k] = {'type': 'integer', 'enum': ids}
    for k, v in FIELDS[kind].items():
        props[k] = v if isinstance(v, dict) else {'type': v}
    return {'type': 'object', 'required': ['check', 'skip', 'reason', *picks, *FIELDS[kind]], 'properties': props}


def changed_values(edits):
    """Number+unit tokens (e.g. '20명', '8.2%') of the original spans that the edits replaced, plus the bare number when it
    is distinctive (3+ digits or a decimal): tables repeat values without their unit ('458개' in text, '458' in a table)."""
    out = []
    for old, new in edits:
        for m in re.finditer(r'\d[\d,]*(?:\.\d+)?', old):
            unit = old[m.end():m.end() + 1].strip()
            bare = m.group(0)
            for key in (bare + (unit if unit and not unit.isdigit() else ''), bare):
                distinctive = key != bare or len(bare.replace(',', '')) >= 3 or '.' in bare
                if distinctive and not value_in(key, new) and key not in out:
                    out.append(key)
    return out


def value_in(key, text):
    return re.search(r'(?<![\d.,])' + re.escape(key) + (r'(?![\d])' if key[-1].isdigit() else ''), text) is not None


def outside_text(passage, title, doc_text):
    """Title plus every part of the document that is not in this passage. Passages (build_teacher_qa.passages) join
    whitespace-normalised blank-line paragraphs and may cut the last one, so removal is per paragraph: a paragraph
    contained in the passage is dropped, a paragraph cut at the passage end keeps only its part after the cut."""
    cut = re.sub(r'\s+', ' ', passage).strip()
    rest, used = [], 0
    for para in re.split(r'\n\s*\n', doc_text):
        p = re.sub(r'\s+', ' ', para).strip()
        if p and p in cut:
            used += len(p)
            continue
        k = next((k for k in range(len(p), 24, -1) if cut.endswith(p[:k])), 0)
        used += k
        rest.append(p[k:])
    if used < .9 * len(cut):
        raise ValueError('passage not found in its document')
    return title + '\n' + '\n'.join(rest)


def repeated_values(passage, title, doc_text):
    """Number+unit tokens of the passage that also appear in the title or elsewhere in the document; editing one of them
    would contradict text outside the passage, so the generator is told not to touch them and to_item rejects it."""
    outside = outside_text(passage, title, doc_text)
    keys = []
    for m in re.finditer(r'\d[\d,]*(?:\.\d+)?', passage):
        unit = passage[m.end():m.end() + 1].strip()
        bare = m.group(0)
        key = bare + (unit if unit and not unit.isdigit() else '')
        distinctive = len(bare.replace(',', '')) >= 3 or '.' in bare
        if key not in keys and (value_in(key, outside) or (distinctive and value_in(bare, outside))):
            keys.append(key)
    return keys


UNITS = sorted('% 퍼센트 명 건 개소 개월 개 곳 원 억 조 만 천 톤 kg ㎡ ha km 대 척 회 배 가구 세대 종 편 권 마리 그루 점 석 병상 시간'.split(), key=len, reverse=True)
QUANTITY = re.compile(r'(?<![\d.,])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:' + '|'.join(map(re.escape, UNITS)) + ')')   # longest unit first: 4개월 is not 4개


def edit_target(cand):
    """Strong numeric_change: the value to edit is fixed here, not by the generator. A quantity (number + unit) of the passage
    that does not appear in the title or elsewhere in the document, chosen by hash; None when no such value exists."""
    rep = cand.get('repeated') or {}
    for i, p in cand['passages'].items():
        options = []
        for m in QUANTITY.finditer(p):
            key = m.group(0).replace(' ', '')
            num = re.match(r'\d[\d,]*(?:\.\d+)?', key).group(0)
            unit = key[len(num):len(num) + 1]
            distinctive = len(num.replace(',', '')) >= 3 or '.' in num
            repeats_bare = distinctive and len(re.findall(r'(?<![\d.,])' + re.escape(num) + r'(?![\d])', p)) > 1   # e.g. the same value in a table row
            if num + unit in rep.get(i, []) or p.count(m.group(0)) != 1 or repeats_bare:
                continue
            options.append((m.start(), m.group(0)))
        if options:
            at, value = options[int(hashlib.sha256((cand['cid'] + i).encode()).hexdigest(), 16) % len(options)]
            return int(i), value, p[max(0, at - 40):at + len(value) + 40]
    return None


def guard_note(cand):
    target = edit_target(cand)
    rep = (cand.get('repeated') or {}).get(str(target[0]), []) if target else []
    return (f'\n\n편집 대상: 문단 {target[0]}의 "{target[1]}" (주변: …{target[2]}…). passage는 {target[0]}로 하고, edits의 첫 항목은 이 값을 포함한 구간입니다. '
            f'편집 금지 수치(제목이나 문서의 다른 곳에도 나옴): {", ".join(rep) or "없음"}') if target else ''


def parse_reply(out, schema):
    """JSON object from a thinking-mode reply (no grammar constraint); missing keys or a wrong type are a failure."""
    text = re.sub(r'^```(?:json)?|```$', '', out.strip()).strip()
    try:
        reply = json.loads(text[text.index('{'):text.rindex('}') + 1])
    except ValueError as exc:
        raise ValueError(f'JSON 형식 오류: {exc}') from exc
    missing = [k for k in schema['required'] if k not in reply]
    if missing and not reply.get('skip'):
        raise ValueError(f'JSON에 빠진 키: {missing}')
    return reply


RECORD_TYPES = ('condition', 'exception', 'explanation', 'combine')
RECORD_TASK = {
    'condition': ('아래 조문 문단은 "{condition}"일 때 "{effect}"라고 정합니다. 이 효과(누가 무엇을 하거나 할 수 있는지)를 질문에 밝히고, '
                  '그 효과가 적용되는 경우(조건)를 묻는 질문 하나를 쓰세요. 질문에 법령·규정 이름을 넣고, 조건 내용은 질문에 쓰지 마세요. '
                  '정답은 조건 구간으로 이미 정해져 있으므로 질문만 씁니다.'),
    'exception': ('아래 조문 문단의 원칙은 "{rule}"이고, 단서(예외)는 "{exception}"입니다. 원칙을 질문에 밝히고, 그 원칙에 대해 단서가 정한 '
                  '예외가 무엇인지 묻는 질문 하나를 쓰세요. 질문에 법령·규정 이름을 넣고, 예외 내용은 질문에 쓰지 마세요. 정답은 단서 구간으로 이미 정해져 있습니다.'),
    'explanation': ('아래는 법령해석례의 회답(결론)과 그 근거가 된 이유 문단입니다. 해석 대상(법령 조항과 쟁점)을 밝히고 그것이 어떻게 해석되는지 '
                    '"근거와 함께 설명해 주세요."로 끝나는 질문 하나를 쓰세요. 결론 내용은 질문에 쓰지 마세요. 결론과 근거 구간은 이미 정해져 있습니다.'),
    'combine': ('문단 0은 조문 A, 문단 1은 A를 "{reference}"로 가리키는 조문 B입니다. 조문 번호 대신 A가 정한 내용을 풀어 써서 질문의 대상을 특정하고, '
                '그 대상에 대해 B가 정한 내용을 묻는 질문 하나를 쓰세요. 그래서 A를 읽어야 대상을 알 수 있고 B를 읽어야 답할 수 있어야 합니다. '
                'value_a는 질문의 대상을 특정하는 A의 구간, value_b는 질문의 답이 되는 B의 구간이며 둘 다 원문 그대로 복사합니다. '
                'why_both는 두 조문이 왜 모두 필요한지 한 문장으로 씁니다.'),
}
RECORD_FIELDS = {'condition': ['question'], 'exception': ['question'], 'explanation': ['question'],
                 'combine': ['question', 'value_a', 'value_b', 'why_both']}


def record_prompt(cand):
    kind, rec = cand['type'], cand['record']
    body = '\n\n'.join(f'문단 {i}:\n{p}' for i, p in cand['passages'].items())
    schema = {'type': 'object', 'required': ['check', 'skip', 'reason', *RECORD_FIELDS[kind]],
              'properties': {'check': {'type': 'string'}, 'skip': {'type': 'boolean'}, 'reason': {'type': 'string'},
                             **{k: {'type': 'string'} for k in RECORD_FIELDS[kind]}}}
    task = RECORD_TASK[kind].format(**{k: v for k, v in rec.items() if isinstance(v, str)})
    return f'{task}\n\n{body}', schema


def record_item(cand, d, iid, q):
    """C-002 item whose answer spans come from the evidence record (written before generation); only the question and,
    for combine, the two verbatim spans inside the fixed passages come from the generator."""
    kind, rec, doc, ps = cand['type'], cand['record'], cand['doc'], cand['passages']
    if kind == 'condition':
        return dict(id=iid, type=kind, doc=doc, passage=0, question=q, answer=exact(rec['condition'], ps['0']), note=f"effect: {rec['effect']}")
    if kind == 'exception':
        return dict(id=iid, type=kind, doc=doc, passage=0, question=q, answer=exact(rec['exception'], ps['0']), note=f"rule: {rec['rule']}")
    if kind == 'explanation':
        return dict(id=iid, type=kind, doc=doc, passage=0, question=q, conclusion=exact(rec['conclusion'], ps['0']),
                    evidence=exact(rec['ground'], ps['0']), condition=None)
    a, b = exact(d['value_a'], ps['0']), exact(d['value_b'], ps['1'])
    return dict(id=iid, type=kind, question=q, parts=[[doc, 0, a], [doc, 1, b]],
                note=f"relation: {rec['citing_article']} cites {rec['cited_article']} ({rec['reference']}); why both: {d['why_both']}")

def draft_one(url, cand, retries=2):
    """Draft with the generator's thinking mode, then validate locally with to_item; on a failure the generator gets
    the exact reason and rewrites (at most `retries` times). Every attempt's call metadata and failure reason are kept.
    A truncated reply (finish_reason != stop) ends the attempts and is recorded as such."""
    kind = cand['type']
    ids = [int(i) for i in cand.get('passages', {})]
    if cand.get('record') and kind in RECORD_TYPES:
        prompt, schema = record_prompt(cand)
        messages = [{'role': 'system', 'content': ROLE}, {'role': 'user', 'content': (
            f'{prompt}\n\ncheck에는 질문이 정해진 정답 구간 하나만을 가리키는지, 정답을 노출하지 않는지 점검해 씁니다. 그런 질문을 쓸 수 없으면 skip합니다.\n\n'
            f"출력: 다음 키를 모두 가진 JSON 객체 하나만 출력합니다(코드 블록 없이). {json.dumps(schema['properties'], ensure_ascii=False)}")}]
    else:
        if kind == 'table':
            body = f"표 원문:\n{cand['table']['raw_block']}"
        else:
            body = '\n\n'.join(f'문단 {i}:\n{p}' for i, p in cand['passages'].items())
            body += ('\n\n서로 다른 두 문단을 passage_a, passage_b로 고르세요.' if kind == 'combine'
                     else '\n\n문제를 만들 문단 하나를 passage로 고르세요.')
        if kind == 'numeric_change':
            if not edit_target(cand):
                return dict(skip=True, reason='no quantity that appears only in this passage (unsafe to edit)'), None
            body += guard_note(cand)
        if kind == 'combine' and len(ids) < 2:
            return dict(skip=True, reason='fewer than two passages'), None
        schema = draft_schema(kind, ids)
        shape = json.dumps(schema['properties'], ensure_ascii=False)
        messages = [{'role': 'system', 'content': ROLE},
                    {'role': 'user', 'content': f'{TASK[kind]} {CHECK}\n\n{body}\n\n출력: 다음 키를 모두 가진 JSON 객체 하나만 출력합니다(코드 블록 없이). {shape}'}]
    attempts = []
    reply = None
    for _ in range(retries + 1):
        out, meta = chat(url, messages, 6000, None, {'enable_thinking': True})
        if meta['finish_reason'] != 'stop':
            attempts.append(dict(meta, failure='generator reply truncated'))
            raise ValueError(f"generator reply truncated: finish_reason={meta['finish_reason']}, completion_tokens={meta['completion_tokens']} "
                             f"(reasoning {meta['reasoning_chars']} chars, answer {meta['answer_chars']} chars), attempt {len(attempts)}")
        try:
            reply = parse_reply(out, schema)
        except ValueError as exc:
            attempts.append(dict(meta, failure=str(exc)))
            messages += [{'role': 'assistant', 'content': out}, {'role': 'user', 'content': f'{exc}. 같은 키의 JSON 객체 하나만 다시 출력하세요.'}]
            continue
        if reply.get('skip'):
            attempts.append(dict(meta, failure=None))
            break
        try:
            to_item(cand, reply)
            attempts.append(dict(meta, failure=None))
            break
        except (ValueError, KeyError) as exc:
            attempts.append(dict(meta, failure=str(exc)))
            messages += [{'role': 'assistant', 'content': out},
                         {'role': 'user', 'content': f'검증 실패: {exc}\n원문을 다시 확인해 같은 JSON 형식으로 고쳐 쓰세요. '
                                                     '모든 구간은 원문에서 한 글자도 바꾸지 않고 연속으로 복사해야 합니다. 고칠 수 없으면 skip하세요.'}]
    if reply is None:
        raise ValueError(f"no valid JSON reply after {len(attempts)} attempts: {attempts[-1]['failure']}")
    return reply, dict(attempts[-1], attempts=attempts)


def exact(span, text):
    """Verbatim span or the same characters with different whitespace; nothing looser."""
    span = span.strip()
    if span and span in text:
        return span
    m = re.search(r'\s*'.join(re.escape(ch) for ch in re.sub(r'\s+', '', span)), text) if span else None
    if not m:
        raise ValueError(f'원문에 그대로 있는 구간이 아님: {span[:60]!r}')
    return m.group(0)


def to_item(cand, d):
    kind, doc = cand['type'], cand['doc']
    iid = f'{kind}-{doc}'
    q = d['question'].strip()
    if cand.get('record') and kind in RECORD_TYPES:
        return record_item(cand, d, iid, q), None
    if kind in ('condition', 'exception'):
        text = cand['passages'][str(d['passage'])]
        return dict(id=iid, type=kind, doc=doc, passage=d['passage'], question=q, answer=exact(d['answer'], text)), None
    if kind == 'explanation':
        text = cand['passages'][str(d['passage'])]
        cond = d['condition'].strip()
        conclusion, evidence = exact(d['conclusion'], text), exact(d['evidence'], text)
        if re.sub(r'\s+', '', conclusion) in re.sub(r'\s+', '', evidence) or re.sub(r'\s+', '', evidence) in re.sub(r'\s+', '', conclusion):
            raise ValueError('evidence가 conclusion과 같은 문장임(결론과 다른, 결론을 뒷받침하는 문장을 고를 것)')
        return dict(id=iid, type=kind, doc=doc, passage=d['passage'], question=q, conclusion=conclusion,
                    evidence=evidence, condition=exact(cond, text) if cond else None), None
    if kind == 'numeric_change':
        text = cand['passages'][str(d['passage'])]
        edits, taken = [], []
        for old, new in d['edits']:
            old, new = exact(old, text), new.strip()
            if old == new:
                continue   # no-op edit
            if text.count(old) != 1:
                raise ValueError(f'원문 구간이 문단에 여러 번 나옴: {old!r} (앞뒤 단어를 포함해 한 번만 나오게)')
            if not re.search(r'\d', new):
                raise ValueError(f'바뀐 구간에 숫자가 없음: {new!r}')
            at = text.find(old)
            if any(at < e and s < at + len(old) for s, e in taken):
                raise ValueError(f'편집 구간이 서로 겹침: {old!r}')
            taken.append((at, at + len(old)))
            edits.append([old, new])
        if not edits:
            raise ValueError('바뀐 수치가 없음')
        edited = text
        for old, new in edits:
            edited = edited.replace(old, new)
        left = [k for k in changed_values(edits) if value_in(k, edited)]
        if left:
            raise ValueError(f'원래 값 {left}이 편집된 문단의 다른 곳(제목·요약줄·다른 문장)에 남아 있음 — 그 구간도 함께 바꾸거나 skip')
        banned = [k for k in changed_values(edits) if k in (cand.get('repeated') or {}).get(str(d['passage']), [])]
        if banned:
            raise ValueError(f'편집 금지 수치 {banned}를 바꿈(제목이나 문서의 다른 곳에도 나옴) — 다른 수치를 고르거나 skip')
        target = edit_target(cand)
        if target and (d['passage'] != target[0] or target[1] not in edits[0][0]):
            raise ValueError(f'edits의 첫 항목이 지정된 편집 대상 "{target[1]}"(문단 {target[0]})을 포함하지 않음')
        answer = d['answer'].strip()
        if not answer or answer not in edits[0][1]:
            raise ValueError(f'answer {answer!r}가 첫 편집의 바뀐 구간 {edits[0][1]!r} 안에 없음')
        return dict(id=iid, type=kind, doc=doc, passage=d['passage'], edits=edits, question=q,
                    answer=exact(answer, edited), note=f"edits: {json.dumps(edits, ensure_ascii=False)}"), None
    if kind == 'combine':
        if d['passage_a'] == d['passage_b']:
            raise ValueError('passage_a와 passage_b가 같은 문단임(서로 다른 문단에서 하나씩)')
        if '각각' in q:
            raise ValueError('질문이 "각각"으로 두 가지를 따로 묻고 있음 — 두 사실을 결합해야 답 하나가 나오는 질문으로 다시 쓰거나 skip')
        a = exact(d['value_a'], cand['passages'][str(d['passage_a'])])
        b = exact(d['value_b'], cand['passages'][str(d['passage_b'])])
        return dict(id=iid, type=kind, question=q, parts=[[doc, d['passage_a'], a], [doc, d['passage_b'], b]],
                    note=f"answer: {d.get('answer_sentence', '')}; why both: {d['why_both']}"), None
    t = cand['table']
    block = t['raw_block']
    answer = d['answer'].strip()
    if not answer or re.search(r'\s', answer) or not re.search(r'(^|\s)' + re.escape(answer) + r'(\s|$)', block):
        raise ValueError(f'answer {answer!r}가 표 원문의 셀 값 하나(공백 없는 값 하나)와 그대로 일치하지 않음')
    key = f'tab-{doc}'
    table = dict(doc=doc, raw_span=[t['raw_start'], t['raw_end']],
                 note='Raw table block copied verbatim with its full header hierarchy (C-002 raw_span).')
    return dict(id=iid, type='table', table=key, question=q, answer=answer,
                note=f"row: {d['row']}; column path: {d['column_path']}"), (key, table)


def candidate_text(b):
    item, kind = b['item'], b['type']
    if kind == 'explanation':
        return f"결론: {item['conclusion']} / 근거: {item['evidence']}" + (f" / 적용 조건: {item['condition']}" if item['condition'] else '')
    if kind == 'combine':
        return f"답: {item['parts'][1][2]} (대상 특정 근거: {item['parts'][0][2]})"
    return item['answer']


def verify_one(url, b):
    body, question = b['row']['prompt'].rsplit('\n질문: ', 1)
    context = body[body.find('[D1]'):]
    if b.get('doc_title'):   # the title is evidence for consistency (e.g. a headline repeating an edited number)
        context = f"(정답 문서의 제목) {b['doc_title']}\n\n{context}"
    calls = {}
    independent, calls['answer'] = chat(url, [{'role': 'user', 'content': ANSWER.format(context=context, question=question)}],
                                        2500, None, {'reasoning_effort': 'medium'})
    keys = JUDGE_KEYS + (('both_needed',) if b['type'] in EXTRA else ())
    schema = {'type': 'object', 'required': [*keys, 'reason'],
              'properties': {**{k: {'type': 'string', 'enum': ['yes', 'no', 'unsure']} for k in keys}, 'reason': {'type': 'string'}}}
    edits = b['item'].get('edits')
    note = EDIT_NOTE.format(edits=json.dumps(edits, ensure_ascii=False)) if edits else ''
    raw, calls['judge'] = chat(url, [{'role': 'user', 'content': JUDGE.format(
        type_def=TYPE_DEF[b['type']], edits=note, extra=EXTRA.get(b['type'], ''), context=context, question=question,
        candidate=candidate_text(b), independent=independent)}], 3000, schema, {'reasoning_effort': 'medium'})
    judge = json.loads(raw)
    failed = [k for k in keys if judge.get(k) == 'no']
    unsure = [k for k in keys if judge.get(k) not in ('yes', 'no')]
    for name, c in calls.items():
        if c['finish_reason'] != 'stop':
            unsure.append(f'{name}_truncated')
    state = 'semantic_reject' if failed else 'quarantine' if unsure else 'accepted'
    return dict(independent=independent, judge=judge, failed=failed, unsure=unsure, state=state, calls=calls,
                candidate=candidate_text(b))
