"""한국어 사전학습 문서의 품질 판정(assess)과 PII 치환(redact).

임계값은 FineWeb-2의 한국어 설정을 그대로 옮겼다.
  https://raw.githubusercontent.com/huggingface/fineweb-2/main/configs/kor_Hang.yml
  dup_line_frac 0.374; top_n_grams (2,0.282)(3,0.228)(4,0.185);
  dup_n_grams (5,0.201)(6,0.184)(7,0.168)(8,0.154)(9,0.14)(10,0.127);
  line_punct_thr 0.062; new_line_ratio 0.196; min/max_avg_word_length 1/6;
  max_non_alpha_words_ratio 0.791; stopwords 21개; language_score 0.842(GlotLID).
필터 로직은 datatrove(Apache-2.0)의 GopherRepetitionFilter, FineWebQualityFilter, GopherQualityFilter,
C4QualityFilter를 따르고, fineweb-2-pipeline.py가 바꾼 전역값(dup_para·dup_line_char 비활성,
char_duplicates_ratio 0.1, short_line 비활성, min_stop_words 2, min/max_doc_words 50/100000)을 반영했다.
  https://raw.githubusercontent.com/huggingface/fineweb-2/main/fineweb-2-pipeline.py
단어 분리는 datatrove KiwiTokenizer와 같이 kiwipiepy(Apache-2.0) 형태소 경계의 표면형을 쓴다. 축약형('관한'→관하+ㄴ)은
같은 구간이 두 번 나오는데(['관한','한']) datatrove도 동일하므로 임계값과 맞는다.
  https://raw.githubusercontent.com/huggingface/datatrove/main/src/datatrove/utils/word_tokenizers.py

자체 추가분(FineWeb-2에 없음):
- language_score는 GlotLID 모델이 필요하므로 글자 비율로 대체한다. 글자(isalpha) 중 한글 ≥ HANGUL_MIN,
  한자 ≤ HANJA_MAX, 로마자 ≤ LATIN_MAX. 값은 법령·보도자료 분포를 보고 정했다(법령의 한자 병기 허용).
- C4 줄 필터(FineWeb-2는 다국어에서 성능이 떨어져 제외)는 한국어 개조식 문장이 많아 "종결 부호 없는 줄
  제거"는 쓰지 않고, javascript 안내·짧은 사이트 메뉴 줄·1,000자 초과 단어 줄만 지운다. lorem ipsum은 문서 제거.

PII는 ko-pii(MIT, https://github.com/Marker-Inc-Korea/ko-pii)의 탐지 결과를 고정 토큰으로 바꾼다.
구조적 식별자(주민·외국인등록번호, 전화, 이메일, 계좌, 카드, 여권, 운전면허)만 켠다; 이름·주소는 오탐이 많다.
더 긴 하이픈 숫자열(사업코드 '061-1300-1334-500')에 걸친 탐지는 오탐으로 보고 건너뛴다.
"""
import re
from collections import Counter
from dataclasses import dataclass, replace

STOPWORDS = frozenset("의 다 이 는 에 을 한 년 하 은 를 했 고 에서 가 있 으로 인 월 로 되".split())


@dataclass(frozen=True)
class Thresholds:
    """기본값 = FineWeb-2 kor_Hang.yml + fineweb-2-pipeline.py 전역값. 글자 비율 셋은 자체 값."""
    dup_line_frac: float = 0.374
    top_n_grams: tuple = ((2, 0.282), (3, 0.228), (4, 0.185))
    dup_n_grams: tuple = ((5, 0.201), (6, 0.184), (7, 0.168), (8, 0.154), (9, 0.14), (10, 0.127))
    line_punct_thr: float = 0.062
    char_dup_ratio: float = 0.1
    new_line_ratio: float = 0.196
    min_words: int = 50
    max_words: int = 100_000
    min_avg_word_len: float = 1
    max_avg_word_len: float = 6
    max_symbol_word_ratio: float = 0.1
    max_bullet_lines: float = 0.9
    max_ellipsis_lines: float = 0.3
    min_alpha_words: float = 0.791
    min_stopwords: int = 2
    hangul_min: float = 0.6
    hanja_max: float = 0.3
    latin_max: float = 0.3


FINEWEB2 = Thresholds()
# 법령·보도자료용. 파일럿 640+137건 측정에서 FineWeb-2 값은 dup_n_grams로 법률 73%·법령해석례 92%를, 비알파벳
# 어절 비율로 보도자료 26%를 지웠다(조문 상투구와 숫자·기호 때문; 수동 검토상 정상 문서). 그 둘만 완화한다:
# dup_n_grams는 2배(법률 p95 0.348 기준), min_alpha_words는 0.7(보도자료 p5 0.695; 표 위주 문서만 남아 제거됨).
# max_words 100k(웹 기준)는 공직선거법·관세법·도로교통법 시행규칙(20만~30만 자)을 지우므로 1M으로 올린다.
PUBLIC_DOCS = replace(FINEWEB2, dup_n_grams=tuple((n, round(f * 2, 3)) for n, f in FINEWEB2.dup_n_grams),
                      min_alpha_words=0.7, max_words=1_000_000)
PROFILES = {"fineweb2": FINEWEB2, "public-docs": PUBLIC_DOCS}

# datatrove.utils.text.PUNCTUATION + TERMINAL_PUNCTUATION 중 한국어 텍스트에 나오는 것
PUNCTUATION = set("!/—”:％１〈&(、━\\【#%「」，】；+^]~“《„';’{|∶´[=-`*．（–？！：$～«〉,><》)?）。…@_.\"}►»｡") | {
    chr(c) for a, b in ((0, 9), (11, 13), (13, 32), (127, 160)) for c in range(a, b)}
TERMINAL = tuple(".?!。！．？｡")
BIG_NUMBER = re.compile(r"[0-9,]{20,}")  # datatrove KiwiTokenizer.preprocess
LINE = re.compile(r"\n+")
JAVASCRIPT = re.compile(r"javascript|자바스크립트", re.I)
SITE_MENU = re.compile(r"개인정보\s*처리\s*방침|이용\s*약관|쿠키|로그인|회원\s*가입|사이트\s*맵|all rights reserved|copyright", re.I)

PII_LABELS = {"RRN": "[주민번호]", "FRN": "[외국인번호]", "PHONE": "[전화]", "EMAIL": "[이메일]", "ACCOUNT": "[계좌]",
              "CARD": "[카드]", "PASSPORT": "[여권]", "DRIVER_LICENSE": "[운전면허]"}
# ko-pii 호출 전 선별: 위 라벨 중 하나라도 잡힐 수 있는 문자열(6자리 이상 숫자, 전화형 구분자, @, 여권 영문+8자리)
PII_HINT = re.compile(r"\d{6}|\d{2,4}[-.)－ ]{1,2}\d{3,4}[-.－ ]{1,2}\d{4}|@|[A-Z]\d{8}")
# ko-pii는 공백·줄바꿈을 접어 매칭하므로 표를 펼친 숫자열("0 1 396 136", "2010 2011 2012 2013")을 전화·카드·주민번호로
# 잡는다(3,000건 샘플에서 PHONE 238건 중 대부분). 원문 구간이 라벨의 표기 형식과 정확히 맞고, 앞뒤에 숫자 묶음이
# 더 이어지지 않을 때만 치환한다. 형식: 주민등록번호 6-7자리(주민등록법 시행규칙 별표), 전화 02/0NN-NNN(N)-NNNN(전기통신번호
# 관리세칙), 여권 M+8자리 또는 M+3자리+영문+4자리(2020 차세대 여권), 운전면허 지역2-연도2-일련6-검증2(도로교통법 시행규칙).
SHAPES = {"RRN": re.compile(r"\d{6}[- ]?\d{7}"), "FRN": re.compile(r"\d{6}[- ]?\d{7}"),
          "PHONE": re.compile(r"\(?(\+?82[-. ]?\d{1,2}|0\d{1,2})\)?[-.)－ ]{0,2}\d{3,4}[-.－ ]{0,2}\d{4}"),
          "EMAIL": re.compile(r"[^\s@]+@[^\s@]+\.[^\s@]+"),
          "ACCOUNT": re.compile(r"\d[\d-]{8,20}\d"), "CARD": re.compile(r"\d{4}([- ]?\d{4}){3}|\d{4}[- ]?\d{6}[- ]?\d{5}"),
          "PASSPORT": re.compile(r"[A-Z]\d{3}[A-Z0-9]\d{4}"), "DRIVER_LICENSE": re.compile(r"(\d{2}|[가-힣]{2})-?\d{2}-\d{6}-\d{2}")}
# 표 숫자열("2010 2011 2012 2013")이나 코드('DATE0703221230', '154.16341649')에 붙은 구간 제외; 'T.055-…'는 허용.
# \Z(문자열 끝)를 쓴다: '$'는 끝의 줄바꿈 앞에서도 맞아서 줄 끝 글자("CENTER\n010-…", "7630\n010-…")를 붙은 것으로
# 오판해 줄 단위 연락처가 남았다(B-004, FineWeb clean-v1 표본).
GLUED_AFTER = re.compile(r"[-－ .]?\d|[A-Za-z]")
GLUED_BEFORE = re.compile(r"\d[-－ .]?\Z|[A-Za-z]\Z")

_kiwi = None


def kiwi():
    global _kiwi
    if _kiwi is None:
        from kiwipiepy import Kiwi
        _kiwi = Kiwi(num_workers=4)
    return _kiwi


def words_batch(texts):
    """Kiwi 형태소 경계의 표면형 어절 목록; 문서 여러 개를 한 번에 넘기면 멀티스레드로 돈다."""
    texts = [BIG_NUMBER.sub("", t) for t in texts]
    return [[w for w in (t[k.start:k.end].strip() for k in toks) if w]
            for t, toks in zip(texts, kiwi().tokenize(texts))]


def clean_lines(text):
    """C4식 줄 필터(한국어 조정). (정리된 텍스트, 지운 줄 수)."""
    kept, dropped = [], 0
    for line in text.split("\n"):
        s = line.strip()
        if s and (JAVASCRIPT.search(s) or (len(s) <= 40 and SITE_MENU.search(s))
                  or any(len(w) > 1000 for w in s.split())):
            dropped += 1
            continue
        kept.append(line)
    return "\n".join(kept).strip() if dropped else text, dropped


def duplicates(items):
    seen, n, chars = set(), 0, 0
    for x in items:
        if x in seen:
            n += 1
            chars += len(x)
        seen.add(x)
    return n, chars


def dup_ngram_chars(words, n):
    seen, chars, i = set(), 0, 0
    while i < len(words) - n + 1:
        g = "".join(words[i:i + n])
        if g in seen:
            chars += len(g)
            i += n
        else:
            seen.add(g)
            i += 1
    return chars


def script_ratios(text):
    hangul = hanja = latin = alpha = 0
    for c in text:
        if c.isalpha():
            alpha += 1
            if "가" <= c <= "힣" or "ㄱ" <= c <= "ㆎ":
                hangul += 1
            elif "A" <= c <= "z":
                latin += 1
            elif "一" <= c <= "鿿" or "豈" <= c <= "﫿" or "㐀" <= c <= "䶿":
                hanja += 1
    return (hangul / alpha, hanja / alpha, latin / alpha) if alpha else (0.0, 0.0, 0.0)


def assess(text, words=None, t=FINEWEB2):
    """(ok, reason). reason은 datatrove 필터의 사유명을 따른다; ok면 'ok'."""
    if words is None:
        words = words_batch([text])[0]
    if not text.strip():
        return False, "empty"
    if "lorem ipsum" in text.lower():
        return False, "lorem_ipsum"
    hangul, hanja, latin = script_ratios(text)
    if hangul < t.hangul_min:
        return False, "hangul_ratio"
    if hanja > t.hanja_max:
        return False, "hanja_ratio"
    if latin > t.latin_max:
        return False, "latin_ratio"
    # Gopher repetition (fineweb-2: 문단·줄 글자 비율 필터는 비활성)
    lines = LINE.split(text)
    if duplicates(lines)[0] / len(lines) > t.dup_line_frac:
        return False, "dup_line_frac"
    size = len(text)
    for n, frac in t.top_n_grams:
        grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
        if grams:
            g, k = Counter(grams).most_common(1)[0]
            if len(g) * k / size > frac:
                return False, f"top_{n}_gram"
    for n, frac in t.dup_n_grams:
        if dup_ngram_chars(words, n) / size > frac:
            return False, f"duplicated_{n}_n_grams"
    # FineWeb quality (short_line 비활성, char_dup 0.1)
    full = [l for l in text.split("\n") if l.strip()]
    if sum(l.endswith(TERMINAL) for l in full) / len(full) < t.line_punct_thr:
        return False, "line_punct_ratio"
    if duplicates(full)[1] / len(text.replace("\n", "")) > t.char_dup_ratio:
        return False, "char_dup_ratio"
    if not words:
        return False, "gopher_short_doc"
    if text.count("\n") / len(words) > t.new_line_ratio:
        return False, "list_ratio"
    # Gopher quality
    real = [w for w in words if any(c not in PUNCTUATION for c in w)]
    if len(real) < t.min_words:
        return False, "gopher_short_doc"
    if len(real) > t.max_words:
        return False, "gopher_long_doc"
    avg = sum(map(len, real)) / len(real)
    if avg < t.min_avg_word_len:
        return False, "gopher_below_avg_threshold"
    if avg > t.max_avg_word_len:
        return False, "gopher_above_avg_threshold"
    if text.count("#") / len(words) > t.max_symbol_word_ratio:
        return False, "gopher_too_many_hashes"
    if (text.count("...") + text.count("…")) / len(words) > t.max_symbol_word_ratio:
        return False, "gopher_too_many_ellipsis"
    raw_lines = text.splitlines()
    if sum(l.lstrip().startswith(("•", "-")) for l in raw_lines) / len(raw_lines) > t.max_bullet_lines:
        return False, "gopher_too_many_bullets"
    if sum(l.rstrip().endswith(("...", "…")) for l in raw_lines) / len(raw_lines) > t.max_ellipsis_lines:
        return False, "gopher_too_many_end_ellipsis"
    if sum(any(c.isalpha() for c in w) for w in words) / len(words) < t.min_alpha_words:
        return False, "gopher_below_alpha_threshold"
    if len(STOPWORDS.intersection(words)) < t.min_stopwords:
        return False, "gopher_enough_stop_words"
    return True, "ok"


def assess_batch(texts, t=FINEWEB2):
    return [assess(x, w, t) for x, w in zip(texts, words_batch(texts))]


# ko-pii가 구간 끝에 붙여 오는 구분 문자(표 구분선으로 쓰는 'ㅣ', '|', 공백)는 형식 검사 전에 뗀다.
SPAN_TAIL = " \t\nㅣ|"


def well_formed(text, label, start, end):
    """원문 구간이 라벨 형식과 일치하고, 앞뒤에 숫자 묶음이나 영문자가 붙어 있지 않은 탐지만 인정한다."""
    return bool(SHAPES[label].fullmatch(text, start, end)) and not (
        GLUED_AFTER.match(text[end:end + 2]) or GLUED_BEFORE.search(text[max(0, start - 2):start]))


def _redact_once(text):
    from ko_pii import detect_all
    counts, out, pos = Counter(), [], 0
    for d in sorted(detect_all(text, include=list(PII_LABELS)), key=lambda d: (d.start, -d.end)):
        end = d.end
        while end > d.start and text[end - 1] in SPAN_TAIL:
            end -= 1
        if d.start < pos or not well_formed(text, d.label, d.start, end):
            continue
        out.append(text[pos:d.start])
        out.append(PII_LABELS[d.label])
        counts[d.label] += 1
        pos = end
    out.append(text[pos:])
    return "".join(out), counts


def redact(text):
    """PII를 고정 토큰으로 치환. (텍스트, 라벨별 치환 수). 결과가 바뀌지 않을 때까지(최대 3회) 반복한다: ko-pii는
    붙어 있는 연락처(전화 바로 뒤 이메일)를 겹친 구간으로 내놓아 한 번에 하나만 치환되는 경우가 있다(B-004)."""
    counts = Counter()
    for _ in range(3):
        if not PII_HINT.search(text):
            break
        text, found = _redact_once(text)
        if not found:
            break
        counts += found
    return text, counts
