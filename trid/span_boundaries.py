"""One position-based span boundary rule shared by trid.span_scoring.candidates (string candidates) and
trid.extractive (start/end masks, label placement, best_span). A span is text[start:end] of one passage; the rule
only reads characters around the cut and never mutates the text.

Start (cut before `pos`): the passage start, after whitespace, or after a punctuation delimiter (parentheses,
quotes, middle dot, tilde, hyphen, ...) is fine, and an opening bracket/quote may start a span even right after a
word ('아파트(6억원)' -> '(6억원)'); a cut between two word characters (Hangul, letters, digits: '김석동' -> '석동',
'3일' -> '일', 'A3' -> '3') is not, nor a start on other punctuation glued to a word ('30.4%' -> '%'), nor a cut
beside an internal comma/period (below).
End (cut after `pos`): the passage end, or any cut that does not separate two ASCII alphanumerics and does not
sit beside an internal comma/period. Ending inside a Hangul word stays allowed so particles drop off ('3일 전'
out of '3일 전까지', '102%' out of '102%를').
Internal punctuation protects both of its cuts: a comma between digits ('1,000') and a period between ASCII
alphanumerics ('30.4', 'foo.bar'). A separating comma ('7.31 대 1, 7.94') or a terminal period ('...다.') is
a delimiter, so ending right before it (or starting right after it) is allowed.
Whitespace, bracket characters, passage limits, the length limits and the word limit are the callers' business.
"""

BOUNDARY_POLICY = 'span_boundary_v2'  # recorded in training metadata and reader outputs; bump when this file's rule changes

DIGITS = frozenset('0123456789')
OPENERS = frozenset('([{<〈《「『【‘“"\'')  # may start a span even when glued to the preceding word


def _core(ch):
    """ASCII letter or digit: the characters that are never split from each other at either end."""
    return ch.isascii() and ch.isalnum()


def _internal(text, i, lo, hi):
    """text[i] is a comma between digits or a period between ASCII alphanumerics inside text[lo:hi]."""
    if not lo < i < hi - 1:
        return False
    ch = text[i]
    if ch == ',':
        return text[i - 1] in DIGITS and text[i + 1] in DIGITS
    if ch == '.':
        return _core(text[i - 1]) and _core(text[i + 1])
    return False


def start_ok(text, pos, lo=0, hi=None):
    """A span of text[lo:hi] may start at `pos` (text[pos] assumed non-whitespace)."""
    hi = len(text) if hi is None else hi
    if pos == lo or text[pos - 1].isspace():
        return True
    if text[pos - 1].isalnum() and (text[pos].isalnum() or text[pos] not in OPENERS):
        return False
    return not (_internal(text, pos, lo, hi) or _internal(text, pos - 1, lo, hi))


def end_ok(text, pos, lo=0, hi=None):
    """A span of text[lo:hi] may end after `pos` (its last character, assumed non-whitespace)."""
    hi = len(text) if hi is None else hi
    if pos + 1 >= hi:
        return True
    if _core(text[pos]) and _core(text[pos + 1]):
        return False
    return not (_internal(text, pos, lo, hi) or _internal(text, pos + 1, lo, hi))
