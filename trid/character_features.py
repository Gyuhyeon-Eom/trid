"""Collision-free modern Hangul/jamo and ASCII IDs for span-boundary features."""
RANGES=((0,256),(0x1100,0x1200),(0x3130,0x3190),(0xA960,0xA980),(0xAC00,0xD7A4),(0xD7B0,0xD800))
EXPLICIT_SIZE=sum(end-start for start,end in RANGES)
HANGUL_VOCAB_SIZE=EXPLICIT_SIZE+512


def character_id(ch,encoding='hashed512'):
    code=ord(ch)
    if encoding=='hashed512':return code%512
    if encoding!='hangul_v1':raise ValueError('Unknown character encoding')
    offset=0
    for start,end in RANGES:
        if start<=code<end:return offset+code-start
        offset+=end-start
    return EXPLICIT_SIZE+code%512


def inherited_hash_indices():
    """Exact warm start: each expanded entry inherits its old hashed embedding."""
    return [code%512 for start,end in RANGES for code in range(start,end)]+list(range(512))
