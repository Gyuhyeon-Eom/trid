import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('korean_corpus',Path(__file__).resolve().parents[1]/'scripts/prepare_korean_continuation.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class KoreanDataTests(unittest.TestCase):
    def test_near_duplicate_is_checked_by_actual_shingle_overlap(self):
        index=module.NearDuplicates()
        text=' '.join(f'한국어{i}' for i in range(120))
        self.assertIsNone(index.match_and_add('valid:1',text))
        result=index.match_and_add('train:1',text+' 마지막 단어 추가')
        self.assertEqual(result[0],'valid:1')
        self.assertGreaterEqual(result[1],.8)
        self.assertIsNone(index.match_and_add('train:2','완전히 다른 내용의 독립적인 문서입니다.'))

    def test_nfc_shingles_match(self):
        import unicodedata
        text='한글 문장을 정규화하여 같은 문자로 읽습니다.'
        self.assertEqual(module.shingles(text),module.shingles(unicodedata.normalize('NFD',text)))

if __name__=='__main__':unittest.main()
