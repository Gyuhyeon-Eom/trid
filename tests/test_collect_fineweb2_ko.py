"""FineWeb-2 rows must map to the corpus contract with stable ids and traceable sources."""
import gzip
import hashlib
import io
import json
import unicodedata
import unittest
from scripts.collect_fineweb2_ko import ShardWriter, row_of

RECORD = {'text': '\u1100\u1161 공공기관 안내', 'id': '<urn:uuid:ad224e20-74af-4f43-8e94-81f2c15355dc>',
          'dump': 'CC-MAIN-2013-20', 'url': 'http://example.kr/a', 'date': '2013-05-19T10:59:46Z',
          'language_score': 0.99, 'minhash_cluster_size': 3}


class FakeS3:
    def __init__(self):
        self.objects = {}

    def put_object(self, Bucket, Key, Body):
        self.objects[Key] = Body


class FineWebRowTests(unittest.TestCase):
    def test_contract_fields_and_stable_id(self):
        row = row_of(RECORD, '2026-10-30T00:00:00+00:00')
        text = unicodedata.normalize('NFC', RECORD['text'])
        self.assertEqual(row['text'], text)
        self.assertEqual(row['id'], 'fineweb2-ko:ad224e20-74af-4f43-8e94-81f2c15355dc')
        self.assertEqual(row['group'], row['id'])
        self.assertEqual(row['sha256'], hashlib.sha256(text.encode()).hexdigest())
        self.assertEqual(row['date'], '2013-05-19')
        self.assertEqual(row['source'], 'http://example.kr/a')
        self.assertIn('ODC-By', row['license'])
        self.assertEqual(row['meta']['dump'], 'CC-MAIN-2013-20')
        self.assertEqual(row_of(dict(RECORD, id=None), 'x')['id'], row_of(dict(RECORD, id=None), 'y')['id'])

    def test_shards_split_by_characters_and_round_trip(self):
        s3 = FakeS3()
        writer = ShardWriter(s3, '000_00000', limit=20)
        rows = [row_of(dict(RECORD, id=f'<urn:uuid:{i:08d}-0000-0000-0000-000000000000>'), 't') for i in range(5)]
        for row in rows:
            writer.add(row)
        writer.flush()
        self.assertEqual(len(writer.keys), 2)  # 9-char texts: flush after the 3rd row, rest at the end
        back = [json.loads(line) for key in writer.keys
                for line in gzip.GzipFile(fileobj=io.BytesIO(s3.objects[key])).read().decode().splitlines()]
        self.assertEqual([r['id'] for r in back], [r['id'] for r in rows])


if __name__ == '__main__':
    unittest.main()
