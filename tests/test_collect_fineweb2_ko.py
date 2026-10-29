"""FineWeb-2 clean-v4: contract rows, PII redaction, licence/reservation/TLD quarantine, spam, deterministic splits."""
import base64
import gzip
import hashlib
import io
import json
import unittest
from collections import Counter
from scripts.collect_fineweb2_ko import ShardWriter, clean_record, spam_reason, split_of, verify

RECORD = {'text': '\u1100\u1161 공공기관 안내입니다. 신청서는 누리집에서 내려받아 작성한 뒤 증빙서류와 함께 제출합니다. '
                  '자세한 문의는 담당자 hong@example.kr 로 보내 주세요.',
          'id': '<urn:uuid:ad224e20-74af-4f43-8e94-81f2c15355dc>', 'dump': 'CC-MAIN-2013-20',
          'url': 'http://www.example.kr/a', 'date': '2013-05-19T10:59:46Z', 'language_score': 0.99,
          'minhash_cluster_size': 3}


class FakeS3:
    def __init__(self):
        self.objects = {}

    def put_object(self, Bucket, Key, Body, ChecksumSHA256=None):
        assert base64.b64decode(ChecksumSHA256) == hashlib.sha256(Body).digest()
        self.objects[Key] = Body

    def head_object(self, Bucket, Key, ChecksumMode=None):
        body = self.objects[Key]
        return {'ContentLength': len(body), 'ChecksumSHA256': base64.b64encode(hashlib.sha256(body).digest()).decode()}


class CleanV4Tests(unittest.TestCase):
    def test_row_contract_pii_and_split(self):
        row, reason, pii = clean_record(RECORD, '2026-10-30T00:00:00+00:00')
        self.assertEqual(reason, 'ok')
        self.assertTrue(row['text'].startswith('가 공공기관'))
        self.assertNotIn('hong@example.kr', row['text'])
        self.assertIn('[이메일]', row['text'])
        self.assertEqual(pii, Counter({'EMAIL': 1}))
        self.assertEqual(row['sha256'], hashlib.sha256(row['text'].encode()).hexdigest())
        self.assertEqual(row['id'], 'fineweb2-ko:ad224e20-74af-4f43-8e94-81f2c15355dc')
        self.assertEqual((row['clean_version'], row['split'], row['meta']['source_split']), ('clean-v4', split_of(row['group']), 'train'))
        self.assertIn('pii_redacted', row['flags'])
        self.assertEqual(row['provenance'], {'institution_type': 'web', 'site': 'example.kr'})

    def test_paid_report_sites_are_quarantined_and_non_korean_dropped(self):
        row, reason, _ = clean_record(dict(RECORD, url='https://www.happycampus.com/report/1'), 't')
        self.assertEqual(reason, 'quarantine:paid_report_site')
        self.assertIn('quarantine:paid_report_site', row['flags'])
        self.assertEqual(clean_record(dict(RECORD, text='This page is entirely English text about travel.'), 't')[1], 'script_ratio')
        row, reason, _ = clean_record(dict(RECORD, url='http://report.kyobobook.co.kr/view/2003/1219838/'), 't')
        self.assertEqual(reason, 'quarantine:paid_report_site')

    def test_pages_of_reserved_institutions_are_held_back_pending_text_check(self):
        reason = clean_record(dict(RECORD, url='https://home.kepco.co.kr/kepco/main.do'), 't')[1]
        self.assertEqual(reason, 'quarantine:benchmark_reserved_host_pending_text_check')
        self.assertEqual(clean_record(dict(RECORD, url='https://www.alio.go.kr/x'), 't')[1],
                         'quarantine:benchmark_reserved_host_pending_text_check')

    def test_uncommon_tlds_are_held_not_dropped(self):
        row, reason, _ = clean_record(dict(RECORD, url='https://pokertis.info/a'), 't')
        self.assertEqual(reason, 'quarantine:uncertain_tld_mt_seo_risk')
        self.assertIn('quarantine:uncertain_tld_mt_seo_risk', row['flags'])
        self.assertEqual(clean_record(dict(RECORD, url='https://news.example.co.kr/a'), 't')[1], 'ok')

    def test_ad_spam_is_dropped_but_policy_texts_on_gambling_are_kept(self):
        ads = '먹튀검증 완료된 토토사이트 추천! 안전놀이터 가입 시 이벤트. 우리카지노 주소 안내. ' * 3
        self.assertEqual(spam_reason(ads), 'spam_gambling_adult_ads')
        policy = ('경찰청은 사이버도박 특별단속 결과 불법 온라인 카지노 등 도박사이트 운영자를 검거했다. '
                  '먹튀공탁 등 형사공탁 악용을 막고, 조건만남 피해 청소년 보호 대책도 함께 추진한다. ') * 5
        self.assertIsNone(spam_reason(policy))
        one_mention = '토토사이트 추천이라는 광고 문구에 속지 않도록 주의하라는 안내입니다. ' + '정부 정책 설명 문장입니다. ' * 40
        self.assertIsNone(spam_reason(one_mention))

    def test_split_is_stable_and_mostly_train(self):
        splits = Counter(split_of(f'fineweb2-ko:{i}') for i in range(20000))
        self.assertEqual(split_of('fineweb2-ko:x'), split_of('fineweb2-ko:x'))
        self.assertGreater(splits['train'], 19500)
        self.assertTrue(50 < splits['valid'] < 150 and 50 < splits['test'] < 150)

    def test_shards_round_trip_and_verify(self):
        s3 = FakeS3()
        writer = ShardWriter(s3, 'corpus/clean/fineweb2-ko/clean-v1/docs', '000_00000', limit=60)
        rows = [clean_record(dict(RECORD, id=f'<urn:uuid:{i:08d}-0000-0000-0000-000000000000>'), 't')[0] for i in range(5)]
        for row in rows:
            writer.add(row)
        writer.flush()
        verify(s3, writer.shards)
        back = [json.loads(line) for shard in writer.shards
                for line in gzip.GzipFile(fileobj=io.BytesIO(s3.objects[shard['key']])).read().decode().splitlines()]
        self.assertEqual([r['id'] for r in back], [r['id'] for r in rows])
        self.assertEqual(sum(s['rows'] for s in writer.shards), 5)
        s3.objects[writer.shards[0]['key']] += b'x'
        with self.assertRaises(RuntimeError):
            verify(s3, writer.shards)


if __name__ == '__main__':
    unittest.main()
