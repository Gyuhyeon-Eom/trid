import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_workspace_index import api_snapshot


def record(source_id, text, restrictions=None, facts=None):
    return {'source_id': source_id, 'version': '2026-10-30', 'content_hash': hashlib.sha256(text.encode()).hexdigest(),
            'kind': 'rule', 'title': '가기관 여비 규정', 'issuer': {'id': 'C0001', 'name': '가기관'},
            'applies_to': {'scope': 'institution', 'ids': ['C0001']},
            'dates': {'published': '2026-01-01', 'effective_from': '2026-01-01', 'effective_to': None, 'retrieved_at': '2026-10-30'},
            'rights': {'license': 'KOGL-Type1', 'url': 'https://example.invalid', 'restrictions': restrictions},
            'location': {'url': 'https://example.invalid/rule', 'page': 1, 'article_path': ['제1조'], 'char_start': 0, 'char_end': len(text)},
            'text': text, 'structured_facts': facts or []}


class ApiSnapshotTests(unittest.TestCase):
    def snapshot(self, rows):
        d = Path(tempfile.mkdtemp())
        (d / 'normalized').mkdir()
        body = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
        (d / 'normalized/records-00000.jsonl').write_text(body)
        sha = hashlib.sha256(body.encode()).hexdigest()
        (d / 'manifest.json').write_text(json.dumps({'snapshot_id': 'api-test', 'producer': 'B/API-S3-001',
                                                     'files': [{'path': 'normalized/records-00000.jsonl', 'sha256': sha, 'records': len(rows)}]}))
        return d

    def test_records_become_documents_with_issuer_applicability_and_facts(self):
        fact = {'metric': '출장 식비 실지출 평균', 'value': 21000, 'unit': '원', 'basis': 'actual_spending',
                'subject': {'id': 'C0001', 'name': '가기관'}, 'period': {'from': '2025-01', 'to': '2025-12'}}
        d = self.snapshot([record('t:a', '제1조(식비) 식비는 하루 2만원으로 한다.', facts=[fact]),
                           record('t:b', '제1조 비공개 조항', restrictions='이용 제한')])
        meta = {}
        docs, facts, info = api_snapshot(d, meta)
        self.assertEqual([doc.doc_id for doc in docs], ['t:a@2026-10-30'])
        self.assertEqual(meta['t:a@2026-10-30']['institution'], '가기관')
        self.assertEqual(meta['t:a@2026-10-30']['applies_to']['scope'], 'institution')
        self.assertEqual(info['skipped']['restricted'], 1)
        self.assertEqual((facts[0]['value'], facts[0]['source_id']), (21000, 't:a'))

    def test_file_changed_after_manifest_is_rejected(self):
        d = self.snapshot([record('t:a', '제1조 본문')])
        (d / 'normalized/records-00000.jsonl').write_text('{}\n')
        with self.assertRaises(ValueError):
            api_snapshot(d, {})


if __name__ == '__main__':
    unittest.main()
