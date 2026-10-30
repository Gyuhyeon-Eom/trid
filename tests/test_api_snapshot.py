import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_workspace_index import api_snapshot, guides


def record(source_id, version, text, restrictions=None, facts=None):
    return {'source_id': source_id, 'version': version, 'content_hash': hashlib.sha256(text.encode()).hexdigest(),
            'kind': 'rule', 'title': '가기관 여비 규정', 'issuer': {'id': 'C0001', 'name': '가기관'},
            'applies_to': {'scope': 'institution', 'ids': ['C0001']},
            'dates': {'published': '2026-01-01', 'effective_from': '2026-01-02', 'effective_to': None, 'retrieved_at': '2026-10-30'},
            'date_kinds': {'published': '발령일자', 'effective_from': '시행일자'},
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
        d = self.snapshot([record('law:admrul:1', '901', '제1조(식비) 식비는 하루 2만원으로 한다.', facts=[fact]),
                           record('law:admrul:2', '902', '제1조 비공개 조항', restrictions='이용 제한'),
                           record('law:admrul:3', '903', '제1조 이미 색인된 규정')])
        meta = {'law:admrul:903': {'title': '이미 색인된 규정'}}  # same rule, serial number already a document id
        docs, facts, info = api_snapshot(d, meta)
        self.assertEqual([doc.doc_id for doc in docs], ['law:admrul:901'])
        self.assertEqual((meta['law:admrul:901']['institution'], meta['law:admrul:901']['date_type']), ('가기관', '발령일자'))
        self.assertEqual(meta['law:admrul:901']['applies_to']['scope'], 'institution')
        self.assertEqual((info['skipped']['restricted'], info['skipped']['duplicate']), (1, 1))
        self.assertEqual(meta['law:admrul:903']['applies_to']['scope'], 'institution')  # metadata kept, text not re-added
        self.assertEqual((facts[0]['value'], facts[0]['source_id']), (21000, 'law:admrul:1'))

    def test_file_changed_after_manifest_is_rejected(self):
        d = self.snapshot([record('law:admrul:1', '901', '제1조 본문')])
        (d / 'normalized/records-00000.jsonl').write_text('{}\n')
        with self.assertRaises(ValueError):
            api_snapshot(d, {})


class GuidesTests(unittest.TestCase):
    def test_only_substantive_licensed_rows_are_indexed_and_figure_limits_are_carried(self):
        def row(id_, text, license_='이용허락범위 제한 없음', figures=0, holds=()):
            return {'id': id_, 'title': f'기관_{id_}', 'sha256': id_, 'text': text, 'license': license_, 'source': 'https://example.invalid',
                    'flags': ['figure_content_missing'] if figures else [],
                    'meta': {'extractor': 'kordoc', 'table_check': 'unverified', 'status_holds': list(holds), 'figures_omitted': figures},
                    'provenance': {'institution': '가기관', 'doc_type': '업무매뉴얼', 'applies_to': '가기관 업무'}}
        body = '제1장 총칙\n' + '업무 절차를 설명한다. ' * 60
        rows = [row('ok', body), row('cover', '표지'), row('held', body, holds=['reserved_overlap']),
                row('nc', body, license_='공공누리 제3유형'), row('kogl1', body, license_='공공저작물 : 출처표시 (제 1유형)'),
                row('fig', body + '[그림 생략]', figures=423)]
        d = Path(tempfile.mkdtemp())
        (d / 'documents.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        meta, tables = {}, []
        docs, info = guides(d, meta, tables)
        self.assertEqual([x.doc_id for x in docs], ['ok', 'kogl1', 'fig'])
        self.assertEqual(info['skipped'], {'empty_or_cover': ['cover'], 'hold': ['held'], 'license': ['nc']})
        self.assertEqual((meta['ok']['limitations'], info['limited']), ([], ['fig']))
        self.assertIn('423곳', meta['fig']['limitations'][0])


if __name__ == '__main__':
    unittest.main()
