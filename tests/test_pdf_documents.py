from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import unittest
from pypdf import PdfWriter
from trid.pdf_documents import extract_pdf,DocumentError,Document,Chunk,make_chunks,retrieve,answer_document


class CharTokenizer:
    def encode(self,text):return SimpleNamespace(ids=list(range(len(text))),offsets=[(i,i+1) for i in range(len(text))])


class PdfDocumentTests(unittest.TestCase):
    def test_korean_pdf_pages_and_text(self):
        path=Path(__file__).resolve().parents[1]/'output/pdf/trid-pdf-sample.pdf'
        if not path.exists():self.skipTest('PDF smoke fixture unavailable')
        pages,warnings=extract_pdf(path.read_bytes(),'sample.pdf')
        self.assertEqual(len(pages),2)
        self.assertIn('별누리문화재단',pages[0]);self.assertIn('운영지원팀',pages[1]);self.assertTrue(warnings)

    def test_invalid_blank_encrypted_pdf(self):
        with self.assertRaisesRegex(DocumentError,'형식'):extract_pdf(b'not pdf','x.pdf')
        for encrypted in [False,True]:
            writer=PdfWriter();writer.add_blank_page(width=100,height=100)
            if encrypted:writer.encrypt('password')
            data=BytesIO();writer.write(data)
            with self.assertRaisesRegex(DocumentError,'암호화' if encrypted else 'OCR'):
                extract_pdf(data.getvalue(),'x.pdf')

    def test_chunks_preserve_page_offsets(self):
        pages=['가나다 abc 123\n' * 25,'두 번째 페이지.']
        chunks=make_chunks(pages,CharTokenizer(),budget=40,overlap=12)
        self.assertGreater(len(chunks),2)
        for c in chunks:self.assertEqual(c.text,pages[c.page-1][c.start:c.end])
        self.assertTrue(any(c.page==2 for c in chunks))

    def test_no_filename_institution_inference(self):
        doc=Document('별누리문화재단.pdf',['식비는 3만원이다.'],[Chunk(1,0,11,'식비는 3만원이다.')],[])
        self.assertEqual(retrieve(doc,'기관명'),[])

    def test_multiple_candidates_require_review_and_grounding(self):
        doc=Document('a.pdf',['해외 식비 8만원','해외 식비 9만원'],[Chunk(1,0,9,'해외 식비 8만원'),Chunk(2,0,9,'해외 식비 9만원')],[])
        def reader(s,p,t,text,q,d):
            a=text.index(' ' ,3)+1
            return {'abstains':False,'answer':text[a:],'source_character_span':[a,len(text)]}
        result=answer_document(doc,'해외 식비',None,None,CharTokenizer(),'cpu',reader)
        self.assertEqual(result['status'],'review');self.assertEqual(len(result['candidates']),2)
        self.assertEqual([x['page'] for x in result['candidates']],[1,2])
        def invalid(*args):return {'abstains':False,'answer':'invented','source_character_span':[0,1]}
        with self.assertRaisesRegex(DocumentError,'원문 위치'):
            answer_document(doc,'해외 식비',None,None,CharTokenizer(),'cpu',invalid)
