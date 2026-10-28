"""Isolated, bounded text extraction for the public test demo."""
import json
import logging
import resource
import sys

# CPU and address-space limits supplement the parent's 15 second wall timeout.
resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
if sys.platform != 'darwin':
    resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, 1024 * 1024 * 1024))
logging.disable(logging.CRITICAL)
from trid import pdf_documents as pdf
pdf.MAX_BYTES = 2 * 1024 * 1024
pdf.MAX_PAGES = 20
pdf.MAX_CHARACTERS = 50_000

try:
    raw = sys.stdin.buffer.read(pdf.MAX_BYTES + 1)
    pages, warnings = pdf.extract_pdf(raw, 'document.pdf')
    print(json.dumps({'pages': pages, 'warnings': warnings}, ensure_ascii=False))
except Exception:
    print(json.dumps({'error': '텍스트가 있는 PDF 2MB·20페이지·5만 자 이하로 다시 시도해 주세요.'}, ensure_ascii=False))
