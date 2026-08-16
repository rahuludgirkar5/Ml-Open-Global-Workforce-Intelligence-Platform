from pathlib import Path
import sys

try:
    import PyPDF2
except ImportError:
    print('PyPDF2 not installed')
    raise

if len(sys.argv) < 2:
    print('Usage: python extract_pdf.py <pdf_path>')
    sys.exit(1)

pdf_path = Path(sys.argv[1])
if not pdf_path.exists():
    print(f'PDF not found: {pdf_path}')
    sys.exit(1)

out_path = pdf_path.with_suffix('.txt')

with pdf_path.open('rb') as f:
    reader = PyPDF2.PdfReader(f)
    pages = []
    for p in reader.pages:
        text = p.extract_text()
        if text:
            pages.append(text)

out_path.write_text('\n\n'.join(pages), encoding='utf-8')
print(f'Extracted text written to: {out_path}')
