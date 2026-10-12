"""Bound Docling conversion in a disposable CPU process; no uploaded URLs."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def extract_document(path):
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / 'sections.json'
        try:
            subprocess.run([sys.executable, '-m', 'utils.document_parser', str(path), str(output)],
                           check=True, timeout=180, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            raise ValueError('Document processing exceeded three minutes; use a smaller document.') from None
        except subprocess.CalledProcessError:
            raise ValueError('Cannot read this document. Use a valid text PDF/DOCX with at most 50 PDF pages.') from None
        return json.loads(output.read_text(encoding='utf-8'))


def convert(path):
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import InputFormat, ConversionStatus
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions(do_ocr=False, do_table_structure=False,
                                 document_timeout=150,
                                 accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU, num_threads=2))
    converter = DocumentConverter(allowed_formats=[InputFormat.PDF, InputFormat.DOCX],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
    result = converter.convert(path, max_num_pages=50, max_file_size=10*1024*1024)
    if result.status != ConversionStatus.SUCCESS:
        raise ValueError('Incomplete document conversion')
    sections = []
    for item, _ in result.document.iterate_items():
        text = getattr(item, 'text', '').strip()
        if text:
            provenance = getattr(item, 'prov', [])
            page = provenance[0].page_no if path.suffix.lower() == '.pdf' and provenance else None
            sections.append({'text': text, 'page': page})
    if sum(len(section['text']) for section in sections) > 500000:
        raise ValueError('Extracted document is too long')
    return sections


if __name__ == '__main__':
    Path(sys.argv[2]).write_text(json.dumps(convert(Path(sys.argv[1])), ensure_ascii=False), encoding='utf-8')
