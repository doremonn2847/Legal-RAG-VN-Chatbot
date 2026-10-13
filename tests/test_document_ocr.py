import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from api import create_app
from app import VietnameseLegalChatbot, create_chatbot_interface
from utils.document_parser import extract_document, convert


class DocumentOcrTests(unittest.TestCase):
    def test_api_forwards_ocr_and_rejects_invalid_boolean(self):
        documents = MagicMock()
        documents.add.return_value = {'token': 'a'*43, 'files': [], 'expires_at': 1}
        with TestClient(create_app(lambda: MagicMock(), lambda r: None, lambda v: documents)) as client:
            response = client.post('/documents', files={'file': ('scan.pdf', b'%PDF-1.4')}, data={'force_ocr': 'true'})
            self.assertEqual(response.status_code, 200)
            documents.add.assert_called_once_with('scan.pdf', b'%PDF-1.4', None, force_ocr=True)
            self.assertEqual(client.post('/documents', files={'file': ('scan.pdf', b'%PDF-1.4')},
                                        data={'force_ocr': 'invalid'}).status_code, 422)

    def test_ui_sends_ocr_and_allows_longer_processing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'scan.pdf'
            path.write_bytes(b'%PDF-1.4')
            with patch('app.requests.post') as post:
                post.return_value.json.return_value = {'files': [], 'token': 'a'*43}
                VietnameseLegalChatbot().upload_document(path, force_ocr=True)
            self.assertEqual(post.call_args.kwargs['data'], {'force_ocr': 'true'})
            self.assertEqual(post.call_args.kwargs['timeout'], (5, 660))
        interface = create_chatbot_interface()
        handler = next(f.fn for f in interface.fns.values() if getattr(f.fn, '__name__', '') == 'handle_upload')
        with patch('app.VietnameseLegalChatbot.upload_document', return_value=('ready', 'token')) as upload:
            updates = list(handler('scan.pdf', None, True))
            self.assertIn('OCR tiếng Việt', updates[0][0])
            upload.assert_called_once_with('scan.pdf', None, force_ocr=True)

    def test_parser_subprocess_flag_and_timeout(self):
        def run(command, **kwargs):
            Path(command[4]).write_text(json.dumps([{'text': 'Điều 6', 'page': 4}]), encoding='utf8')
        for force, timeout in [(False, 180), (True, 630)]:
            with patch('utils.document_parser.subprocess.run', side_effect=run) as process:
                self.assertEqual(extract_document(Path('scan.pdf'), force)[0]['page'], 4)
                self.assertEqual('--ocr' in process.call_args.args[0], force)
                self.assertEqual(process.call_args.kwargs['timeout'], timeout)

    def test_ocr_forces_full_page_vietnamese_and_docx_stays_normal(self):
        from docling.datamodel.base_models import ConversionStatus, InputFormat
        with patch('docling.document_converter.DocumentConverter') as converter:
            result = converter.return_value.convert.return_value
            result.status = ConversionStatus.SUCCESS
            result.document.iterate_items.return_value = []
            for suffix, enabled in [('pdf', True), ('docx', False)]:
                convert(Path('document.'+suffix), force_ocr=True)
                options = converter.call_args.kwargs['format_options'][InputFormat.PDF].pipeline_options
                self.assertEqual(options.do_ocr, enabled)
                if enabled:
                    self.assertEqual(options.ocr_options.lang, ['vie', 'eng'])
                    self.assertEqual(str(options.ocr_options.mode.value), 'full_page')
