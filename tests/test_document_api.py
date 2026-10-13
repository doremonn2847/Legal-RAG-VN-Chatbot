from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from api import create_app


def test_upload_ask_and_delete_use_private_document_token():
    token = 'a' * 43
    rag = MagicMock()
    rag.answer_question.return_value = {'answer': 'Có căn cứ.', 'retrieved_documents': [{
        'id': 'upload_1', 'title': 'legal.pdf', 'content': 'Evidence', 'score': .9,
        'metadata': {'source': 'upload', 'filename': 'legal.pdf', 'page': 2}}]}
    documents = MagicMock()
    documents.add.return_value = {'token': token, 'files': [{'filename': 'legal.pdf', 'chunks': 1}], 'expires_at': 9999999999}
    app = create_app(lambda: rag, lambda r: None, lambda vector: documents)
    with TestClient(app) as client:
        response = client.post('/documents', files={'file': ('legal.pdf', b'%PDF-1.4')})
        assert response.status_code == 200
        assert response.json()['token'] == token
        documents.add.assert_called_once_with('legal.pdf', b'%PDF-1.4', None, force_ocr=False)
        response = client.post('/ask', json={'question': 'Nghỉ phép?'}, headers={'X-Document-Token': token})
        assert response.status_code == 200
        assert response.json()['sources'][0]['page'] == 2
        assert response.json()['sources'][0]['source_type'] == 'upload'
        retriever = rag.answer_question.call_args.kwargs['uploaded_documents']
        retriever('q')
        documents.search.assert_called_once_with('q', token)
        assert client.delete('/documents', headers={'X-Document-Token': token}).status_code == 200
        documents.delete.assert_called_once_with(token)


def test_invalid_document_session_is_not_scored_or_answered():
    rag, documents = MagicMock(), MagicMock()
    documents.session.side_effect = ValueError('expired')
    documents.delete.side_effect = ValueError('expired')
    documents.add.side_effect = ValueError('Invalid PDF file.')
    app = create_app(lambda: rag, lambda r: None, lambda vector: documents)
    with TestClient(app) as client:
        assert client.post('/ask', json={'question': 'q'}, headers={'X-Document-Token': 'b'*43}).status_code == 401
        assert client.delete('/documents').status_code == 401
        assert client.post('/documents', files={'file': ('bad.pdf', b'x')}).status_code == 422
        assert client.post('/documents', content=b'x', headers={'Content-Length': str(12*1024*1024)}).status_code == 413
    rag.answer_question.assert_not_called()
