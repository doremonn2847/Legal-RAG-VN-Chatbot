import io
import time
import zipfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from qdrant_client import QdrantClient
from qdrant_client import models

from main.uploaded_documents import UploadedDocuments, chunk_sections, validate_file


def docx_bytes():
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w') as archive:
        archive.writestr('[Content_Types].xml', '<Types/>')
        archive.writestr('word/document.xml', '<document/>')
    return out.getvalue()


def test_validation_and_chunk_provenance():
    validate_file('legal.docx', docx_bytes())
    for name, content in [('a.exe', b'x'), ('bad.pdf', b'x'), ('big.pdf', b'%PDF' + b'x' * (10*1024*1024))]:
        with pytest.raises(ValueError):
            validate_file(name, content)
    chunks = chunk_sections([{'text': 'a' * 1400, 'page': 7}])
    assert len(chunks) == 2
    assert chunks[0]['text'][-150:] == chunks[1]['text'][:150]
    assert all(c['page'] == 7 for c in chunks)
    with pytest.raises(ValueError):
        chunk_sections([{'text': ' ', 'page': None}])


@pytest.fixture
def store():
    client = QdrantClient(':memory:')
    encoder = SimpleNamespace(get_sentence_embedding_dimension=lambda: 3,
                              encode=lambda texts, **kw: np.array([[1., 0., 0.] for _ in texts]))
    vector = SimpleNamespace(client=client, collection_name='legal', embedding_model=encoder,
                             embed_text=lambda query: [1., 0., 0.])
    with patch('main.uploaded_documents.extract_document', return_value=[{'text': 'Quyền nghỉ phép của người lao động.', 'page': 2}]):
        yield UploadedDocuments(vector)
    client.close()


def test_session_isolation_expiry_and_deletion(store):
    first = store.add('a.pdf', b'%PDF-1.4')
    second = store.add('b.pdf', b'%PDF-1.4')
    assert first['token'] != second['token']
    results = store.search('nghỉ phép', first['token'])
    assert len(results) == 1
    assert results[0]['metadata']['filename'] == 'a.pdf'
    assert results[0]['metadata']['page'] == 2
    with pytest.raises(ValueError):
        store.search('q', 'guessed-session')
    store.sessions[store.session_hash(first['token'])]['expires_at'] = time.time() - 1
    with pytest.raises(ValueError):
        store.search('q', first['token'])
    store.delete(second['token'])
    with pytest.raises(ValueError):
        store.search('q', second['token'])


def test_embedding_failure_does_not_publish_document(store):
    store.vector.embedding_model.encode = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('provider offline'))
    with pytest.raises(RuntimeError):
        store.add('a.pdf', b'%PDF-1.4')
    assert all(not s['files'] for s in store.sessions.values())


def test_file_quota_and_raw_file_cleanup(store):
    result = store.add('a.pdf', b'%PDF-1.4')
    store.add('b.pdf', b'%PDF-1.4', result['token'])
    store.add('c.pdf', b'%PDF-1.4', result['token'])
    with pytest.raises(ValueError, match='three'):
        store.add('d.pdf', b'%PDF-1.4', result['token'])
    from main.uploaded_documents import extract_document
    path = extract_document.call_args.args[0]
    assert not path.exists()


def test_expired_points_are_deleted(store):
    result = store.add('a.pdf', b'%PDF-1.4')
    assert store.client.count(store.collection).count == 1
    with patch('main.uploaded_documents.time.time', return_value=result['expires_at'] + 1):
        store.cleanup()
    assert store.client.count(store.collection).count == 0
    assert not store.sessions


def test_partial_upsert_is_rolled_back(store):
    with patch('main.uploaded_documents.extract_document', return_value=[{'text': 'Evidence', 'page': 1}] * 70):
        upsert = store.client.upsert
        calls = []
        def fail_second(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError('connection failed')
            return upsert(*args, **kwargs)
        with patch.object(store.client, 'upsert', side_effect=fail_second):
            with pytest.raises(RuntimeError):
                store.add('a.pdf', b'%PDF-1.4')
    assert store.client.count(store.collection).count == 0
    assert not store.sessions


def test_failed_rollback_never_exposes_uncommitted_chunks(store):
    token = store.add('committed.pdf', b'%PDF-1.4')['token']
    upsert = store.client.upsert
    calls = []
    delete = store.client.delete
    def fail_rollback(collection, selector, **kwargs):
        if isinstance(selector, models.PointIdsList):
            raise RuntimeError('offline')
        return delete(collection, selector, **kwargs)
    def fail_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError('offline')
        return upsert(*args, **kwargs)
    with patch('main.uploaded_documents.extract_document', return_value=[{'text': 'Uncommitted', 'page': 1}] * 70), \
            patch.object(store.client, 'upsert', side_effect=fail_second), \
            patch.object(store.client, 'delete', side_effect=fail_rollback):
        with pytest.raises(RuntimeError):
            store.add('rejected.pdf', b'%PDF-1.4', token)
    assert store.client.count(store.collection).count > 1
    assert {d['metadata']['filename'] for d in store.search('q', token)} == {'committed.pdf'}


def test_collection_setup_repairs_missing_indexes(store):
    create_index = store.client.create_payload_index
    with patch.object(store.client, 'create_payload_index', side_effect=RuntimeError('offline')):
        with pytest.raises(RuntimeError):
            store.ensure_collection()
    assert store.client.collection_exists(store.collection)
    with patch.object(store.client, 'create_payload_index', wraps=create_index) as create:
        store.ensure_collection()
    assert create.call_count == 3


def test_add_repairs_indexes_before_strict_cloud_cleanup(store):
    indexed = set()
    attempts = []
    create_index, delete = store.client.create_payload_index, store.client.delete
    def create(collection, field, schema, **kwargs):
        if field == 'expires_at' and not attempts:
            attempts.append(1)
            raise RuntimeError('index creation interrupted')
        indexed.add(field)
        return create_index(collection, field, schema, **kwargs)
    def strict_delete(*args, **kwargs):
        if 'expires_at' not in indexed:
            raise RuntimeError('unindexed Cloud filter')
        return delete(*args, **kwargs)
    with patch.object(store.client, 'create_payload_index', side_effect=create), \
            patch.object(store.client, 'delete', side_effect=strict_delete):
        with pytest.raises(RuntimeError):
            store.add('first.pdf', b'%PDF-1.4')
        result = store.add('retry.pdf', b'%PDF-1.4')
    assert result['files'][0]['filename'] == 'retry.pdf'
