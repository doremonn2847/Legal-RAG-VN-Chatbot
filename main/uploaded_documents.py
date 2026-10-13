"""Private, expiring upload chunks; never modifies the legal corpus collection."""
import hashlib
import io
from pathlib import Path
import secrets
import tempfile
import time
import uuid
import zipfile

import numpy as np
from qdrant_client import models
from utils.document_parser import extract_document

MAX_FILE_BYTES = 10 * 1024 * 1024


def validate_file(filename, content):
    suffix = Path(filename).suffix.lower()
    if suffix not in ('.pdf', '.docx') or not content or len(content) > MAX_FILE_BYTES:
        raise ValueError('Use a PDF/DOCX file of at most 10 MiB.')
    if suffix == '.pdf' and not content.startswith(b'%PDF-'):
        raise ValueError('Invalid PDF file.')
    if suffix == '.docx':
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                entries = archive.infolist()
                if (len(entries) > 10000 or sum(e.file_size for e in entries) > 100*1024*1024
                        or '[Content_Types].xml' not in archive.namelist()
                        or 'word/document.xml' not in archive.namelist()):
                    raise ValueError('Invalid or oversized DOCX archive.')
        except zipfile.BadZipFile:
            raise ValueError('Invalid DOCX file.') from None


def chunk_sections(sections):
    chunks = []
    for section in sections:
        text = section['text'].strip()
        for start in range(0, len(text), 1050):
            chunks.append({'text': text[start:start+1200], 'page': section.get('page')})
            if start + 1200 >= len(text):
                break
        if len(chunks) > 300:
            raise ValueError('Document exceeds the 300-chunk session limit.')
    if not chunks:
        raise ValueError('No readable text found. Enable OCR for scanned PDFs.')
    return chunks


class UploadedDocuments:
    def __init__(self, vector):
        self.vector = vector
        self.client = vector.client
        self.collection = vector.collection_name + '_uploads'
        self.sessions = {}

    @staticmethod
    def session_hash(token):
        return hashlib.sha256(token.encode()).hexdigest()

    def session(self, token):
        if not isinstance(token, str) or not token:
            raise ValueError('Document session is required.')
        key = self.session_hash(token)
        session = self.sessions.get(key)
        if not session or session['expires_at'] <= time.time():
            raise ValueError('Document session expired. Please upload your documents again.')
        return key, session

    def ensure_collection(self):
        dimensions = self.vector.embedding_model.get_sentence_embedding_dimension()
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(self.collection, vectors_config=models.VectorParams(
                size=dimensions, distance=models.Distance.COSINE))
        info = self.client.get_collection(self.collection)
        if info.config.params.vectors.size != dimensions:
            raise RuntimeError('Upload collection has incompatible embedding dimensions.')
        for field, schema in [('session_hash', models.PayloadSchemaType.KEYWORD),
                              ('expires_at', models.PayloadSchemaType.INTEGER),
                              ('document_id', models.PayloadSchemaType.KEYWORD)]:
            if field not in info.payload_schema:
                self.client.create_payload_index(self.collection, field, schema, wait=True)

    def add(self, filename, content, token=None, force_ocr=False):
        validate_file(filename, content)
        self.cleanup()
        if token:
            key, session = self.session(token)
        else:
            if len(self.sessions) >= 20:
                raise ValueError('Upload capacity reached; try again later.')
            token = secrets.token_urlsafe(32)
            key = self.session_hash(token)
            session = {'expires_at': int(time.time()) + 86400, 'files': [], 'chunks': 0}
        if len(session['files']) >= 3:
            raise ValueError('A session supports at most three files; clear documents before adding more.')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ('document' + Path(filename).suffix.lower())
            path.write_bytes(content)
            chunks = chunk_sections(extract_document(path, force_ocr=force_ocr and path.suffix == '.pdf'))
        if session['chunks'] + len(chunks) > 300:
            raise ValueError('Documents exceed the 300-chunk session limit.')
        vectors = self.vector.embedding_model.encode([c['text'] for c in chunks], batch_size=16)
        dimensions = self.vector.embedding_model.get_sentence_embedding_dimension()
        if np.shape(vectors) != (len(chunks), dimensions) or not np.isfinite(vectors).all():
            raise RuntimeError('Invalid document embeddings.')
        self.ensure_collection()
        identifier = uuid.uuid4().hex
        filename = filename.replace('\\', '/').rsplit('/', 1)[-1][:200]
        points = []
        for chunk, vector in zip(chunks, vectors):
            point_id = str(uuid.uuid4())
            points.append(models.PointStruct(id=point_id, vector=vector.tolist(), payload={
                'session_hash': key, 'expires_at': session['expires_at'], 'document_id': identifier,
                'id': 'upload_' + point_id, 'title': filename, 'content': chunk['text'],
                'metadata': {'source': 'upload', 'filename': filename, 'page': chunk['page']}}))
        try:
            for start in range(0, len(points), 64):
                self.client.upsert(self.collection, points[start:start+64], wait=True)
        except Exception:
            try:
                self.client.delete(self.collection, models.PointIdsList(points=[p.id for p in points]), wait=True)
            except Exception:
                # Uncommitted IDs are excluded by search; expiry retries physical cleanup.
                pass
            raise
        session['files'].append({'id': identifier, 'filename': filename, 'chunks': len(chunks)})
        session['chunks'] += len(chunks)
        self.sessions[key] = session
        return {'token': token, 'files': session['files'], 'expires_at': session['expires_at']}

    def search(self, query, token):
        key, session = self.session(token)
        if not self.client.collection_exists(self.collection):
            return []
        results = self.client.query_points(self.collection, query=self.vector.embed_text(query), limit=25,
            query_filter=models.Filter(must=[models.FieldCondition(key='session_hash', match=models.MatchValue(value=key)),
                models.FieldCondition(key='document_id', match=models.MatchAny(any=[f['id'] for f in session['files']])),
                models.FieldCondition(key='expires_at', range=models.Range(gt=int(time.time())))]),
            with_payload=True).points
        return [{**{k: p.payload[k] for k in ('id', 'title', 'content', 'metadata')},
                 'score': max(0., min(1., (p.score+1)/2))} for p in results]

    def delete(self, token):
        key, _ = self.session(token)
        if self.client.collection_exists(self.collection):
            self.client.delete(self.collection, models.FilterSelector(filter=models.Filter(must=[
                models.FieldCondition(key='session_hash', match=models.MatchValue(value=key))])), wait=True)
        self.sessions.pop(key, None)

    def cleanup(self):
        now = int(time.time())
        if self.client.collection_exists(self.collection):
            self.ensure_collection()
            self.client.delete(self.collection, models.FilterSelector(filter=models.Filter(must=[
                models.FieldCondition(key='expires_at', range=models.Range(lte=now))])), wait=True)
        self.sessions = {k: v for k, v in self.sessions.items() if v['expires_at'] > now}
