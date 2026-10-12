# Document Uploads Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add private legal document uploads to the existing chatbot.

**Architecture:** Docling converts local temporary PDF/DOCX files in a subprocess with a timeout. A small upload store manages tokens, limits, expiry and a separate Qdrant collection. API/UI reuse existing retrieval, reranking, generation and tracing.

**Tech Stack:** Docling 2.x, FastAPI multipart uploads, Qdrant, Gradio.

**Spec:** ../specs/2026-10-13-document-uploads.md

## Global Constraints

Use the approved 10 MiB/50-page/three-file/300-chunk/24-hour limits. Preserve the no-upload pipeline and README Results. Do not expose document tokens to logs/traces. CPU conversion without OCR.

## Review Focus

- Cross-session retrieval: filters must use server-issued tokens, never caller-selected telemetry IDs.
- Failed upload: do not publish partially indexed documents.
- Malformed ZIP/PDF: validate format/size before conversion and bound runtime.
- Expired sessions: exclude points immediately, delete them periodically.
- Citations: PDF pages must survive chunking; DOCX must not invent pages.

### Task 1: Parsing and private storage

Files: `utils/document_parser.py`, `main/uploaded_documents.py`, `tests/test_document_uploads.py`, requirements/Dockerfile.
Interfaces: parser `extract_document(path) -> list[dict]`; store `add(filename, content, token=None)`, `search(query, token)`, `delete(token)`, `cleanup()`.

- [x] Write/run validation, chunking, isolation, expiry and rollback tests.
- [x] Implement bounded CPU conversion and Qdrant storage using existing encoder/client.
- [x] Verify actual Docling conversion plus focused tests.

### Task 2: API, retrieval and citations

Files: `api.py`, `main/chatbot.py`, `config.py`, `tests/test_document_api.py`, `tests/test_rag_dependencies.py`.
Interfaces: POST/DELETE `/documents`, header `X-Document-Token`; optional `uploaded_documents` retriever on `answer_question` and `retrieve_documents`; additive source fields.

- [x] Test upload/ask/delete authorization and cleanup without hosted API calls.
- [x] Merge upload candidates before reranking; add precise uploaded source labels/instruction boundaries.
- [x] Run existing API/RAG regressions.

### Task 3: Gradio and documentation

Files: `app.py`, `tests/test_gradio_api.py`, `.env.example`, `compose.yaml`, README descriptive sections, `docs/document-uploads.md`.

- [x] Add upload control, processing status, disclosure, token state and delete-on-clear.
- [x] Test token transport and escaped source citations.
- [x] Run integrated conversion/upload smoke and UI construction; document remaining OCR/hosting limitations.
