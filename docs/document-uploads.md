# Uploaded legal documents

The Gradio sidebar accepts PDF and DOCX files as additional sources. Select a file, then click **Xử lý tài liệu**. Wait for confirmation before asking a question. Repeat to add another file. Clearing the conversation deletes the session's uploaded chunks before resetting the UI.

Extracted text is sent to configured embedding, reranking and generation providers and stored temporarily in Qdrant. Existing Langfuse content settings apply to questions, retrieved excerpts and generation context; use `LANGFUSE_CAPTURE_CONTENT=false` to exclude content from traces. Upload indexing spans record counts, never document session tokens.

- PDF/DOCX only; 10 MiB/file, 50 PDF pages, three files and 300 chunks/session.
- Docling runs on CPU. Enable **PDF scan / sửa văn bản lỗi bằng OCR tiếng Việt** for image-only PDFs or broken selectable text. Tesseract re-reads every PDF page in Vietnamese/English, replacing the existing text layer. Normal extraction remains the default; DOCX ignores OCR. Verify legal wording, dates and article numbers against the original. Table reconstruction is disabled; complex tables may lose structure. First PDF conversion may download Docling layout models and take longer.
- Chunks use 1,200 characters with 150-character overlap within extracted text items. PDF page citations come from Docling provenance. DOCX page numbers are omitted because pagination depends on the renderer.
- Normal conversion is limited to three minutes; OCR conversion gets ten minutes plus subprocess cleanup time. Invalid, oversized, empty and excessive decompressed DOCX archives are rejected.
- Uploads are separate from the legal corpus and BM25 index. Search adds up to 25 session chunks to the usual legal candidates, then reranks the combined set once to final top-5.
- Server-issued random tokens control access. Telemetry `X-Session-ID` values cannot access uploads. Tokens expire after 24 hours and become invalid after API restart. Do not share tokens: they grant access to session documents.
- Expired chunks are excluded immediately and deleted every minute while the API runs, including after startup. Cleanup retries if Qdrant is unavailable. Raw API files are deleted after conversion; Gradio's temporary cache is cleaned hourly after files reach 24 hours.
- Uploaded files are labeled as user-provided sources, not verified current law. Instructions inside sources must not override chatbot instructions.

## API

`POST /documents` accepts one multipart `file` and optional boolean `force_ocr` (default `false`), returning `token`, `files`, and `expires_at` (Unix seconds). Content-Length is required. Send `X-Document-Token` on subsequent uploads and `/ask` requests. `DELETE /documents` with the same header removes uploaded chunks. Invalid/expired tokens cannot retrieve documents; upload again after clearing the old session.

One API worker serializes indexing, deletion and answering. A busy API returns 429. Sessions are temporary, not permanent personal storage.

Example from a local terminal (use `curl.exe` in Windows PowerShell):

```bash
curl -X POST http://127.0.0.1:8000/documents -F "file=@document.pdf" -F "force_ocr=true"
```

Use the returned `token` as `X-Document-Token` when asking with uploaded sources.
Leaving out `force_ocr` retains normal extraction. Full-page OCR is necessary for
a corrupted existing text layer; text selectability alone does not prove quality.

## Installation

Docker installs compatible CPU-only PyTorch/torchvision and Tesseract with Vietnamese language data. For local Python, install Tesseract with `vie` and `eng` language data on PATH, install `torch==2.8.0 torchvision==0.23.0` from `https://download.pytorch.org/whl/cpu` first, then install `requirements.txt`. Rebuild the API image when upgrading. No new API keys are required.

The upload collection needs Qdrant permissions to create collections/payload indexes, read, upsert and delete points. It uses the active embedding dimensions; its name is the legal collection name followed by `_uploads`.
