# Session-private legal document uploads

Approved direction: PDF/DOCX uploads supplement the legal corpus, with private sessions, filename/page citations, processing feedback, limits, and deletion. Use Docling on CPU. Hosted embeddings, reranking, and generation receive extracted content; disclose this before upload.

The API issues an unguessable document token, independent of the existing telemetry session ID. Require that token to retrieve/delete uploads. Qdrant stores chunks in a separate collection derived from the embedding collection name, filtered by a hash of the token and expiry. No upload changes the original corpus or BM25 pickle. Sessions expire after 24 hours; delete expired points periodically and on startup. Restarting the API invalidates tokens. Delete source files immediately after conversion; Gradio temporary upload files use automatic cache cleanup.

Limits: PDF/DOCX only, 10 MiB per file, 50 PDF pages, three files and 300 chunks per session, 180-second conversion timeout. Disable OCR and table reconstruction for the first version; reject image-only/no-text documents. Bound DOCX uncompressed ZIP size to 100 MiB. Chunk extracted text at 1,200 characters with 150-character overlap and preserve PDF page provenance (DOCX has no stable page numbers).

The existing no-upload flow remains unchanged. With uploads, collect the same legal corpus candidates without early reranking, add up to 25 session chunks, and rerank the combined set once to final five. Generation labels user-provided sources separately and treats document contents as evidence, never instructions. The API exposes source type, filename, and page information; the UI renders these safely. Clear chat deletes uploads before resetting the token. Langfuse upload observations contain counts only, never session tokens or raw file content.

Verification: reject unsupported/oversized/malformed documents, test chunk overlap/provenance, token isolation, expiry and deletion, embedding failure rollback, API upload/ask/delete wiring, source citations, and existing API/UI regression tests. Run actual Docling DOCX/PDF conversion and an isolated hosted upload smoke check when dependencies are ready.
