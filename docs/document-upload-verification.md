# Document upload verification

Verified on 13 October 2026:

- Full suite: **110 tests passed**, plus 29 subtests. Includes existing API/UI/RAG behavior and upload session isolation, quotas, expiry, deletion, failed writes/rollback and interrupted Qdrant index setup.
- Actual Docling 2.138.0 CPU conversion: Vietnamese DOCX extracted two sections without invented page numbers; a text PDF extracted one section with page 1 provenance.
- Hosted integration: uploaded a synthetic company-policy DOCX to a separate Qdrant Cloud collection, embedded it with BGE-M3, searched the combined legal/upload candidates, reranked with hosted BGE, and generated a DeepSeek answer. Five sources included two uploaded chunks; the answer reproduced the document's `SM-2026` reference code. An invalid session token returned 401. Deletion succeeded afterward.
- Langfuse Cloud evidence: the answer trace contains retrieval, reranking, generation and token usage. Upload indexing records only byte/file/chunk counts. Test trace ID: `4affade46f83e09cc1649ec2925e2efe`.
- Both API/UI Docker targets build with CPU PyTorch/torchvision. Dependency check reports no broken requirements.
- Local production demo restored on port 7860 with Qdrant Cloud. Authenticated UI configuration includes the upload control; a DOCX upload and deletion both returned 200 from the built API image.
- Fresh review identified failed-rollback visibility and interrupted-index repair risks; both were fixed and covered by regression tests, including the strict Cloud cleanup retry path.

Limits: this verifies small text documents, not complex table fidelity, hostile-parser containment, concurrent public traffic or HF hosting. The example document explicitly identifies itself as test material, not law. Upload testing does not change the corpus benchmark or README Results.

## Vietnamese OCR check (2026-10-13)

Forced full-page Tesseract OCR (`vie` + `eng`) processed the supplied 13-page `06.2026.TT-TTCP.PDF` on CPU in 113.2 seconds and produced 213 text sections. Page 4's embedded text `Diu 6. Ung x trong thirc hin van hóa cong so'` became `Điều 6. Ứng xử trong thực hiện văn hóa công sở`, with page provenance retained. The broken heading did not survive in the OCR result. Other words still show recognition errors; this is a targeted check, not a document-wide accuracy measurement. Extracted evidence is saved locally under ignored `results/ocr-legal-sections.json`.

Focused tests cover API boolean validation, UI forwarding/status/timeouts, bounded parser execution, full-page Vietnamese OCR options, and DOCX retaining normal extraction. Tesseract and Vietnamese language data are installed in the API image; the existing legal corpus is not reindexed.

The deployed API also accepted an image-only PDF made from page 4 with `force_ocr=true`, extracted and indexed 16 chunks in Qdrant Cloud in 34.9 seconds, and successfully deleted that test session afterward. The deployed Gradio checkbox was verified visible and selectable. All 16 focused document/OCR tests and 19 existing Gradio tests passed.
