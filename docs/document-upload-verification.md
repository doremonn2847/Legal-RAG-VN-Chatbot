# Document upload verification

Verified on 13 October 2026:

- Full suite: **110 tests passed**, plus 29 subtests. Includes existing API/UI/RAG behavior and upload session isolation, quotas, expiry, deletion, failed writes/rollback and interrupted Qdrant index setup.
- Actual Docling 2.138.0 CPU conversion: Vietnamese DOCX extracted two sections without invented page numbers; a text PDF extracted one section with page 1 provenance.
- Hosted integration: uploaded a synthetic company-policy DOCX to a separate Qdrant Cloud collection, embedded it with BGE-M3, searched the combined legal/upload candidates, reranked with hosted BGE, and generated a DeepSeek answer. Five sources included two uploaded chunks; the answer reproduced the document's `SM-2026` reference code. An invalid session token returned 401. Deletion succeeded afterward.
- Langfuse Cloud evidence: the answer trace contains retrieval, reranking, generation and token usage. Upload indexing records only byte/file/chunk counts. Test trace ID: `4affade46f83e09cc1649ec2925e2efe`.
- Both API/UI Docker targets build with CPU PyTorch/torchvision. Dependency check reports no broken requirements.
- Fresh review identified failed-rollback visibility and interrupted-index repair risks; both were fixed and covered by regression tests, including the strict Cloud cleanup retry path.

Limits: this verifies small text documents, not scanned-PDF OCR, complex table fidelity, hostile-parser containment, concurrent public traffic or HF hosting. The example document explicitly identifies itself as test material, not law. Upload testing does not change the corpus benchmark or README Results.
