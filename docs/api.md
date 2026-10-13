# Local FastAPI service

The API reuses the existing original-query hybrid retrieval, BGE reranking and
configured DeepSeek/Ollama answer generator. Answers use up to five selected sources.
Legal-domain detection may make an additional generation call; document-session
questions bypass this filter. The current demo uses hosted BGE and DeepSeek.

## Start

Run from the repository root, with the existing virtual environment activated:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1
```

Copy `.env.example` to `.env` and configure provider credentials. For native
Cloud access use `QDRANT_URL` and `QDRANT_API_KEY`; Compose uses `QDRANT_CLOUD_*`.
Ollama is required only when `INFERENCE_PROVIDER=ollama`. See
[hosted configuration](hosted-retrieval.md). Native OCR requires Tesseract with
Vietnamese/English language data on PATH. Keep `ENABLE_RERANKING=True` in `config.py`.
The populated Qdrant collection and trusted `index/bm25_index.pkl` must already
exist and have matching nonzero document counts. Starting an empty Docker volume
does not import the collection: use the project's existing index transfer workflow
first. The API never rebuilds or deletes indexes. A pickle is executable data;
only load your own trusted project index.

Startup creates the configured provider clients and checks dependencies.
Local mode loads embedding/reranker weights; hosted mode does not. Docling layout
weights may be downloaded on the first PDF conversion. Startup fails if required components,
indexes, or the configured generation provider are unavailable. Use one worker to avoid
duplicating large models and overlapping inference. CPU inference can be slow.

Open http://127.0.0.1:8000/docs to try requests interactively.

```powershell
$body = @{ question = 'Người lao động được nghỉ phép năm bao nhiêu ngày?' } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/ask -ContentType 'application/json; charset=utf-8' -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

## Interface

- `GET /health`: lightweight process liveness.
- `GET /ready`: checks Qdrant collection and configured generation model availability; returns
  503 when dependencies are unavailable. This does not prove generation succeeds.
- `POST /ask`: JSON `{ "question": "...", "use_web_fallback": false }`.
  Questions are trimmed and limited to 1–2000 characters. Extra fields are rejected.
  Query refinement and top-k are not request options.

Responses include `answer`, `question`, `sources`, `web_sources`, `fallback_used`,
`rejected_non_legal`, and `elapsed_seconds`. Retrieved sources contain corpus IDs,
titles, law/article metadata, scores and excerpts capped at 2,000 characters. Uploaded sources also include filename/page provenance.
These identify retrieved evidence; citations in generated prose are not independently
validated. Scores are fused ranking scores, not probabilities.

Web fallback is off by default. Set `use_web_fallback` to true to permit the
existing web search path. Nonlegal questions and empty retrieval return normal
200 responses explaining that the question cannot be answered.

Answering, document indexing and deletion share one worker lock. Concurrent requests receive 429 with
`Retry-After: 5`; dependency/inference errors return a sanitized 502. Health checks
remain responsive during blocking inference. An HTTP disconnect does not forcibly
stop inference; wait for it to finish before retrying. Shutdown waits for inference
and closes the Qdrant client.

The API binds to localhost and does not implement general user authentication or
token-by-token answer streaming. Document tokens only authorize private uploads.
Gradio supports login, progress updates and temporary public links; place the API
behind authentication before exposing it directly. See
[Docker and Langfuse](docker-langfuse.md).

## Gradio UI

Keep FastAPI running and open a second terminal in the repository root:

```powershell
.\.venv\Scripts\python.exe app.py
```

Run `python app.py` if the virtual environment is already activated.
Open http://127.0.0.1:7860 for the chat interface. Gradio sends questions to
`POST /ask` and displays the answer and retrieved source excerpts. It never loads
models, opens Qdrant, or builds indexes. Set `API_BASE_URL` in `.env` to change
the API address (default `http://127.0.0.1:8000`), then restart Gradio.

The UI keeps chat history in the browser session; each request sends only the
current question. There is no conversational memory in retrieval/generation.
Web fallback stays off. Requests have a 5-second connection timeout and a
600-second response timeout with no automatic retries; a timed-out API request
may still be running. Busy and unavailable services produce visible messages.
Native services bind to localhost by default. Gradio sharing is opt-in and requires
credentials; the Docker ports also bind to localhost.

## Documents

- `POST /documents`: multipart `file` and optional boolean `force_ocr=false`.
  `true` forces full-page Vietnamese/English OCR for PDFs, including PDFs with a
  damaged text layer. DOCX uses normal extraction.
- Return value: `token`, `files`, `expires_at`. Send the returned token through
  `X-Document-Token` on later uploads and `/ask`; never use the telemetry session ID.
- `DELETE /documents`: deletes the token's uploads. Clearing chat calls this endpoint.
- 10 MiB/file, 50 PDF pages, three files and 300 chunks/session; 24-hour expiry.
  API restart invalidates sessions. Normal parser timeout is 180 seconds; OCR is
  630 seconds including cleanup. UI response timeout is 210/660 seconds respectively.

Invalid uploads return 422, oversized requests 413, missing Content-Length 411,
busy requests 429 and invalid/expired document capabilities 401. Uploads occupy
the same lock as chat; an HTTP disconnect does not cancel parsing. Raw files are
deleted after parsing. See [document processing and privacy](document-uploads.md).

## Verification

```powershell
python -m unittest discover -s tests -p 'test_api.py'
```

API tests inject a fake pipeline, exercising HTTP validation, startup/shutdown,
readiness failure, concurrency and error handling without downloading models.
Additional tests verify opt-in error propagation in the real pipeline classes.
For live validation, start services and submit a legal question through `/docs`.
