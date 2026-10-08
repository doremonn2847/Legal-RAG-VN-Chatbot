# Local FastAPI service

The API reuses the existing original-query hybrid retrieval, BGE reranking and
Qwen/Ollama answer generator. Answers use up to five retrieved articles. Legal
domain detection remains enabled and may make an additional Ollama call.

## Start

Run from the repository root, with the existing virtual environment activated:

```powershell
python -m pip install -r requirements.txt
docker compose up -d qdrant
ollama pull qwen3.5:9b
python -m uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1
```

Ollama must be running. Configure `OLLAMA_BASE_URL` and `OLLAMA_MODEL` in `.env`
if using another Ollama server/model. Keep `ENABLE_RERANKING=True` in `config.py`.
The populated Qdrant collection and trusted `index/bm25_index.pkl` must already
exist and have matching nonzero document counts. Starting an empty Docker volume
does not import the collection: use the project's existing index transfer workflow
first. The API never rebuilds or deletes indexes. A pickle is executable data;
only load your own trusted project index.

Startup loads the embedding and reranker models once and checks dependencies.
Missing cached weights may be downloaded. Startup fails if required components,
indexes, or the configured Ollama model are unavailable. Use one worker to avoid
duplicating large models and overlapping inference. CPU inference can be slow.

Open http://127.0.0.1:8000/docs to try requests interactively.

```powershell
$body = @{ question = 'Người lao động được nghỉ phép năm bao nhiêu ngày?' } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/ask -ContentType 'application/json; charset=utf-8' -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

## Interface

- `GET /health`: lightweight process liveness.
- `GET /ready`: checks Qdrant collection and Ollama model availability; returns
  503 when dependencies are unavailable. This does not prove generation succeeds.
- `POST /ask`: JSON `{ "question": "...", "use_web_fallback": false }`.
  Questions are trimmed and limited to 1–2000 characters. Extra fields are rejected.
  Query refinement and top-k are not request options.

Responses include `answer`, `question`, `sources`, `web_sources`, `fallback_used`,
`rejected_non_legal`, and `elapsed_seconds`. Retrieved sources contain corpus IDs,
titles, law/article metadata, scores and excerpts capped at 1000 characters.
These identify retrieved evidence; citations in generated prose are not independently
validated. Scores are fused ranking scores, not probabilities.

Web fallback is off by default. Set `use_web_fallback` to true to permit the
existing web search path. Nonlegal questions and empty retrieval return normal
200 responses explaining that the question cannot be answered.

Only one question is processed at a time. Concurrent requests receive 429 with
`Retry-After: 5`; dependency/inference errors return a sanitized 502. Health checks
remain responsive during blocking inference. An HTTP disconnect does not forcibly
stop inference; wait for it to finish before retrying. Shutdown waits for inference
and closes the Qdrant client.

This first service binds to localhost. Authentication, Docker packaging of the
API and UI containerization and Langfuse Cloud tracing are described in
[Docker and Langfuse](docker-langfuse.md). Authentication and streaming remain
subsequent work.

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
Both services bind to localhost and the UI does not create a public share link.

## Verification

```powershell
python -m unittest discover -s tests -p 'test_api.py'
```

API tests inject a fake pipeline, exercising HTTP validation, startup/shutdown,
readiness failure, concurrency and error handling without downloading models.
Additional tests verify opt-in error propagation in the real pipeline classes.
For live validation, start services and submit a legal question through `/docs`.
