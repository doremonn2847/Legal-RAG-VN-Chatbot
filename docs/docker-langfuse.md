# Docker and Langfuse Cloud

The local stack has three containers: Gradio (`ui`), FastAPI (`api`), and Qdrant.
Ollama stays on the Windows host so the downloaded Qwen model can be reused.
Embedding and BGE reranking run on CPU in the API container; there is no Docker
GPU configuration. Langfuse runs in the Cloud, not in another local container.

## Start the stack

Start Docker Desktop and Ollama. The existing populated Qdrant storage must be
in `index/qdrant-server`, BM25 in `index/bm25_index.pkl`, and stopwords in
`data/utils/stopwords.txt`. Startup never rebuilds the indexes. Stop a native API
or Gradio process using ports 8000/7860 before starting containers on those ports.

```powershell
docker compose up -d --build
docker compose ps
docker compose logs -f api
```

Open http://127.0.0.1:7860 for Gradio and http://127.0.0.1:8000/docs for FastAPI.
Only these localhost ports are exposed, even though each container listens on
all its internal interfaces. The API uses a single worker and rejects concurrent
questions with 429. Gradio waits for API readiness before starting.

The first build downloads Python dependencies and CPU PyTorch. First API startup
downloads embedding/reranker weights into the persistent `model-cache` volume;
this can take several minutes and substantial disk space. Weights are not baked
into the images. BM25 and stopwords are mounted read-only, while Qdrant keeps its
existing host storage. The corpus ZIP, results, virtual environment, and `.env`
are excluded from Docker's build context. Containers run as a non-root user.

Set these values in `.env` when needed:

```dotenv
OLLAMA_MODEL=qwen3.5:9b
DOCKER_OLLAMA_BASE_URL=http://host.docker.internal:11434
API_PORT=8000
UI_PORT=7860
```

Compose overrides `QDRANT_URL` with `http://qdrant:6333` and disables embedded
storage. `API_BASE_URL` is `http://api:8000` inside Gradio. Native-process settings
remain usable outside containers.

If the API cannot reach host Ollama, test the host gateway from a container.
Ollama must listen on an address reachable through that gateway. On Windows,
`OLLAMA_HOST=0.0.0.0:11434` followed by an Ollama restart may be required;
restrict inbound access using Windows Firewall. Do not expose Ollama to the
public internet. Override `DOCKER_OLLAMA_BASE_URL` when using another server.

## Enable Langfuse Cloud

Create a project and API key pair in Langfuse Cloud, then set these in `.env`:

```dotenv
LANGFUSE_TRACING_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-your-project-key
LANGFUSE_SECRET_KEY=sk-lf-your-project-key
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_CAPTURE_CONTENT=true
```

Use your project's region URL (EU shown; US uses `https://us.cloud.langfuse.com`).
Keep `.env` private. Compose injects only the required settings at runtime; keys
are not copied into an image. Recreate the API after changing runtime settings:

```powershell
docker compose up -d --build api ui
```

Integration uses Langfuse Python SDK v4 (verified with 4.17.0) and its LangChain
`CallbackHandler` for actual Ollama calls. Native API runs also support tracing;
restart the native process after changing `.env`.

One question creates an `answer-legal-question` root observation. Nested steps:

- `classify-legal-domain` (guardrail), with an Ollama generation if LLM detection runs.
- `retrieve-context` (retriever), containing `rerank-context` and its article scores.
- `generate-response` (chain), containing an Ollama generation with prompt, model,
  response, parameters and token usage as reported by Ollama.

Original queries, generated answers, selected article IDs, feature settings, timing
and errors are recorded. The root output is the answer, not a raw response blob.
Requests with tracing active return `trace_id`; open that trace in the Cloud UI.
Gradio supplies a random `X-Session-ID` per browser conversation. Clearing chat
starts a new session. This groups traces; it does not add model conversation memory.
Direct API clients may send the same header (1–64 letters, digits, `_` or `-`).

Tracing is disabled by default. Missing keys or telemetry/export failures do not
prevent inference. The client flushes buffered events on API shutdown rather than
blocking each request for upload. Telemetry availability is not an API readiness
dependency. Health checks and rejected busy requests do not generate RAG traces.

## Content and privacy

Export-stage masking covers both custom spans and LangChain generation spans. It
redacts common emails, Vietnamese phone numbers, 12-digit IDs and Langfuse keys.
This is pattern masking, not complete anonymization: personal names, addresses,
and unusual number formats may remain. Avoid confidential questions when content
capture is enabled. Set `LANGFUSE_CAPTURE_CONTENT=false` and restart the API to
redact observation input/output entirely while retaining timing, model and usage
data. Application console logs are separate from Langfuse masking.

Local Ollama token usage measures context consumption, not provider spend. Do not
interpret automatically assigned model prices as money paid to a hosted provider.

## Stop and troubleshoot

```powershell
docker compose stop ui api
docker compose up -d api ui
```

`stop` preserves containers and all data. Avoid `down -v`, which deletes named
volumes such as the model cache. Qdrant's host folder is independent of that cache.
If startup fails, inspect `docker compose logs api` and `/ready`. Verify indexes,
Ollama model availability, host connectivity and free RAM. Do not run the native
RAG API alongside its container on a memory-constrained laptop.

Tracing follows the official [Langfuse skill](https://github.com/langfuse/skills/tree/main/skills/langfuse),
[best practices](https://langfuse.com/docs/observability/best-practices), and
[LangChain integration](https://langfuse.com/integrations/frameworks/langchain).

## Audit a live trace

With Node.js/npx installed, run the provided check using the `trace_id` returned
by `/ask`:

```powershell
python tools/verify_langfuse_trace.py TRACE_ID
```

This uses the official Langfuse CLI's modern observations endpoint and checks
the root input/output, conversation grouping, nested stages, model names and
nonzero input/output tokens. It is intended for a successful legal question with
content capture enabled. Nonlegal rejections intentionally skip retrieval and
answer generation. The saved audit is `results/langfuse-live-audit.json`.
