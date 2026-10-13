# Hosted demo deployment on this laptop

The current API/UI build from **main** in the project checkout. The hosted branch has been merged. The demo uses DeepSeek Flash, hosted BGE-M3 embeddings, hosted BGE reranking and Qdrant Cloud collection `hosted_baai_bge_m3_1024` (61,068 articles). Retrieval uses the original question and final top-5; query refinement is disabled. Private PDF/DOCX uploads and optional Vietnamese full-page OCR are available.

## Configure and restart

Use `.env` inside the main project directory, not the former worktree. Set `INFERENCE_PROVIDER=deepseek`, `EMBEDDING_PROVIDER=hosted`, `RERANK_PROVIDER=hosted`, BGE-M3 model/dimensions and their private API keys as shown in [hosted retrieval setup](hosted-retrieval.md). Set `QDRANT_CLOUD_URL`/`QDRANT_CLOUD_API_KEY` for Compose and matching `QDRANT_URL`/`QDRANT_API_KEY` for native indexing. BM25 and stopwords remain local trusted files.

```bash
docker compose up -d --build api ui
docker compose ps
docker compose logs -f api
```

Open [Gradio](http://127.0.0.1:7860/) or [FastAPI docs](http://127.0.0.1:8000/docs). Set `DEMO_USERNAME`/`DEMO_PASSWORD` for UI login. Hosted mode does not require host Ollama. OCR and BM25 still consume laptop CPU; provider calls require network/quota and can incur charges.

This laptop currently tags its production images `legal-document-api` and `legal-document-ui` and uses an ignored `results/document-demo.override.yaml` plus restart helper. Those local helpers are not required by a fresh clone; the standard Compose build above uses the same source. Avoid old worktree overrides when rebuilding main.

## Persistence and stopping

```bash
docker compose stop ui api
```

Cloud vectors and the corpus BM25 index survive stopping the app. Docling/model caches live in named Docker volumes. Document capability registries are in API memory: restarting the API invalidates upload sessions even though expired chunks are physically cleaned later. Users must re-upload. Clear chat deletes the current session's uploads.

Compose still starts a local Qdrant container through its dependency chain; configured Cloud credentials route the app to Cloud. The old local corpus storage has a known payload-corruption problem in this laptop deployment and should not be used as a fallback. The Cloud corpus was verified separately; unrelated Cloud collections are not deleted by application startup.

## Public hosting status

Local demo: [http://127.0.0.1:7860/](http://127.0.0.1:7860/). A Cloudflare quick tunnel or Gradio sharing can expose a temporary password-protected UI, but requires the laptop and tunnel to remain running. No currently verified permanent public demo URL is documented.

The [Hugging Face Space](https://huggingface.co/spaces/doremonn2847/Vietnamese-Legal-Chatbot) exists but is **not operational**. The account's ZeroGPU runtime requires a genuine GPU function; this app is CPU/server-based, and switching to CPU Basic returned a payment-required response. No paid upgrade or artificial GPU function was added. Qdrant Cloud alone does not host the API/UI.

## Verification

The Cloud corpus contains 61,068 indexed article vectors with rebuilt full payloads verified against the source corpus. The [completed hosted benchmark](hosted-benchmark-comparison.md) evaluates all 3,196 training questions; it is a whole-stack comparison, not held-out answer accuracy. The README Results tables now report this hosted benchmark; the comparison report retains the earlier local-stack scores.

Live document checks verified upload, combined retrieval/reranking, DeepSeek answering, rejected invalid tokens and deletion. Vietnamese OCR on the supplied 13-page PDF repaired the selected Điều 6 heading in 113.2 seconds; other words still contain recognition errors. See [document verification and limits](document-upload-verification.md) and [Langfuse setup](docker-langfuse.md).
