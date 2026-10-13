# Hosted retrieval and indexing

Integrated into **main** from `codex/hosted-retrieval`. The existing bi-encoder collection, BM25 index,
selectable Qwen/DeepSeek generator, original-query behavior, top-25 candidate pools, top-5 answer
evidence, and 2,000-character source excerpts are retained. The new embedding collection
is `hosted_baai_bge_m3_1024`.

## Verified provider capability

On October 12, 2026, authenticated checks against `https://api.dek.io.vn` returned:

- `/info`: `BAAI/bge-m3`, embedding model, float16, maximum input 8,192 tokens,
  automatic truncation enabled, maximum client batch size 128.
- `/v1/embeddings`: HTTP 200, 1,024-dimensional vector, model `BAAI/bge-m3`.
- `/rerank`: HTTP 200 after the provider routing fix, returning indexed raw scores.

Hosted embeddings and reranking now both work. A two-document reranking smoke
check scored the relevant legal text 4.328125 and unrelated text -10.9765625.
`/info` still describes the embedding backend. The host owner confirmed the
reranker is `BAAI/bge-reranker-v2-m3`; that identity comes from the owner rather
than the embedding metadata. The rerank API is provider-managed,
so a local model-name setting does not select its backend. `/v1/rerank` is not
available; use the full URL `https://api.dek.io.vn/rerank`.

## Configure

Keep the key in a private `.env`; never commit it:

```dotenv
INFERENCE_PROVIDER=deepseek
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
DEEPSEEK_API_KEY=your-private-key
EMBEDDING_PROVIDER=hosted
EMBEDDING_BASE_URL=https://api.dek.io.vn/v1/
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIMENSIONS=1024
EMBEDDING_API_KEY=your-private-key
RERANK_PROVIDER=hosted
RERANK_URL=https://api.dek.io.vn/rerank
RERANK_API_KEY=
QDRANT_PATH=
QDRANT_URL=
QDRANT_API_KEY=
```

For Qdrant Cloud, native scripts use `QDRANT_URL`/`QDRANT_API_KEY`. Compose
instead forwards `QDRANT_CLOUD_URL`/`QDRANT_CLOUD_API_KEY`; set both pairs to the
same cluster in `.env`. With Cloud configured the demo uses the remote collection.
The client must have permissions to create collections/payload indexes, search,
upsert and delete private uploads. Empty Qdrant settings use the local server on port 6333. Without
`EMBEDDING_PROVIDER=hosted`, the original local embedding pipeline remains the
default. Hosted embedding mode never loads local embedding weights.
`RERANK_PROVIDER=hosted` also avoids loading local reranker weights; an empty
`RERANK_API_KEY` reuses `EMBEDDING_API_KEY`. Set `RERANK_PROVIDER=local` for the
original local BGE reranker. The remote adapter sends up to 50 deduplicated
corpus candidates (plus up to 25 private upload chunks when present) in one
request, passes full title-plus-content text (the provider still enforces its token limit), restores
response scores to input order, validates finite scores and complete indexes,
and retains the existing alpha-0.8 fusion and final top-5. It requests raw scores
rather than applying a new probability threshold. Temporary HTTP failures have
bounded retries, and live API request failures propagate rather than silently
falling back to local inference. Set `INFERENCE_PROVIDER=deepseek` to use
`langchain-deepseek` for answers and legal-domain detection. Query refinement
stays disabled. Thinking is explicitly disabled, output is capped at 2,048
tokens, requests time out after 120 seconds with at most two retries. Existing
Langfuse LangChain callbacks remain attached to both inference paths.

The original credential file is exactly
`C:\Users\Lenovo\vietnamese-legal-chatbot\.env` (a file named `.env` inside
the project directory). The current demo builds from main and uses that file. Former worktrees may have
separate stale environment files; do not use them to configure main. Keep keys out of Git.

DeepSeek `/models` listed `deepseek-flash` and `deepseek-v4-pro` on October 12,
2026. A short live `ChatDeepSeek` call using `deepseek-flash` succeeded with
37 input and 26 output tokens; this verifies connectivity, not legal-answer
quality. `/ready` checks authenticated model availability without generating
answers, then verifies matching Qdrant/BM25 counts. It does not contact Ollama
in DeepSeek mode. Set `INFERENCE_PROVIDER=ollama` to restore local inference.
Without an explicit provider, Ollama remains the default for compatibility.

After indexing and BM25 preparation complete, run `docker compose up -d --build api ui` from this checkout
with its `.env`, BM25 index and data files present. Compose forwards the DeepSeek
settings to the API container. The Gradio UI continues to call the same API.
Restart/recreate the API after changing environment settings.

## Estimate and index

Activate the existing Python environment and run from the main project checkout:

```powershell
python index_hosted_corpus.py --env-file C:/Users/Lenovo/vietnamese-legal-chatbot/.env --corpus C:/Users/Lenovo/vietnamese-legal-chatbot/data/corpus/legal_corpus.json
```

This defaults to an estimate: no API calls, collection creation, or vector writes.
The script selects hosted mode unless explicitly configured otherwise. Current
corpus: 61,068 articles, 79,756,393 input characters, longest article 252,881
characters, approximately 3,817 requests at 16 articles per request. This is a
request/character estimate, **not a token or monetary cost estimate**. The
provider's billing, quota, rate limits, and available capacity are unverified.

After checking quota/cost and ensuring Qdrant is running, add `--run`:

```powershell
python index_hosted_corpus.py --env-file C:/Users/Lenovo/vietnamese-legal-chatbot/.env --corpus C:/Users/Lenovo/vietnamese-legal-chatbot/data/corpus/legal_corpus.json --run
```

The script validates provider metadata and vector dimensions, creates only the
new collection, embeds batches of 16, and writes each batch before continuing.
Stable article IDs make reruns skip completed points. It saves a manifest under
`index/hosted_baai_bge_m3_1024.manifest.json` containing the provider/model,
dimensions, corpus SHA-256, article count, and truncation settings. Keep that
manifest to resume in this checkout. A different corpus/provider or an existing
unmanifested collection is rejected. There is no automatic destructive rebuild.
No Qwen, local embedding model, reranker, or BM25 build is loaded by this script.
Run `python setup_system.py` afterward to build/load BM25 and reuse the completed
vector collection; keep the same provider and Qdrant target settings.

The provider truncates articles beyond its 8,192-token limit; the full article is
still stored in Qdrant payloads. This retains article-level retrieval rather than
introducing chunks. The previous bi-encoder also truncated inputs, but the token
limits differ, so this is part of the model comparison. The HTTP client validates
returned dimensions, finite values, count, and result indexes; 429/temporary
gateway failures have bounded retries. Connection failures stop the run for an
explicit resume.

## Application and comparison

Use the hosted environment settings for the API after the complete collection
exists. The existing BM25 pickle and stopwords must be available in the checkout
(or their Docker mounts must point at the original files). An isolated Git
checkout does not copy ignored datasets, credentials, or indexes automatically.
Do not start another CPU-heavy API beside the main one on a memory-constrained
laptop; select separate ports when testing both services.

Retain the existing original-query protocol, final top-5, and gold article IDs.
The configured dense cosine threshold remains 0.25;
it is not known to be calibrated for BGE-M3. Record it in any comparison and
separate later tuning from a fixed-setting model swap. Report retrieval Recall@5,
nDCG@5, MRR@5, request usage, latency and failures independently from answer
generation. The README Results tables now describe the completed hosted benchmark;
the comparison report retains the original pipeline scores. Full-corpus indexing is complete at 61,068 articles.
The [end-to-end smoke check](hosted-pipeline-smoke.md) passed functional checks,
with one gold-article hit across two selected legal questions. The
[full training-set benchmark](hosted-benchmark-comparison.md) is now complete
across all 3,196 questions; it excludes generation and reports limitations.

## Private document uploads

Main also supports [Docling uploads and optional Vietnamese OCR](document-uploads.md).
Upload chunks go to a separate token-filtered, expiring Cloud collection; corpus
article indexing and BM25 are unchanged. Uploaded chunks join the corpus candidates
for one combined rerank. Hosted generation and telemetry may receive their text.
