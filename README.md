# Legal RAG Vietnamese Chatbot

A Vietnamese legal chatbot using hybrid BM25/Qdrant retrieval, BGE reranking,
and local Qwen generation through Ollama. FastAPI serves questions; Gradio
provides the chat interface and source excerpts.

## How it works

The legal-domain filter checks each question. Retrieval uses the original
question, collects up to 25 sparse and 25 dense candidates, deduplicates article
IDs, and selects five articles using BGE reranking and score fusion. Qwen answers
from those excerpts, with instructions to cite evidence and acknowledge missing
information. Query rewriting is disabled. Web fallback is opt-in through the API
and disabled in Gradio. Chat history is displayed but is not sent to the model.

The embedding model is `bkai-foundation-models/vietnamese-bi-encoder`; the reranker
is `BAAI/bge-reranker-v2-m3`. Full articles remain in the indexes; reranking and
answer context use excerpts capped at 1,000 characters plus a truncation ellipsis.

## Setup

Use Python 3.11+, [Ollama](https://ollama.com/download), and Docker with Compose.
Run commands from the repository root.

```bash
git clone https://github.com/doremonn2847/Legal-RAG-VN-Chatbot.git
cd Legal-RAG-VN-Chatbot
python -m venv .venv
```

Activate the environment with `.\.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` on Linux/macOS, then install dependencies:

```bash
python -m pip install -r requirements.txt
ollama pull qwen3.5:9b
```

Copy `.env.example` to `.env`. The defaults use local Ollama and Qdrant on
`localhost:6333`; leave `QDRANT_PATH`, `QDRANT_URL`, and `QDRANT_API_KEY` empty
for that configuration. Set `OLLAMA_MODEL` to change the downloaded model.
No hosted inference API key is required.

Download the [Zalo AI 2021 legal corpus](https://www.kaggle.com/datasets/hariwh0/zaloai2021-legal-text-retrieval/data)
and build the indexes:

```bash
python download_dataset.py
docker compose up -d qdrant
python setup_system.py
```

The downloader prepares `data/corpus/legal_corpus.json`, stopwords, and the
source dataset files, recording their counts and archive hash in
`data/dataset_manifest.json`. Indexing stores one vector per nonempty article
and saves `index/bm25_index.pkl`. Interrupted vector builds resume on rerun;
BM25 is saved after its build completes. First use downloads model weights.
Building all 61,068 article vectors can take hours on CPU.

`python setup_system.py --rebuild` replaces both indexes. Stop the API/UI and
retain a backup before using it. Normal API startup only loads existing indexes.

### Optional GPU indexing

Run [notebooks/colab_indexing.ipynb](notebooks/colab_indexing.ipynb) with a Colab
GPU to build a portable vector index. Regenerate its source snapshot after code
changes with `python tools/create_colab_notebook.py`.

Download the resulting ZIP, stop processes using the embedded index, and import
and transfer it to local Qdrant:

```bash
python import_vector_index.py colab_vector_index.zip
docker compose up -d qdrant
python upload_vector_index.py
python setup_system.py
```

Import verifies the build manifest and preserves the previous embedded index as
a backup. Setup builds BM25 if needed and retains a complete vector collection.
Only load trusted index files: the BM25 index uses Python pickle.

## Run

With Ollama running and indexes prepared:

```bash
docker compose up -d --build
```

Open [Gradio](http://127.0.0.1:7860) or the
[interactive API documentation](http://127.0.0.1:8000/docs).
Docker runs API, UI, and Qdrant; Ollama runs on the host. Embeddings and reranking
use CPU in the API container. See [Docker and optional Langfuse tracing](docs/docker-langfuse.md)
for configuration, model caching, content privacy, and troubleshooting.

For native serving, run these commands in separate terminals with the environment
activated, while Qdrant and Ollama are running:

```bash
python -m uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1
python app.py
```

The API exposes `/health`, `/ready`, and `/ask`, validates input, and processes
one inference request at a time. Concurrent questions receive HTTP 429.
See [the API guide](docs/api.md) for request and response details.

## Project layout

- `api.py`, `app.py`: HTTP service and Gradio interface.
- `main/`: chatbot orchestration, sparse/dense retrieval, and reranking.
- `utils/`: text/data processing, legal-domain detection, web search, and tracing.
- `config.py`, `css/`: model/pipeline settings and existing interface styles.
- `download_dataset.py`, `setup_system.py`: dataset preparation and indexing.
- `import_vector_index.py`, `upload_vector_index.py`: portable index transfer.
- `notebooks/`, `tools/`: GPU indexing and live trace verification.
- `docs/`, `tests/`: operational guides and offline regression tests.

Local datasets, indexes, results, secrets, and virtual environments are excluded
from Git. Historical experiments and reports remain available in Git history.

## Verification

The offline tests use test doubles and temporary indexes, without downloading
models or requiring live services. Disable tracing and use UTF-8 for Vietnamese
text and console symbols:

```powershell
$env:LANGFUSE_TRACING_ENABLED='false'
$env:PYTHONIOENCODING='utf-8'
python -m unittest discover -s tests -v
```

On Linux/macOS:

```bash
LANGFUSE_TRACING_ENABLED=false PYTHONIOENCODING=utf-8 python -m unittest discover -s tests -v
```

These checks cover API/UI behavior, query routing, retrieval limits, dataset and
index handling, dependency compatibility, and telemetry. Live model quality and
citation faithfulness require separate review.

## License

[MIT](LICENSE).
