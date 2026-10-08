"""Package the current source into a self-contained Colab GPU indexing notebook."""
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
from zipfile import ZipFile, ZIP_DEFLATED

root = Path(__file__).resolve().parents[1]
paths = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=root, text=True).splitlines()
buffer = io.BytesIO()
with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
    for name in sorted(set(paths)):
        if name.startswith(("notebooks/", "artifacts/")) or name == ".env" or name.endswith((".zip", ".ipynb")):
            continue
        path = root / name
        if path.is_file():
            archive.writestr(name, path.read_bytes())
payload = buffer.getvalue()
encoded = base64.b64encode(payload).decode()
digest = hashlib.sha256(payload).hexdigest()
cells = []

def markdown(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})

def code(text, hidden=False):
    cells.append({"cell_type": "code", "execution_count": None, "outputs": [],
                  "metadata": {"jupyter": {"source_hidden": True}} if hidden else {},
                  "source": text.strip().splitlines(keepends=True)})

markdown("""# Build the Vietnamese legal vector index on a Colab GPU

In VS Code choose **Select Kernel → Colab**, sign in, and connect to a **GPU** runtime (a T4 is sufficient). Then run the cells in order.

This notebook restores a verified snapshot of the project source. It embeds no `.env`, API key, model weights, or dataset. No inference API key is needed for indexing.

The result is `colab_vector_index.zip`, containing the complete portable Qdrant index and a build manifest. Download it through the Colab file browser or the final download cell. On your local machine, stop the chatbot and run `python import_vector_index.py <downloaded zip path>`. The existing partial index is backed up. The separately built local BM25 index is retained.
""")
code(f'''
from pathlib import Path
import base64, hashlib, io, os, zipfile
SOURCE_SHA256 = {digest!r}
SOURCE_BASE64 = {encoded!r}
source = base64.b64decode(SOURCE_BASE64)
assert hashlib.sha256(source).hexdigest() == SOURCE_SHA256
REPO = Path("/content/vietnamese-legal-chatbot")
REPO.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(source)) as archive:
    archive.extractall(REPO)
os.chdir(REPO)
print("Current source restored:", SOURCE_SHA256)
''', hidden=True)
markdown("## Install dependencies and download the dataset")
code('''
import sys, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
subprocess.run([sys.executable, "download_dataset.py"], check=True)
''')
markdown("## Verify the GPU and build all 61,068 article vectors")
code('''
import os, time, torch
os.environ["QDRANT_PATH"] = "index/qdrant"
os.environ.pop("QDRANT_URL", None)
os.environ.pop("QDRANT_API_KEY", None)
assert torch.cuda.is_available(), "Connect a GPU Colab runtime before indexing."
print("GPU:", torch.cuda.get_device_name(0))
from config import Config
from main.vector_store import QdrantVectorStore
from utils.data_loader import LegalDataLoader
Config.QDRANT_PATH = "index/qdrant"
Config.QDRANT_URL = Config.QDRANT_API_KEY = None
Config.EMBEDDING_BATCH_SIZE = 64
documents = LegalDataLoader().prepare_documents_for_indexing()
store = QdrantVectorStore()
assert store.embedding_model.device.type == "cuda", "Embedding model must use the GPU."
dimensions = store.embedding_model.get_embedding_dimension()
store.create_collection(vector_size=dimensions)
started = time.time()
store.add_documents(documents)
info = store.get_collection_info()
assert info["points_count"] == len(documents), "Index is incomplete. Rerun this cell to resume."
print("Indexed", len(documents), "articles in", round(time.time() - started, 1), "seconds")
''')
markdown("## Check dense retrieval and the multilingual BGE reranker")
code('''
from main.reranker import DocumentReranker
query = "Người lao động có quyền nhận lương đúng hạn không?"
candidates = store.search_similar_documents(query, top_k=20)
assert candidates, "Dense retrieval returned no documents."
reranker = DocumentReranker()
assert reranker.model is not None, "BGE reranker failed to load."
ranked = reranker.rerank_with_fusion(query, candidates, alpha=Config.RERANKER_FUSION_ALPHA, top_k=5)
assert ranked and all("reranker_score" in doc for doc in ranked)
for doc in ranked:
    print(doc["id"], round(doc["score"], 4), doc["title"])
''')
markdown("## Export the index and its build manifest")
code('''
import importlib.metadata, json, zipfile
dataset_manifest = json.loads(Path("data/dataset_manifest.json").read_text())
manifest = {
    "embedding_model": Config.EMBEDDING_MODEL,
    "embedding_dimensions": dimensions,
    "collection": Config.COLLECTION_NAME,
    "article_count": len(documents),
    "dataset_archive_sha256": dataset_manifest["archive_sha256"],
    "source_sha256": SOURCE_SHA256,
    "reranker_model": Config.RERANKER_MODEL,
    "qdrant_client_version": importlib.metadata.version("qdrant-client"),
    "sentence_transformers_version": importlib.metadata.version("sentence-transformers"),
    "smoke_test_article_ids": [doc["id"] for doc in ranked],
}
store.client.close()  # Flush SQLite and release the local-store lock before exporting.
manifest_file = Path("index/build_manifest.json")
manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
EXPORT = REPO / "colab_vector_index.zip"
with zipfile.ZipFile(EXPORT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
    archive.write(manifest_file, "build_manifest.json")
    for file in Path(Config.QDRANT_PATH).rglob("*"):
        if file.is_file() and file.name != ".lock":
            archive.write(file, "qdrant/" + file.relative_to(Config.QDRANT_PATH).as_posix())
print("Download:", EXPORT, "Size MB:", round(EXPORT.stat().st_size / 1024**2, 1))
print(json.dumps(manifest, indent=2))
''')
markdown("## Download to your local machine\nIf the VS Code output cannot trigger a download, use the Colab runtime file browser to download `/content/vietnamese-legal-chatbot/colab_vector_index.zip`.")
code('''
from google.colab import files
files.download(str(EXPORT))
''')
notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}, "colab": {"name": "colab_indexing.ipynb"}, "accelerator": "GPU"}}
for i, cell in enumerate(cells):
    cell["id"] = f"index-cell-{i}"
destination = root / "notebooks/colab_indexing.ipynb"
destination.parent.mkdir(exist_ok=True)
destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("Created:", destination, "Source bytes:", len(payload), "SHA256:", digest)
