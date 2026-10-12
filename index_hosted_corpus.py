"""Estimate or resume hosted BGE-M3 corpus indexing without loading the RAG/LLM."""
import argparse
import hashlib
import json
import math
import os
import uuid
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv


def pending_documents(store, documents):
    """Read checkpoint IDs without downloading stored texts or vectors."""
    existing_ids = set()
    offset = None
    while True:
        points, offset = store.client.scroll(
            collection_name=store.collection_name, limit=10000, offset=offset,
            with_payload=False, with_vectors=False,
        )
        existing_ids.update(str(point.id) for point in points)
        if offset is None:
            break
    pending = [doc for doc in documents if str(uuid.uuid5(
        uuid.NAMESPACE_URL, f"{store.collection_name}/{doc.get('id', '')}"
    )) not in existing_ids]
    print(f"Resume: {len(documents) - len(pending):,} saved articles skipped; {len(pending):,} remaining.")
    return pending


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus/legal_corpus.json"))
    parser.add_argument("--run", action="store_true", help="Send the corpus to the provider; default only estimates usage")
    args = parser.parse_args()
    load_dotenv(args.env_file)
    os.environ.setdefault("EMBEDDING_PROVIDER", "hosted")
    from config import Config
    from utils.data_loader import LegalDataLoader

    if Config.EMBEDDING_PROVIDER != "hosted":
        raise ValueError("This script requires EMBEDDING_PROVIDER=hosted")
    Config.CORPUS_PATH = str(args.corpus.resolve())
    documents = LegalDataLoader().prepare_documents_for_indexing()
    if not documents:
        raise ValueError("The corpus is missing or contains no indexable articles")
    print(json.dumps({"collection": Config.COLLECTION_NAME, "model": Config.EMBEDDING_MODEL,
                      "dimensions": Config.EMBEDDING_DIMENSIONS, "articles": len(documents),
                      "input_characters": sum(len(d["content"]) for d in documents),
                      "longest_article_characters": max(len(d["content"]) for d in documents),
                      "full_build_requests": math.ceil(len(documents) / Config.INDEX_BATCH_SIZE)}, indent=2))
    if not args.run:
        print("Estimate only: no provider calls or index changes. Add --run to begin/resume.")
        return

    import requests
    from utils.hosted_embeddings import HostedEmbeddings
    HostedEmbeddings(Config.EMBEDDING_BASE_URL, Config.EMBEDDING_MODEL, Config.EMBEDDING_DIMENSIONS, Config.EMBEDDING_API_KEY)
    url = urlsplit(Config.EMBEDDING_BASE_URL)
    try:
        response = requests.get(f"{url.scheme}://{url.netloc}/info", headers={"Authorization": "Bearer " + Config.EMBEDDING_API_KEY}, timeout=(10, 30))
    except requests.RequestException:
        raise RuntimeError("Provider metadata check failed") from None
    if response.status_code != 200:
        raise RuntimeError(f"Provider metadata check returned HTTP {response.status_code}")
    info = response.json()
    if info.get("model_id") != Config.EMBEDDING_MODEL or "embedding" not in info.get("model_type", {}):
        raise ValueError("Provider serves a different model or is not an embedding server")

    from main.vector_store import QdrantVectorStore
    store = QdrantVectorStore()
    try:
        # A manifest prevents an interrupted build from mixing corpora or providers.
        with args.corpus.open("rb") as corpus_file:
            corpus_hash = hashlib.file_digest(corpus_file, "sha256").hexdigest()
        manifest = {"embedding_provider": "hosted", "base_url": Config.EMBEDDING_BASE_URL.rstrip("/"),
                    "model": Config.EMBEDDING_MODEL, "dimensions": Config.EMBEDDING_DIMENSIONS,
                    "collection": Config.COLLECTION_NAME, "articles": len(documents),
                    "corpus_sha256": corpus_hash,
                    "max_input_length": info.get("max_input_length"), "auto_truncate": info.get("auto_truncate"),
                    "model_sha": info.get("model_sha")}
        path = Path("index") / (Config.COLLECTION_NAME + ".manifest.json")
        path.parent.mkdir(exist_ok=True)
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != manifest:
                raise ValueError("Existing indexing manifest differs; use a separate checkout/collection for changed inputs")
        elif store.client.collection_exists(store.collection_name):
            raise ValueError("Collection exists without this build manifest; refusing to mix indexes")
        # Validate dimensions before writing the manifest or inserting any articles.
        probe = store.embed_text("Kiểm tra kết nối lập chỉ mục pháp luật Việt Nam.")
        if len(probe) != Config.EMBEDDING_DIMENSIONS:
            raise ValueError("Provider embedding dimensions do not match configuration")
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        store.create_collection()
        store.add_documents(pending_documents(store, documents))
        count = store.client.count(store.collection_name, exact=True).count
        if count != len(documents):
            raise RuntimeError(f"Index contains {count} articles; expected {len(documents)}")
        print(f"Verified complete hosted index: {count:,} articles. BM25 and the original collection were not modified.")
    finally:
        store.client.close()


if __name__ == "__main__":
    main()
