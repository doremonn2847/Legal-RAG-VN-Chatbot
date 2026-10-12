"""Copy saved vectors from embedded/local Qdrant to Docker or Qdrant Cloud."""
import argparse
import os
import json
import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models
from tqdm import tqdm
from config import Config


def upload_index(source, destination, collection=Config.COLLECTION_NAME, payloads=None):
    info = source.get_collection(collection)
    if payloads is not None and len(payloads) != info.points_count:
        raise ValueError("Corpus payload count differs from source vector count")
    if destination.collection_exists(collection):
        existing = destination.get_collection(collection)
        if existing.config.params.vectors != info.config.params.vectors:
            raise ValueError("Destination collection has incompatible vector configuration.")
    else:
        destination.create_collection(collection, vectors_config=info.config.params.vectors)
    offset = None
    with tqdm(total=info.points_count, desc="Transferring vectors") as progress:
        while True:
            points, offset = source.scroll(collection, limit=256, offset=offset, with_payload=payloads is None, with_vectors=True)
            if points:
                destination.upsert(collection, points=[models.PointStruct(
                    id=p.id, vector=p.vector,
                    payload=payloads[str(p.id)] if payloads is not None else p.payload,
                ) for p in points], wait=True)
                progress.update(len(points))
            if offset is None:
                break
    count = destination.count(collection, exact=True).count
    if count != info.points_count:
        raise ValueError(f"Destination contains {count} points; expected {info.points_count}.")
    print(f"Verified {count:,} vectors on the Qdrant server")
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--source", type=Path, default=Path("index/qdrant"))
    sources.add_argument("--source-url", help="Running source Qdrant, e.g. http://127.0.0.1:6333")
    parser.add_argument("--collection", default=Config.COLLECTION_NAME)
    parser.add_argument("--cloud", action="store_true", help="Use QDRANT_CLOUD_URL and QDRANT_CLOUD_API_KEY from .env")
    parser.add_argument("--corpus", type=Path, help="Restore article payloads from this corpus instead of reading source payload storage")
    args = parser.parse_args(argv)
    url = os.getenv("QDRANT_CLOUD_URL", "").strip() if args.cloud else "http://localhost:6333"
    key = os.getenv("QDRANT_CLOUD_API_KEY", "").strip() if args.cloud else None
    if args.cloud and (not url.startswith("https://") or not key):
        parser.error("--cloud requires an HTTPS QDRANT_CLOUD_URL and QDRANT_CLOUD_API_KEY in .env")
    payloads = None
    if args.corpus:
        from utils.data_loader import LegalDataLoader
        loader = LegalDataLoader()
        loader.legal_corpus = json.loads(args.corpus.read_text(encoding="utf-8-sig"))
        payloads = {}
        for doc in loader.prepare_documents_for_indexing():
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{args.collection}/{doc['id']}"))
            if point_id in payloads:
                raise ValueError("Duplicate corpus article ID")
            payloads[point_id] = {"article_id": doc["id"], "title": doc["title"],
                                  "content": doc["content"], "metadata": doc["metadata"]}
    source = (QdrantClient(url=args.source_url, timeout=120) if args.source_url
              else QdrantClient(path=str(args.source)))
    try:
        destination = QdrantClient(url=url, api_key=key, prefer_grpc=args.cloud, timeout=120)
        try:
            upload_index(source, destination, args.collection, payloads)
        finally:
            destination.close()
    finally:
        source.close()


if __name__ == "__main__":
    main()
