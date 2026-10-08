"""Copy an imported embedded Qdrant index to the local Docker server."""
import argparse
from pathlib import Path

from qdrant_client import QdrantClient, models
from tqdm import tqdm
from config import Config


def upload_index(source, destination, collection=Config.COLLECTION_NAME):
    info = source.get_collection(collection)
    if destination.collection_exists(collection):
        existing = destination.get_collection(collection)
        if existing.config.params.vectors != info.config.params.vectors:
            raise ValueError("Destination collection has incompatible vector configuration.")
    else:
        destination.create_collection(collection, vectors_config=info.config.params.vectors)
    offset = None
    with tqdm(total=info.points_count, desc="Transferring vectors") as progress:
        while True:
            points, offset = source.scroll(collection, limit=256, offset=offset, with_payload=True, with_vectors=True)
            if points:
                destination.upsert(collection, points=[models.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points], wait=True)
                progress.update(len(points))
            if offset is None:
                break
    count = destination.count(collection, exact=True).count
    if count != info.points_count:
        raise ValueError(f"Destination contains {count} points; expected {info.points_count}.")
    print(f"Verified {count:,} vectors on the Qdrant server")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("index/qdrant"))
    args = parser.parse_args()
    source = QdrantClient(path=str(args.source))
    destination = QdrantClient(url="http://localhost:6333", prefer_grpc=True, timeout=120)
    try:
        upload_index(source, destination)
    finally:
        source.close()
        destination.close()
