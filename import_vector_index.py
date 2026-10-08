"""Validate and import a Colab-built local Qdrant index, preserving any old index."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from zipfile import ZipFile

from qdrant_client import QdrantClient
from config import Config


def import_index(archive_path: Path, destination: Path = None):
    root = Path(__file__).resolve().parent
    index_root = (root / "index").resolve()
    target = (destination or root / (Config.QDRANT_PATH or "index/qdrant")).resolve()
    if not target.is_relative_to(index_root) or target == index_root:
        raise ValueError("The destination must be a directory inside the project's index directory.")
    index_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="incoming-", dir=index_root) as temporary:
        staging = Path(temporary)
        with ZipFile(archive_path) as archive:
            names = archive.namelist()
            for name in names:
                path = PurePosixPath(name)
                if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
                    raise ValueError("Unsafe path in index archive.")
                if name != "build_manifest.json" and not name.startswith("qdrant/"):
                    raise ValueError(f"Unexpected file in index archive: {name}")
            archive.extractall(staging)
        manifest = json.loads((staging / "build_manifest.json").read_text(encoding="utf-8"))
        if manifest["embedding_model"] != Config.EMBEDDING_MODEL or manifest["collection"] != Config.COLLECTION_NAME:
            raise ValueError("The index uses a different embedding model or collection.")
        dataset_manifest = root / "data/dataset_manifest.json"
        if dataset_manifest.exists():
            local = json.loads(dataset_manifest.read_text(encoding="utf-8"))
            if local["archive_sha256"] != manifest["dataset_archive_sha256"]:
                raise ValueError("The index was built from a different dataset archive.")
            if local["indexable_articles"] != manifest["article_count"]:
                raise ValueError("The index is incomplete for the local dataset.")
        incoming = staging / "qdrant"
        client = QdrantClient(path=str(incoming))
        try:
            info = client.get_collection(Config.COLLECTION_NAME)
            if info.points_count != manifest["article_count"] or info.config.params.vectors.size != manifest["embedding_dimensions"]:
                raise ValueError("Vector count or dimensions differ from the build manifest.")
        finally:
            client.close()
        backup = None
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            backup = target.with_name(target.name + ".backup-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
            if not backup.resolve().is_relative_to(index_root):
                raise ValueError("Backup path is outside the index directory.")
            target.rename(backup)
        try:
            incoming.rename(target)
        except Exception:
            if backup:
                backup.rename(target)
            raise
        shutil.copyfile(staging / "build_manifest.json", index_root / "build_manifest.json")
    print(f"Imported {manifest['article_count']:,} vectors to {target}")
    if backup:
        print(f"Previous index preserved at {backup}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    import_index(args.archive)
