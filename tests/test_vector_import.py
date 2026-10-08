import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from config import Config
import import_vector_index


class VectorImportTests(unittest.TestCase):
    def make_archive(self, directory):
        source = directory / "source"
        client = QdrantClient(path=str(source))
        client.create_collection(Config.COLLECTION_NAME, vectors_config=VectorParams(size=2, distance=Distance.COSINE))
        client.upsert(Config.COLLECTION_NAME, points=[PointStruct(id=1, vector=[1.0, 0.0])])
        client.close()
        manifest = {"embedding_model": Config.EMBEDDING_MODEL, "collection": Config.COLLECTION_NAME,
                    "article_count": 1, "embedding_dimensions": 2, "dataset_archive_sha256": "test-dataset"}
        archive_path = directory / "export.zip"
        with ZipFile(archive_path, "w") as archive:
            archive.writestr("build_manifest.json", json.dumps(manifest))
            for file in source.rglob("*"):
                if file.is_file() and file.name != ".lock":
                    archive.write(file, "qdrant/" + file.relative_to(source).as_posix())
        root = directory / "project"
        (root / "data").mkdir(parents=True)
        (root / "data/dataset_manifest.json").write_text(json.dumps({"archive_sha256": "test-dataset", "indexable_articles": 1}))
        return archive_path, root

    def test_import_reopens_index_and_preserves_previous_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive, root = self.make_archive(Path(temporary))
            target = root / "index/qdrant"
            target.mkdir(parents=True)
            (target / "previous.txt").write_text("old index")
            with patch.object(import_vector_index, "__file__", str(root / "import_vector_index.py")):
                manifest = import_vector_index.import_index(archive, target)
            client = QdrantClient(path=str(target))
            self.assertEqual(client.count(Config.COLLECTION_NAME).count, 1)
            client.close()
            backups = list((root / "index").glob("qdrant.backup-*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual((backups[0] / "previous.txt").read_text(), "old index")
            self.assertEqual(manifest["article_count"], 1)

    def test_wrong_dataset_does_not_replace_existing_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive, root = self.make_archive(Path(temporary))
            (root / "data/dataset_manifest.json").write_text(json.dumps({"archive_sha256": "different", "indexable_articles": 1}))
            target = root / "index/qdrant"
            target.mkdir(parents=True)
            (target / "previous.txt").write_text("preserve me")
            with patch.object(import_vector_index, "__file__", str(root / "import_vector_index.py")):
                with self.assertRaisesRegex(ValueError, "different dataset"):
                    import_vector_index.import_index(archive, target)
            self.assertEqual((target / "previous.txt").read_text(), "preserve me")

    def test_archive_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive, root = self.make_archive(Path(temporary))
            with ZipFile(archive, "a") as output:
                output.writestr("../outside.txt", "unsafe")
            with patch.object(import_vector_index, "__file__", str(root / "import_vector_index.py")):
                with self.assertRaisesRegex(ValueError, "Unsafe path"):
                    import_vector_index.import_index(archive, root / "index/qdrant")


if __name__ == "__main__":
    unittest.main()
