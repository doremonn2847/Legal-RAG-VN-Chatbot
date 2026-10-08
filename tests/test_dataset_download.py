"""Check dataset preparation using tiny local archives, without network requests."""
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from download_dataset import ARCHIVE_NAME, FILES, prepare_dataset


class DatasetPreparationTests(unittest.TestCase):
    def test_existing_archive_is_arranged_and_validated_without_network(self):
        corpus = [{"law_id": "law", "articles": [
            {"article_id": "1", "text": "Quyền của người lao động"},
            {"article_id": "2", "text": " "},
        ]}]
        csv_text = io.StringIO()
        writer = csv.writer(csv_text)
        writer.writerow(["question", "relevant_articles"])
        writer.writerow(["Quyền lao động?", "[{'law_id': 'law', 'article_id': '1'}]"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with ZipFile(root / ARCHIVE_NAME, "w") as archive:
                for name in FILES:
                    content = "placeholder"
                    if name == "legal_corpus.json":
                        content = json.dumps(corpus, ensure_ascii=False)
                    elif name == "train_qna.csv":
                        content = csv_text.getvalue()
                    elif name == "public_test_question.json":
                        content = json.dumps({"items": [{"question": "test"}]})
                    archive.writestr("nested/" + name, content)
            with patch("download_dataset.urlretrieve") as download:
                manifest = prepare_dataset(root)
            download.assert_not_called()
            self.assertEqual(manifest["laws"], 1)
            self.assertEqual(manifest["articles"], 2)
            self.assertEqual(manifest["indexable_articles"], 1)
            self.assertEqual(manifest["training_questions"], 1)
            self.assertEqual(manifest["public_test_questions"], 1)
            self.assertEqual(len(manifest["archive_sha256"]), 64)
            self.assertTrue(all((root / path).is_file() for path in FILES.values()))
            saved = json.loads((root / "dataset_manifest.json").read_text())
            self.assertEqual(saved, manifest)

    def test_incomplete_archive_fails_before_extracting_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with ZipFile(root / ARCHIVE_NAME, "w") as archive:
                archive.writestr("legal_corpus.json", "[]")
            with patch("download_dataset.urlretrieve") as download:
                with self.assertRaisesRegex(ValueError, "missing files"):
                    prepare_dataset(root)
            download.assert_not_called()
            self.assertFalse((root / "corpus").exists())


if __name__ == "__main__":
    unittest.main()
