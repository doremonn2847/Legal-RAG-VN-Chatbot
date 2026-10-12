import unittest
from unittest.mock import patch
from qdrant_client import QdrantClient, models
from upload_vector_index import main, upload_index


class VectorUploadTests(unittest.TestCase):
    def test_restore_payloads_without_reading_broken_source_payloads(self):
        source = QdrantClient(":memory:")
        target = QdrantClient(":memory:")
        try:
            source.create_collection("test", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
            source.upsert("test", points=[models.PointStruct(id=1, vector=[1.0, 0.0])])
            payloads = {"1": {"article_id": "law_1", "content": "Full legal article"}}
            with patch.object(source, "scroll", wraps=source.scroll) as scroll:
                self.assertEqual(upload_index(source, target, "test", payloads), 1)
                self.assertFalse(scroll.call_args.kwargs["with_payload"])
            point = target.retrieve("test", ids=[1], with_vectors=True)[0]
            self.assertEqual(point.payload, payloads["1"])
            self.assertEqual(point.vector, [1.0, 0.0])
        finally:
            source.close()
            target.close()

    def test_cloud_credentials_validated_before_connecting(self):
        for url, key in (("", ""), ("http://example.com", "private-key"), ("https://example.com", "")):
            with self.subTest(url=url), patch.dict("os.environ", {
                "QDRANT_CLOUD_URL": url, "QDRANT_CLOUD_API_KEY": key,
            }), patch("upload_vector_index.QdrantClient") as client, patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    main(["--source-url", "http://127.0.0.1:6333", "--cloud"])
                client.assert_not_called()

    def test_cloud_copy_closes_connections_after_interruption(self):
        with patch.dict("os.environ", {
            "QDRANT_CLOUD_URL": "https://cluster.example.com", "QDRANT_CLOUD_API_KEY": "test-key",
        }), patch("upload_vector_index.QdrantClient") as client, patch("upload_vector_index.upload_index", side_effect=RuntimeError("interrupted")):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                main(["--source-url", "http://127.0.0.1:6333", "--cloud", "--collection", "hosted-test"])
            self.assertEqual(client.return_value.close.call_count, 2)

    def test_transfer_preserves_vectors_payloads_and_can_resume(self):
        source = QdrantClient(":memory:")
        target = QdrantClient(":memory:")
        try:
            source.create_collection("test", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
            source.upsert("test", points=[models.PointStruct(id=i, vector=[1.0, 0.0], payload={"article_id": str(i)}) for i in range(260)])
            self.assertEqual(upload_index(source, target, "test"), 260)
            self.assertEqual(upload_index(source, target, "test"), 260)
            points = target.retrieve("test", ids=[259], with_vectors=True)
            self.assertEqual(points[0].payload, {"article_id": "259"})
            self.assertEqual(points[0].vector, [1.0, 0.0])
        finally:
            source.close()
            target.close()

    def test_incompatible_collection_is_preserved(self):
        source = QdrantClient(":memory:")
        target = QdrantClient(":memory:")
        try:
            source.create_collection("test", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
            target.create_collection("test", vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE))
            with self.assertRaisesRegex(ValueError, "incompatible"):
                upload_index(source, target, "test")
            self.assertEqual(target.get_collection("test").config.params.vectors.size, 3)
        finally:
            source.close()
            target.close()
