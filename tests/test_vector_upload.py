import unittest
from qdrant_client import QdrantClient, models
from upload_vector_index import upload_index


class VectorUploadTests(unittest.TestCase):
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
