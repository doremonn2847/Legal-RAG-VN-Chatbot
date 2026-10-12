import unittest
from unittest.mock import Mock, patch

from config import Config
from main.vector_store import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from utils.hosted_embeddings import HostedEmbeddings


class HostedEmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.encoder = HostedEmbeddings("https://example.test/v1/", "BAAI/bge-m3", 2, "test-key")

    def response(self, data, status=200):
        response = Mock(status_code=status, headers={})
        response.json.return_value = {"data": data}
        return response

    def test_response_order_and_separate_batches(self):
        data = [{"index": 1, "embedding": [0.0, 1.0]}, {"index": 0, "embedding": [1.0, 0.0]}]
        with patch("utils.hosted_embeddings.requests.post", side_effect=[self.response(data), self.response([data[1]])]) as post:
            result = self.encoder.encode(["first", "second", "third"], batch_size=2)
        self.assertEqual(result.tolist(), [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        self.assertEqual(post.call_args_list[0].args[0], "https://example.test/v1/embeddings")
        self.assertEqual(post.call_args.kwargs["json"]["input"], ["third"])

    def test_invalid_provider_vectors_are_rejected(self):
        cases = [[{"index": 0, "embedding": [1.0]}],
                 [{"index": 0, "embedding": [float("nan"), 0.0]}],
                 [{"index": 1, "embedding": [1.0, 0.0]}],
                 []]
        for data in cases:
            with self.subTest(data=data), patch("utils.hosted_embeddings.requests.post", return_value=self.response(data)):
                with self.assertRaises(RuntimeError):
                    self.encoder.encode("query")

    def test_rate_limit_retries_but_auth_failure_does_not(self):
        good = self.response([{"index": 0, "embedding": [1.0, 0.0]}])
        with patch("utils.hosted_embeddings.requests.post", side_effect=[self.response([], 429), good]) as post, patch("utils.hosted_embeddings.time.sleep") as sleep:
            self.encoder.encode("query")
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once()
        with patch("utils.hosted_embeddings.requests.post", return_value=self.response([], 401)) as post:
            with self.assertRaisesRegex(RuntimeError, "HTTP 401"):
                self.encoder.encode("query")
        self.assertEqual(post.call_count, 1)

    def test_hosted_initialization_does_not_load_local_embedding_weights(self):
        store = QdrantVectorStore.__new__(QdrantVectorStore)
        with patch.multiple(Config, EMBEDDING_PROVIDER="hosted", EMBEDDING_API_KEY="test-key"), patch("main.vector_store.SentenceTransformer") as local:
            store._initialize_embedding_model()
        self.assertIsInstance(store.embedding_model, HostedEmbeddings)
        local.assert_not_called()

    def test_incompatible_existing_collection_is_preserved(self):
        store = QdrantVectorStore.__new__(QdrantVectorStore)
        store.client = QdrantClient(":memory:")
        store.collection_name = "hosted_test"
        store.embedding_model = self.encoder
        store.client.create_collection(store.collection_name, vectors_config=VectorParams(size=3, distance=Distance.COSINE))
        try:
            with patch.object(Config, "EMBEDDING_DIMENSIONS", 2), self.assertRaisesRegex(ValueError, "incompatible"):
                store.create_collection()
            self.assertEqual(store.client.get_collection(store.collection_name).config.params.vectors.size, 3)
        finally:
            store.client.close()


if __name__ == "__main__":
    unittest.main()
