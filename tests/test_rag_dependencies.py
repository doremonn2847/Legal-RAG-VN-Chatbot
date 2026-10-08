"""Offline integration checks: provider wiring, article limits, and Qdrant API."""
import unittest
import tempfile
from unittest.mock import MagicMock, patch
import numpy as np

from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from config import Config
from main.chatbot import VietnameseLegalRAG
from main.reranker import DocumentReranker
from main.vector_store import QdrantVectorStore
from utils.question_refiner import VietnameseLegalQuestionRefiner


class OllamaIntegrationTests(unittest.TestCase):
    def test_generation_and_legacy_helpers_use_ollama(self):
        with (
            patch("main.chatbot.QdrantVectorStore"),
            patch("main.chatbot.BM25Retriever"),
            patch("main.chatbot.DocumentReranker"),
        ):
            rag = VietnameseLegalRAG()
            status = rag.get_system_status()
        self.assertIsInstance(rag.llm, ChatOllama)
        self.assertIsInstance(rag.question_refiner.llm, ChatOllama)
        self.assertEqual(rag.llm.model, Config.MODEL_GEN)
        self.assertEqual(rag.question_refiner.llm.model, Config.MODEL_REFINE)
        self.assertEqual(rag.llm.base_url, Config.OLLAMA_BASE_URL)
        self.assertEqual(rag.llm.temperature, 0.1)
        self.assertEqual(rag.llm.num_predict, 2048)
        self.assertTrue(status["ollama_configured"])

    def test_disabled_llm_keeps_legacy_refinement_rules_available(self):
        with patch.multiple(Config, ENABLE_QUESTION_REFINEMENT=False, USE_LLM_FOR_LEGAL_DETECTION=False):
            refiner = VietnameseLegalQuestionRefiner()
        self.assertIsNone(refiner.llm)
        result = refiner.refine_question("Quyền của người lao động là gì?", use_llm=False)
        self.assertTrue(result["refined_question"])

    def test_generation_uses_current_langchain_messages(self):
        rag = VietnameseLegalRAG.__new__(VietnameseLegalRAG)
        rag.llm = MagicMock()
        rag.llm.invoke.return_value = AIMessage(content="Câu trả lời có căn cứ.")
        answer = rag.generate_answer("Quyền lao động?", "Điều 1: Nội dung pháp luật.")
        self.assertEqual(answer, "Câu trả lời có căn cứ.")
        messages = rag.llm.invoke.call_args.args[0]
        self.assertIsInstance(messages[0], HumanMessage)
        self.assertIn("Điều 1", messages[0].content)


class ArticleLimitTests(unittest.TestCase):
    def test_generation_keeps_characters_after_old_limit(self):
        rag = VietnameseLegalRAG.__new__(VietnameseLegalRAG)
        text = "a" * 600 + "QUYỀN_LỢI" + "b" * 500
        context = rag.format_context([{"content": text}])
        self.assertIn("QUYỀN_LỢI", context)
        self.assertIn(text[:1000] + "...", context)
        self.assertNotIn(text[:1001], context)

    def test_exactly_1000_characters_is_not_truncated(self):
        rag = VietnameseLegalRAG.__new__(VietnameseLegalRAG)
        text = "a" * 1000
        context = rag.format_context([{"content": text}])
        self.assertIn(text, context)
        self.assertNotIn("...", context)

    def test_bge_receives_1000_characters_and_orders_documents(self):
        with patch("main.reranker.CrossEncoder") as encoder:
            encoder.return_value.predict.return_value = [0.2, 0.8]
            reranker = DocumentReranker()
            result = reranker.rerank_documents(
                "query",
                [{"id": "first", "content": "a" * 1100},
                 {"id": "second", "content": "short"}],
            )
        encoder.assert_called_once_with("BAAI/bge-reranker-v2-m3")
        pairs = encoder.return_value.predict.call_args.args[0]
        self.assertEqual(pairs[0][1], "a" * 1000 + "...")
        self.assertEqual([doc["id"] for doc in result], ["second", "first"])

    def test_bge_load_failure_does_not_switch_to_another_model(self):
        with patch("main.reranker.CrossEncoder", side_effect=RuntimeError("unavailable")) as encoder:
            reranker = DocumentReranker()
        encoder.assert_called_once_with(Config.RERANKER_MODEL)
        self.assertIsNone(reranker.model)
        documents = [{"id": "article", "content": "text"}]
        self.assertEqual(reranker.rerank_documents("query", documents), documents)


class QdrantCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.store = QdrantVectorStore.__new__(QdrantVectorStore)
        self.store.client = QdrantClient(":memory:")
        self.store.collection_name = "test_articles"
        self.store.client.create_collection(
            self.store.collection_name,
            vectors_config=VectorParams(size=2, distance=Distance.COSINE),
        )
        self.store.client.upsert(self.store.collection_name, points=[
            PointStruct(id=1, vector=[1.0, 0.0], payload={"article_id": "law_1", "content": "first"}),
            PointStruct(id=2, vector=[0.8, 0.6], payload={"article_id": "law_2", "content": "second"}),
        ])
        self.store.embed_text = lambda query: [1.0, 0.0]

    def tearDown(self):
        self.store.client.close()

    def test_current_qdrant_search_and_collection_info(self):
        result = self.store.search_similar_documents("query", top_k=2, score_threshold=0.1)
        self.assertEqual([doc["id"] for doc in result], ["law_1", "law_2"])
        self.assertEqual(self.store.get_collection_info()["points_count"], 2)

    def test_no_matching_vectors_returns_empty_list(self):
        self.assertEqual(self.store.search_similar_documents("query", score_threshold=1.1), [])

    def test_batch_indexing_resumes_without_duplicate_points(self):
        self.store.embedding_model = MagicMock()
        self.store.embedding_model.encode.side_effect = lambda texts, **kwargs: np.array([[1.0, 0.0] for _ in texts])
        documents = [{"id": f"new_{i}", "content": f"article {i}"} for i in range(3)]
        with patch.object(Config, "INDEX_BATCH_SIZE", 2):
            self.store.add_documents(documents)
            self.store.add_documents(documents)
        self.assertEqual(self.store.get_collection_info()["points_count"], 5)
        self.assertEqual(self.store.embedding_model.encode.call_count, 2)
        self.assertEqual(self.store.embedding_model.encode.call_args_list[0].args[0], ["article 0", "article 1"])

    def test_local_disk_collection_survives_reopening(self):
        with tempfile.TemporaryDirectory() as path:
            store = QdrantVectorStore.__new__(QdrantVectorStore)
            with patch.object(Config, "QDRANT_PATH", path), patch.object(Config, "QDRANT_URL", None), patch.object(Config, "QDRANT_API_KEY", None):
                store._initialize_client()
                store.client.create_collection("persistent", vectors_config=VectorParams(size=2, distance=Distance.COSINE))
                store.client.upsert("persistent", points=[PointStruct(id=1, vector=[1.0, 0.0])])
                store.client.close()
                store._initialize_client()
                self.assertEqual(store.client.count("persistent").count, 1)
                store.client.close()


if __name__ == "__main__":
    unittest.main()
