"""Offline checks for hosted inference without loading models or spending tokens."""
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import requests
from langchain_deepseek import ChatDeepSeek

from api import check_dependencies
from config import Config
from main.chatbot import VietnameseLegalRAG
from utils.inference import check_inference_ready, create_chat_model


class DeepSeekTests(unittest.TestCase):
    def setUp(self):
        self.settings = patch.multiple(
            Config, INFERENCE_PROVIDER="deepseek", MODEL_GEN="deepseek-flash",
            DEEPSEEK_BASE_URL="https://api.deepseek.com", DEEPSEEK_API_KEY="test-key",
        )
        self.settings.start()
        self.addCleanup(self.settings.stop)

    def test_generation_and_legal_detection_use_deepseek_without_ollama(self):
        with (patch("main.chatbot.QdrantVectorStore"),
              patch("main.chatbot.BM25Retriever"),
              patch("main.chatbot.DocumentReranker"),
              patch("langchain_ollama.ChatOllama") as ollama):
            rag = VietnameseLegalRAG()
        for llm in (rag.llm, rag.question_refiner.llm):
            self.assertIsInstance(llm, ChatDeepSeek)
            self.assertEqual(llm.model_name, "deepseek-flash")
            self.assertEqual(llm.max_tokens, 2048)
            self.assertEqual(llm.extra_body, {"thinking": {"type": "disabled"}})
        ollama.assert_not_called()
        status = rag.get_system_status()
        self.assertEqual(status["inference_provider"], "deepseek")
        self.assertFalse(status["ollama_configured"])

    def test_readiness_uses_models_and_still_checks_index_counts(self):
        rag = SimpleNamespace(vector_store=MagicMock(),
                              bm25_retriever=SimpleNamespace(documents=[{}]))
        rag.vector_store.client.get_collection.return_value.points_count = 1
        with patch("utils.inference.requests.get") as get:
            get.return_value.json.return_value = {"data": [{"id": "deepseek-flash"}]}
            check_dependencies(rag)
            get.assert_called_once_with("https://api.deepseek.com/models",
                                        headers={"Authorization": "Bearer test-key"}, timeout=5)
            rag.vector_store.client.get_collection.return_value.points_count = 0
            with self.assertRaisesRegex(RuntimeError, "matching"):
                check_dependencies(rag)

    def test_unavailable_model_and_auth_failures_do_not_fallback(self):
        with patch("utils.inference.requests.get") as get:
            get.return_value.json.return_value = {"data": [{"id": "different-model"}]}
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                check_inference_ready()
            get.return_value.raise_for_status.side_effect = requests.HTTPError("401")
            with self.assertRaises(requests.HTTPError):
                check_inference_ready()

    def test_missing_key_and_unsafe_url_fail_before_requests(self):
        with patch.object(Config, "DEEPSEEK_API_KEY", ""):
            with self.assertRaisesRegex(ValueError, "DEEPSEEK_API_KEY"):
                create_chat_model()
        with patch.object(Config, "DEEPSEEK_BASE_URL", "http://api.deepseek.com"):
            with self.assertRaisesRegex(ValueError, "HTTPS"):
                create_chat_model()
