import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from api import check_dependencies, create_app, load_pipeline


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.rag = MagicMock()
        self.rag.answer_question.return_value = {
            "answer": "Answer", "retrieved_documents": [
                {"id": "law_1", "content": "x" * 1100, "score": 0.8,
                 "metadata": {"law_id": "law", "article_id": "1"}}],
            "fallback_used": False, "rejected_non_legal": False,
        }
        self.factory = MagicMock(return_value=self.rag)
        self.probe = MagicMock()
        self.app = create_app(self.factory, self.probe)

    def test_startup_routing_sources_and_shutdown(self):
        with TestClient(self.app) as client:
            self.factory.assert_called_once_with()
            self.assertEqual(client.get("/health").status_code, 200)
            self.assertEqual(client.get("/ready").status_code, 200)
            response = client.post("/ask", json={"question": "  Question?  "})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(response.json()["sources"][0]["excerpt"]), 1000)
            self.assertEqual(response.json()["sources"][0]["article_id"], "1")
            self.rag.answer_question.assert_called_once_with(
                "Question?", use_fallback=False, refine_question=False, top_k=5)
        self.rag.vector_store.client.close.assert_called_once()

    def test_invalid_requests(self):
        with TestClient(self.app) as client:
            for body in ({"question": " "}, {"question": "x" * 2001},
                         {"question": 42}, {"question": "q", "top_k": 10},
                         {"question": "q", "use_web_fallback": "false"}):
                with self.subTest(body=str(body)[:80]):
                    self.assertEqual(client.post("/ask", json=body).status_code, 422)
        self.rag.answer_question.assert_not_called()

    def test_invalid_session_id(self):
        with TestClient(self.app) as client:
            response = client.post("/ask", json={"question": "q"}, headers={"X-Session-ID": "x" * 65})
            self.assertEqual(response.status_code, 422)
        self.rag.answer_question.assert_not_called()

    def test_web_opt_in_and_nonlegal_result(self):
        self.rag.answer_question.return_value.update(
            rejected_non_legal=True, retrieved_documents=[])
        with TestClient(self.app) as client:
            response = client.post("/ask", json={"question": "q", "use_web_fallback": True})
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["rejected_non_legal"])
            self.assertTrue(self.rag.answer_question.call_args.kwargs["use_fallback"])

    def test_failure_is_sanitized_and_lock_released(self):
        self.rag.answer_question.side_effect = [RuntimeError("private secret"),
                                               self.rag.answer_question.return_value]
        with TestClient(self.app) as client:
            with self.assertLogs("api", level="ERROR"):
                response = client.post("/ask", json={"question": "q"})
            self.assertEqual(response.status_code, 502)
            self.assertNotIn("private secret", response.text)
            self.assertEqual(client.post("/ask", json={"question": "q"}).status_code, 200)

    def test_busy_request_and_health_during_inference(self):
        entered, release = threading.Event(), threading.Event()
        result = self.rag.answer_question.return_value
        def blocked(*args, **kwargs):
            entered.set()
            if not release.wait(10):
                raise RuntimeError("Test inference timed out")
            return result
        self.rag.answer_question.side_effect = blocked
        with TestClient(self.app) as client:
            responses = []
            worker = threading.Thread(target=lambda: responses.append(
                client.post("/ask", json={"question": "first"})))
            worker.start()
            try:
                self.assertTrue(entered.wait(5))
                self.assertEqual(client.get("/health").status_code, 200)
                response = client.post("/ask", json={"question": "second"})
                self.assertEqual(response.status_code, 429)
                self.assertEqual(response.headers["Retry-After"], "5")
                self.assertEqual(self.rag.answer_question.call_count, 1)
            finally:
                release.set()
                worker.join(5)
            self.assertEqual(responses[0].status_code, 200)

    def test_readiness_outage(self):
        with TestClient(self.app) as client:
            self.probe.side_effect = RuntimeError("service offline")
            with self.assertLogs("api", level="ERROR"):
                self.assertEqual(client.get("/ready").status_code, 503)
            self.assertEqual(client.get("/health").status_code, 200)

    def test_startup_dependency_failure_closes_client(self):
        self.probe.side_effect = RuntimeError("missing model")
        with self.assertRaisesRegex(RuntimeError, "missing model"):
            with TestClient(self.app):
                pass
        self.rag.vector_store.client.close.assert_called_once()


class StrictPipelineTests(unittest.TestCase):
    def test_loader_validates_index_and_enables_strict_errors(self):
        rag = SimpleNamespace(
            llm=object(), vector_store=SimpleNamespace(client=MagicMock(), embedding_model=object()),
            bm25_retriever=SimpleNamespace(), reranker=SimpleNamespace(model=object()))
        data = {"bm25": SimpleNamespace(corpus_size=1),
                "documents": [{}], "tokenized_corpus": [[]]}
        with patch("api.Path") as path, patch("api.pickle.load", return_value=data), \
                patch("main.chatbot.VietnameseLegalRAG", return_value=rag) as factory:
            path.return_value.is_file.return_value = False
            with self.assertRaisesRegex(RuntimeError, "required"):
                load_pipeline()
            factory.assert_not_called()
            path.return_value.is_file.return_value = True
            self.assertIs(load_pipeline(), rag)
            for component in (rag, rag.vector_store, rag.reranker):
                self.assertTrue(component.raise_errors)
            data["tokenized_corpus"] = []
            with self.assertRaisesRegex(RuntimeError, "inconsistent"):
                load_pipeline()
            rag.vector_store.client.close.assert_called_once()
            path.return_value.unlink.assert_not_called()

    def test_readiness_requires_model_and_matching_indexes(self):
        from config import Config
        rag = SimpleNamespace(vector_store=MagicMock(),
                              bm25_retriever=SimpleNamespace(documents=[{}, {}]))
        rag.vector_store.client.get_collection.return_value.points_count = 2
        with patch("requests.get") as get:
            get.return_value.json.return_value = {"models": [{"name": Config.MODEL_GEN}]}
            check_dependencies(rag)
            rag.vector_store.client.get_collection.return_value.points_count = 1
            with self.assertRaisesRegex(RuntimeError, "matching"):
                check_dependencies(rag)
            get.return_value.json.return_value = {"models": []}
            with self.assertRaisesRegex(RuntimeError, "Download"):
                check_dependencies(rag)

    def test_retrieval_propagates_errors_only_when_enabled(self):
        from main.chatbot import VietnameseLegalRAG
        rag = VietnameseLegalRAG.__new__(VietnameseLegalRAG)
        rag.bm25_retriever = MagicMock()
        rag.vector_store = MagicMock()
        rag.bm25_retriever.get_relevant_documents.side_effect = RuntimeError("retrieval failed")
        self.assertEqual(rag.retrieve_documents("q"), [])
        rag.raise_errors = True
        with self.assertRaisesRegex(RuntimeError, "retrieval failed"):
            rag.retrieve_documents("q")

    def test_generation_error_opt_in(self):
        from main.chatbot import VietnameseLegalRAG
        rag = VietnameseLegalRAG.__new__(VietnameseLegalRAG)
        rag.llm = MagicMock()
        rag.llm.invoke.side_effect = RuntimeError("inference failed")
        self.assertIn("inference failed", rag.generate_answer("q", "context"))
        rag.raise_errors = True
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            rag.generate_answer("q", "context")

    def test_embedding_search_error_opt_in(self):
        from main.vector_store import QdrantVectorStore
        store = QdrantVectorStore.__new__(QdrantVectorStore)
        store.embedding_model = MagicMock()
        store.embedding_model.encode.side_effect = RuntimeError("embedding failed")
        self.assertEqual(store.search_similar_documents("q"), [])
        store.raise_errors = True
        with self.assertRaisesRegex(RuntimeError, "embedding failed"):
            store.search_similar_documents("q")

    def test_reranking_error_opt_in(self):
        from main.reranker import DocumentReranker
        reranker = DocumentReranker.__new__(DocumentReranker)
        reranker.model = MagicMock()
        reranker.model.predict.side_effect = RuntimeError("rerank failed")
        docs = [{"id": "1", "content": "evidence"}]
        self.assertEqual(reranker.rerank_with_fusion("q", docs), docs)
        reranker.raise_errors = True
        with self.assertRaisesRegex(RuntimeError, "rerank failed"):
            reranker.rerank_with_fusion("q", docs)


if __name__ == "__main__":
    unittest.main()
