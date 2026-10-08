import unittest
from unittest.mock import MagicMock, patch

from config import Config
from main.chatbot import VietnameseLegalRAG


class QueryRoutingTests(unittest.TestCase):
    def setUp(self):
        self.rag = VietnameseLegalRAG.__new__(VietnameseLegalRAG)
        self.rag.question_refiner = MagicMock()
        self.rag.question_refiner.refine_question.return_value = {'refined_question': 'User question?'}
        self.rag.retrieve_documents = MagicMock(return_value=[{'id': 'law_1', 'content': 'Evidence'}])
        self.rag.generate_answer = MagicMock(return_value='Supported answer')
        self.rag.google_search = MagicMock()
        self.rag.google_search.search_legal_info.return_value = [{'title': 'Web evidence'}]
        self.rag.google_search.format_search_results.return_value = 'Web context'
        self.settings = patch.multiple(Config, ENABLE_LEGAL_DOMAIN_FILTER=False,
                                      ENABLE_QUESTION_REFINEMENT=True, ENABLE_GOOGLE_SEARCH=True)
        self.settings.start()
        self.addCleanup(self.settings.stop)

    def assert_routing(self, result):
        self.rag.retrieve_documents.assert_called_once_with('User question?', top_k=5)
        self.assertEqual(result['original_question'], 'User question?')
        self.assertEqual(result['refined_question'], 'User question?')
        self.rag.question_refiner.refine_question.assert_not_called()
        for call in self.rag.generate_answer.call_args_list:
            self.assertEqual(call.args[0], 'User question?')

    def test_generation_uses_original_question(self):
        result = self.rag.answer_question('User question?', use_fallback=False, top_k=5)
        self.assert_routing(result)
        self.assertFalse(result['fallback_used'])

    def test_empty_retrieval_web_fallback_uses_original_question(self):
        self.rag.retrieve_documents.return_value = []
        result = self.rag.answer_question('User question?', top_k=5)
        self.assert_routing(result)
        self.rag.google_search.search_legal_info.assert_called_once_with('User question?')
        self.assertTrue(result['fallback_used'])

    def test_insufficient_answer_web_retry_uses_original_question(self):
        self.rag.generate_answer.side_effect = ['Không có đủ thông tin trong tài liệu tham khảo được cung cấp để trả lời trực tiếp câu hỏi này.', 'Supported answer']
        result = self.rag.answer_question('User question?', top_k=5)
        self.assert_routing(result)
        self.assertEqual(self.rag.generate_answer.call_count, 2)
        self.assertTrue(result['fallback_used'])

    def test_disabled_refinement_preserves_query(self):
        self.rag.answer_question('User question?', refine_question=False, top_k=5)
        self.rag.retrieve_documents.assert_called_once_with('User question?', top_k=5)
        self.rag.question_refiner.refine_question.assert_not_called()

    def test_refinement_request_does_not_rewrite_query(self):
        result = self.rag.answer_question('User question?', refine_question=True, top_k=5)
        self.assert_routing(result)
        self.assertIsNone(result['question_refinement'])

    def test_answer_generation_rejects_top10(self):
        with self.assertRaisesRegex(ValueError, 'top-5'):
            self.rag.answer_question('User question?', top_k=10)
        self.rag.retrieve_documents.assert_not_called()


class RetrievalLimitTests(unittest.TestCase):
    def setUp(self):
        self.rag = VietnameseLegalRAG.__new__(VietnameseLegalRAG)
        self.rag.vector_store = MagicMock()
        self.rag.bm25_retriever = None
        self.rag.reranker = MagicMock()
        self.documents = [{'id': str(i), 'score': 0.9, 'content': 'Evidence'} for i in range(25)]
        self.rag.vector_store.search_similar_documents.return_value = self.documents
        # A failed/disabled reranker may return all candidates: enforce the limit anyway.
        self.rag.reranker.rerank_with_fusion.return_value = self.documents
        self.rag.reranker.rerank_documents.return_value = self.documents

    def test_both_limits_with_fused_pure_and_unavailable_reranking(self):
        for top_k in (5, 10):
            for fusion in (False, True):
                with self.subTest(top_k=top_k, fusion=fusion), patch.object(Config, 'USE_SCORE_FUSION', fusion):
                    self.assertEqual(len(self.rag.retrieve_documents('query', use_reranking=True, top_k=top_k)), top_k)
            self.rag.reranker = None
            self.assertEqual(len(self.rag.retrieve_documents('query', use_reranking=True, top_k=top_k)), top_k)
            self.rag.reranker = MagicMock()
            self.rag.reranker.rerank_with_fusion.return_value = self.documents
            self.rag.reranker.rerank_documents.return_value = self.documents

    def test_limits_without_reranking_and_configured_default(self):
        for top_k in (5, 10):
            with self.subTest(top_k=top_k), patch.object(Config, 'TOP_K_RETRIEVAL', top_k):
                self.assertEqual(len(self.rag.retrieve_documents('query', use_reranking=False)), top_k)
                self.rag.vector_store.search_similar_documents.assert_called_with('query', top_k=top_k)

    def test_invalid_limits_fail_before_retrieval(self):
        for top_k in (0, 20, 5.0, '5'):
            with self.subTest(top_k=top_k), self.assertRaisesRegex(ValueError, '5 or 10'):
                self.rag.retrieve_documents('query', top_k=top_k)
        self.rag.vector_store.search_similar_documents.assert_not_called()
