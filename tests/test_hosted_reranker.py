import unittest
from unittest.mock import MagicMock, patch

from config import Config
from main.reranker import DocumentReranker
from utils.hosted_reranker import HostedReranker


class HostedRerankerTests(unittest.TestCase):
    def setUp(self):
        self.adapter = HostedReranker("https://example.test/rerank", "test-key")
        self.pairs = [["query", "relevant"], ["query", "unrelated"]]

    def response(self, rows, status=200):
        response = MagicMock(status_code=status)
        response.json.return_value = rows
        return response

    def test_sorted_response_is_restored_to_input_order(self):
        with patch("utils.hosted_reranker.requests.post", return_value=self.response(
            [{"index": 1, "score": -10.0}, {"index": 0, "score": 4.0}]
        )) as post:
            self.assertEqual(self.adapter.predict(self.pairs), [4.0, -10.0])
        self.assertTrue(post.call_args.kwargs["json"]["raw_scores"])

    def test_invalid_scores_and_missing_indexes_fail(self):
        cases = [[], [{"index": 0, "score": 1}, {"index": 0, "score": 2}],
                 [{"index": 0, "score": float("nan")}, {"index": 1, "score": 0}],
                 [{"index": 0, "score": 1}, {"index": 2, "score": 0}]]
        for rows in cases:
            with self.subTest(rows=rows), patch("utils.hosted_reranker.requests.post",
                                               return_value=self.response(rows)):
                with self.assertRaisesRegex(RuntimeError, "invalid response"):
                    self.adapter.predict(self.pairs)

    def test_auth_failures_stop_and_rate_limit_retries_are_bounded(self):
        with patch("utils.hosted_reranker.requests.post", return_value=self.response([], 401)) as post:
            with self.assertRaisesRegex(RuntimeError, "HTTP 401"):
                self.adapter.predict(self.pairs)
            self.assertEqual(post.call_count, 1)
        with patch("utils.hosted_reranker.requests.post", return_value=self.response([], 429)) as post, patch("utils.hosted_reranker.time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "HTTP 429"):
                self.adapter.predict(self.pairs)
            self.assertEqual(post.call_count, 3)

    def test_hosted_mode_avoids_local_weights_and_preserves_fusion(self):
        with patch.multiple(Config, RERANK_PROVIDER="hosted", RERANK_API_KEY="test-key"), patch("main.reranker.CrossEncoder") as local:
            reranker = DocumentReranker()
        local.assert_not_called()
        reranker.raise_errors = True
        documents = [{"id": "a", "content": "a" * 2100 + "relevant", "score": 0.5},
                     {"id": "b", "content": "unrelated", "score": 1.0}]
        with patch("utils.hosted_reranker.requests.post", return_value=self.response(
            [{"index": 0, "score": 4.0}, {"index": 1, "score": -10.0}]
        )) as post:
            result = reranker.rerank_with_fusion("query", documents, alpha=0.8, top_k=1)
        self.assertEqual(post.call_args.kwargs["json"]["texts"][0], documents[0]["content"])
        self.assertEqual(result[0]["id"], "a")
        self.assertAlmostEqual(result[0]["fused_score"], 0.9)
        self.assertNotIn("reranker_score", documents[0])
        with patch("utils.hosted_reranker.requests.post", return_value=self.response([], 424)):
            with self.assertRaisesRegex(RuntimeError, "HTTP 424"):
                reranker.rerank_documents("query", documents)
