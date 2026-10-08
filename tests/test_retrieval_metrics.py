import math
import unittest

from utils.retrieval_metrics import RAGEvaluator


class RetrievalMetricTests(unittest.TestCase):
    def setUp(self):
        self.evaluator = RAGEvaluator.__new__(RAGEvaluator)

    def test_exact_article_ids(self):
        e = self.evaluator
        for measure in (e.calculate_precision_at_k, e.calculate_recall_at_k, e.calculate_ndcg_at_k):
            self.assertEqual(measure(['law_70'], ['law_7'], 5), 0)
        self.assertEqual(e.calculate_average_precision(['law_70'], ['law_7']), 0)
        self.assertEqual(e.calculate_reciprocal_rank(['law_70'], ['law_7']), 0)

    def test_ndcg_perfect_missing_and_delayed(self):
        e = self.evaluator
        self.assertAlmostEqual(e.calculate_ndcg_at_k(['a', 'b'], ['a', 'b'], 5), 1)
        self.assertEqual(e.calculate_ndcg_at_k(['x'], ['a'], 5), 0)
        self.assertEqual(e.calculate_ndcg_at_k(['a'], [], 5), 0)
        self.assertEqual(e.calculate_ndcg_at_k(['a'], ['a'], 0), 0)
        self.assertAlmostEqual(e.calculate_ndcg_at_k(['x', 'a'], ['a'], 5), 1 / math.log2(3))
        self.assertAlmostEqual(e.calculate_ndcg_at_k(['a'], ['a', 'b'], 5), 1 / (1 + 1 / math.log2(3)))

    def test_duplicate_retrieval_does_not_inflate_metrics(self):
        e = self.evaluator
        self.assertAlmostEqual(e.calculate_precision_at_k(['a', 'a'], ['a'], 5), .2)
        self.assertEqual(e.calculate_recall_at_k(['a', 'a'], ['a'], 5), 1)
        self.assertEqual(e.calculate_average_precision(['a', 'a'], ['a']), 1)
        self.assertEqual(e.calculate_ndcg_at_k(['a', 'a'], ['a'], 5), 1)

    def test_both_cutoffs_are_reported(self):
        metrics = self.evaluator.calculate_metrics_at_k(['x'] * 5 + ['a'], ['a'], [5, 10])
        self.assertEqual(metrics['ndcg'][5], 0)
        self.assertGreater(metrics['ndcg'][10], 0)

    def test_benchmark_reports_cutoffs_and_binary_ndcg(self):
        from benchmark_retrieval import summarize
        records = []
        for k in (5, 10):
            records.append({'method': 'hybrid_rerank', 'k': k,
                            'retrieved_ids': ['x', 'y', 'z', 'w', 'v', 'a'][:k],
                            'expected_ids': ['a'], 'retrieval_seconds': 2.0})
        rows = summarize(records, self.evaluator)
        self.assertEqual([r['k'] for r in rows], [5, 10])
        self.assertEqual(rows[0]['hit_rate'], 0)
        self.assertEqual(rows[1]['hit_rate'], 1)
        self.assertAlmostEqual(rows[1]['mrr'], 1 / 6)
        self.assertAlmostEqual(rows[1]['ndcg'], 1 / math.log2(7))
        self.assertEqual(rows[0]['mean_seconds'], rows[1]['mean_seconds'])


if __name__ == '__main__':
    unittest.main()
