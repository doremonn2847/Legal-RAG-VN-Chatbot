import json
import math
from pathlib import Path
import tempfile
import unittest

from benchmark_hosted import dataset_rows, measured
from utils.ablation import rank_candidates, score_ranking


class HostedBenchmarkTests(unittest.TestCase):
    def test_metrics_use_all_unique_gold_articles_and_binary_ndcg(self):
        m = score_ranking(["wrong", "a", "b", "a"], ["a", "b", "c"], 5)
        self.assertAlmostEqual(m["recall"], 2 / 3)
        self.assertAlmostEqual(m["map"], (0.5 + 2 / 3) / 3)
        self.assertEqual(m["mrr"], 0.5)
        self.assertAlmostEqual(m["ndcg"], (1 / math.log2(3) + 0.5) / (1 + 1 / math.log2(3) + 0.5))

    def test_candidate_merge_deduplicates_and_preserves_tie_order(self):
        sparse = [{"id": "a", "score": 0.6}, {"id": "b", "score": 0.5}]
        dense = [{"id": "b", "score": 0.9}, {"id": "c", "score": 0.6}]
        self.assertEqual([d["id"] for d in rank_candidates(sparse, dense, "hybrid")], ["b", "a", "c"])
        self.assertEqual(sparse[1]["score"], 0.5)

    def test_reported_retriever_errors_are_not_scored_as_misses(self):
        def failed():
            print("Error loading candidates")
            return []
        with self.assertRaises(RuntimeError):
            measured(failed)

    def test_dataset_preserves_questions_and_deduplicates_gold(self):
        row = {"question_id": "a", "question": "Original query", "relevant_articles": [{"law_id": "law", "article_id": "1"}] * 2}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "train.json"
            path.write_text(json.dumps({"items": [row]}))
            rows = dataset_rows(path)
            self.assertEqual(rows[0]["question"], "Original query")
            self.assertEqual(rows[0]["expected_ids"], ["law_1"])
            path.write_text(json.dumps({"items": [row, row]}))
            with self.assertRaises(ValueError):
                dataset_rows(path)
