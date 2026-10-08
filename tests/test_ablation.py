import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmark_ablation import baseline_rows, strict_retrieve, write_report
from utils.ablation import rank_candidates, score_ranking


class AblationTests(unittest.TestCase):
    def test_hybrid_uses_fixed_pool_max_scores_deduplication_and_stable_ties(self):
        sparse = [{'id':'a','score':.6}, {'id':'b','score':.5}]
        dense = [{'id':'b','score':.9}, {'id':'c','score':.6}]
        self.assertEqual([d['id'] for d in rank_candidates(sparse,dense,'hybrid')], ['b','a','c'])
        self.assertEqual([d['id'] for d in rank_candidates(sparse,dense,'bm25')], ['a','b'])
        self.assertEqual([d['id'] for d in rank_candidates(sparse,dense,'dense')], ['b','c'])
        self.assertEqual(sparse[1]['score'],.5)

    def test_final_five_have_no_extra_score_filter(self):
        docs = [{'id':str(i),'score':.01*i} for i in range(25)]
        result = rank_candidates(docs,[],'bm25')
        self.assertEqual([d['id'] for d in result], ['24','23','22','21','20'])

    def test_metric_denominators_and_binary_ndcg(self):
        result = score_ranking(['irrelevant','a','b','a'], ['a','b','c'],5)
        self.assertAlmostEqual(result['precision'],2/5)
        self.assertAlmostEqual(result['recall'],2/3)
        self.assertAlmostEqual(result['map'],(.5+2/3)/3)
        self.assertEqual(result['mrr'],.5)
        self.assertAlmostEqual(result['ndcg'],(1/math.log2(3)+.5)/(1+1/math.log2(3)+.5))
        self.assertEqual(score_ranking([],[],5)['ndcg'],0)

    def test_failed_retrieval_is_not_scored_as_a_miss(self):
        def failed(query,top_k):
            print('Error searching documents: timed out')
            return []
        with self.assertRaisesRegex(RuntimeError,'reported an error'):
            strict_retrieve(failed,'question',25)
        self.assertEqual(strict_retrieve(lambda q,top_k: [],'question',25)[0],[])

    def test_partial_report_compares_identical_question_ids(self):
        a={'question_id':'a','method':'bm25','k':5,'expected_ids':['x'],'retrieved_ids':['x'],'retrieval_seconds':.1}
        b={'question_id':'b','method':'hybrid_rerank','k':5,'expected_ids':['y'],'retrieved_ids':[],'retrieval_seconds':1}
        rows=[{'question_id':'a','record':{**a,'method':'hybrid_rerank'}},{'question_id':'b','record':b}]
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)
            write_report(output,{'methods':['bm25']},{('a','bm25'):a},rows)
            summary=json.loads((output/'summary.json').read_text())
            self.assertEqual(summary['metrics'][0]['recall'],1)
            self.assertEqual(summary['matched_baseline_metrics'][0]['recall'],1)
            self.assertEqual(summary['matched_baseline_metrics'][0]['questions'],1)

    def test_baseline_rejects_incomplete_or_duplicate_questions(self):
        manifest={'protocol':'qwen-only-original-query-top5-v3','ids':['a','b'],'questions':2,'top_k':5,'refinement':False,'candidate_k':25}
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            (folder/'manifest.json').write_text(json.dumps(manifest))
            (folder/'questions.jsonl').write_text('{"question_id":"a"}\n')
            with self.assertRaisesRegex(ValueError,'incomplete'):
                baseline_rows(folder)


if __name__ == '__main__':
    unittest.main()
