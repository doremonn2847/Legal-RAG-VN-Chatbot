import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage
from utils.model_trial import refine_once, invoke_trial, make_trial_llm
from benchmark_model_trial import save_report
from utils.retrieval_metrics import RAGEvaluator


class ModelTrialTests(unittest.TestCase):
    def test_output_caps_and_thinking_settings(self):
        ollama = MagicMock()
        with patch.dict(sys.modules, {'langchain_ollama': MagicMock(ChatOllama=ollama)}):
            make_trial_llm('ollama', 'qwen3.5:9b')
        self.assertEqual(ollama.call_args.kwargs['num_predict'], 2048)
        self.assertEqual(ollama.call_args.kwargs['num_ctx'], 8192)
        self.assertFalse(ollama.call_args.kwargs['reasoning'])

    def test_finish_reason_is_saved(self):
        llm = MagicMock()
        llm.invoke.return_value = AIMessage(content='Partial', response_metadata={'done_reason': 'length'})
        self.assertEqual(invoke_trial(llm, [])['finish_reason'], 'length')

    def test_single_call_refinement_and_usage(self):
        llm = MagicMock()
        llm.invoke.return_value = AIMessage(content='Người lao động có quyền được trả lương đúng hạn không?',
                                            usage_metadata={'input_tokens': 20, 'output_tokens': 12, 'total_tokens': 32})
        result = refine_once(llm, 'Người lao động được nhận lương đúng hạn không?')
        self.assertEqual(llm.invoke.call_count, 1)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['usage']['total_tokens'], 32)

    def test_invented_article_rejected_and_original_preserved(self):
        llm = MagicMock()
        llm.invoke.return_value = AIMessage(content='Theo Điều 115, trả lương thế nào?')
        question = 'Trả lương đúng hạn là gì?'
        result = refine_once(llm, question)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['refined_question'], question)
        self.assertEqual(result['introduced_references'], ['điều 115'])

    def test_existing_reference_kept(self):
        llm = MagicMock()
        llm.invoke.return_value = AIMessage(content='Điều 7 của 47/2011/TT-BCA quy định thế nào?')
        self.assertTrue(refine_once(llm, 'Điều 7 của 47/2011/tt-bca là gì?')['accepted'])

    def test_rate_limit_wait_and_attempt_count(self):
        error = RuntimeError('rate limit')
        error.status_code = 429
        error.response = MagicMock(headers={'retry-after': '90'})
        llm = MagicMock()
        llm.invoke.side_effect = [error, AIMessage(content='Question?')]
        with patch('utils.model_trial.time.sleep') as sleep:
            result = invoke_trial(llm, [])
        sleep.assert_called_once_with(90.0)
        self.assertEqual(result['rate_limit_retries'], 1)
        self.assertEqual(llm.invoke.call_count, 2)

    def test_non_rate_limit_failure_is_not_silently_scored(self):
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError('unavailable')
        with self.assertRaisesRegex(RuntimeError, 'unavailable'):
            refine_once(llm, 'Question?')
        self.assertEqual(llm.invoke.call_count, 1)

    def test_manual_answer_reviews_survive_report_regeneration(self):
        record = {'method': 'hybrid_rerank', 'k': 5, 'expected_ids': ['law_1'],
                  'retrieved_ids': ['law_1'], 'retrieval_seconds': 1.0}
        response = {'seconds': 1, 'usage': {}, 'rate_limit_retries': 0, 'text': 'Answer',
                    'accepted': True, 'refined_question': 'Query'}
        item = {'question_id': 'id', 'question': 'Question', 'expected_ids': ['law_1'],
                'shared_evidence_ids': ['law_1'], 'arms': {
                    'original_query': {'records': [record]},
                    **{arm: {'answer': response} for arm in ('qwen',)}}}
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            evaluator = RAGEvaluator.__new__(RAGEvaluator)
            save_report(output, {'questions': 1}, {'id': item}, evaluator)
            summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
            self.assertEqual([(r['arm'], r['k']) for r in summary['metrics']], [('original_query', 5)])
            path = output / 'answer_review.csv'
            with path.open(encoding='utf-8-sig', newline='') as handle:
                reader = csv.DictReader(handle)
                fields = reader.fieldnames
                rows = list(reader)
            self.assertEqual(rows[0]['qwen_query'], 'Question')
            rows[0]['qwen_pass'] = 'yes'
            with path.open('w', encoding='utf-8-sig', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            save_report(output, {'questions': 1}, {'id': item}, evaluator)
            with path.open(encoding='utf-8-sig', newline='') as handle:
                self.assertEqual(next(csv.DictReader(handle))['qwen_pass'], 'yes')
