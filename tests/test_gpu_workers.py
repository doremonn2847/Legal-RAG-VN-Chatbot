import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import subprocess

from benchmark_model_trial import check_gpu_residency, shard_questions
from benchmark_multi_gpu import merge_checkpoints, read_checkpoint, repair_checkpoint, stop_workers, worker_environment


class GPUWorkerTests(unittest.TestCase):
    def test_shards_cover_each_question_once_and_preserve_order(self):
        rows = list(range(51))
        left, right = [shard_questions(rows, i, 2) for i in range(2)]
        self.assertEqual(left, rows[::2])
        self.assertEqual(right, rows[1::2])
        self.assertFalse(set(left).intersection(right))
        self.assertEqual(sorted(left + right), rows)
        self.assertEqual(shard_questions(rows, 0, 1), rows)
        with self.assertRaises(ValueError):
            shard_questions(rows, 2, 2)

    def test_each_worker_sees_only_its_gpu_and_ollama_server(self):
        with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0', 'QDRANT_PATH': 'index/qdrant'}):
            env = worker_environment(1, 'http://127.0.0.1:11436')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '1')
        self.assertEqual(env['OLLAMA_BASE_URL'], 'http://127.0.0.1:11436')
        self.assertEqual(env['QDRANT_PATH'], '')

    def test_cpu_inference_and_substantial_offload_are_rejected(self):
        for models in ([], [{'name': 'qwen3.5:9b', 'size_vram': 0}],
                       [{'name': 'qwen3.5:9b', 'size': 100, 'size_vram': 50}]):
            with self.assertRaises(RuntimeError):
                check_gpu_residency(models)
        check_gpu_residency([{'name': 'qwen3.5:9b', 'size': 100, 'size_vram': 98}])

    def test_interrupted_final_append_is_repaired_but_middle_corruption_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'questions.jsonl'
            path.write_text('{"ok": 1}\n{"partial":', encoding='utf-8')
            self.assertEqual(repair_checkpoint(path), [{'ok': 1}])
            self.assertEqual(path.read_text(), '{"ok": 1}\n')
            path.write_text('broken\n{"ok":1}\n', encoding='utf-8')
            with self.assertRaises(json.JSONDecodeError):
                read_checkpoint(path)

    def write_worker(self, output, index, **overrides):
        folder = output / f'worker-{index}'
        folder.mkdir()
        ids = ['a', 'b', 'c']
        manifest = {'all_ids': ids, 'ids': ids[index::2], 'questions': len(ids[index::2]),
                    'worker_index': index, 'worker_count': 2, 'model_digest': 'same', **overrides}
        (folder / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        items = [{'question_id': qid} for qid in ids[index::2]]
        (folder / 'questions.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in items), encoding='utf-8')
        (folder / 'model_calls.jsonl').write_text(json.dumps({'key': f'{index}:answer', 'value': {}}) + '\n', encoding='utf-8')
        return folder

    def test_merge_orders_results_counts_once_and_preserves_worker_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            worker = self.write_worker(output, 0)
            self.write_worker(output, 1)
            before = (worker / 'questions.jsonl').read_bytes()
            with patch('benchmark_model_trial.save_report') as report:
                self.assertEqual(merge_checkpoints(output, 2), (3, 3))
            rows = read_checkpoint(output / 'questions.jsonl')
            self.assertEqual([r['question_id'] for r in rows], ['a', 'b', 'c'])
            self.assertEqual(len(read_checkpoint(output / 'model_calls.jsonl')), 2)
            self.assertEqual((worker / 'questions.jsonl').read_bytes(), before)
            self.assertEqual(list(report.call_args.args[2]), ['a', 'b', 'c'])

    def test_partial_merge_is_explicit_and_late_worker_can_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            self.write_worker(output, 0)
            with patch('benchmark_model_trial.save_report'):
                self.assertEqual(merge_checkpoints(output, 2), (2, 3))
                self.write_worker(output, 1)
                self.assertEqual(merge_checkpoints(output, 2), (3, 3))

    def test_different_worker_settings_cannot_be_merged(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            self.write_worker(output, 0)
            self.write_worker(output, 1, model_digest='different')
            with self.assertRaisesRegex(ValueError, 'manifests differ'):
                merge_checkpoints(output, 2)

    def test_duplicate_or_wrong_shard_question_cannot_be_merged(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            self.write_worker(output, 0)
            worker = self.write_worker(output, 1)
            (worker / 'questions.jsonl').write_text('{"question_id":"a"}\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Unexpected or duplicate'):
                merge_checkpoints(output, 2)

    def test_shutdown_kills_only_launched_worker_processes_if_needed(self):
        stuck, finished = MagicMock(), MagicMock()
        stuck.poll.return_value = None
        stuck.wait.side_effect = [subprocess.TimeoutExpired('worker', 10), 0]
        finished.poll.return_value = 0
        stop_workers([stuck, finished])
        stuck.terminate.assert_called_once()
        stuck.kill.assert_called_once()
        finished.terminate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
