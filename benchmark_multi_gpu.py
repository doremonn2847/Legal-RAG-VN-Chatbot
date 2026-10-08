"""Run isolated Qwen/retrieval workers and merge validated checkpoints."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def read_checkpoint(path):
    if not path.exists():
        return []
    text = path.read_text(encoding='utf-8')
    rows = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A killed append may leave an incomplete final record.
            if index != len(lines) - 1 or text.endswith('\n'):
                raise
    return rows


def repair_checkpoint(path):
    rows = read_checkpoint(path)
    if path.exists() and not path.read_bytes().endswith(b'\n'):
        atomic_write(path, ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    return rows


def atomic_write(path, text):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


def merge_checkpoints(output, count):
    manifests = []
    completed = {}
    calls = {}
    common = None
    for index in range(count):
        folder = output / f'worker-{index}'
        manifest_path = folder / 'manifest.json'
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['worker_index'] != index or manifest['worker_count'] != count:
            raise ValueError('Worker identity differs from requested execution')
        expected = manifest['all_ids'][index::count]
        if manifest['ids'] != expected or manifest['questions'] != len(expected):
            raise ValueError('Worker partition does not match global question order')
        identity = {k: v for k, v in manifest.items() if k not in ('worker_index', 'ids', 'questions')}
        if common is not None and identity != common:
            raise ValueError('Worker manifests differ; refusing to mix experiments')
        common = identity
        manifests.append(manifest)
        for item in read_checkpoint(folder / 'questions.jsonl'):
            qid = item['question_id']
            if qid not in expected or qid in completed:
                raise ValueError('Unexpected or duplicate completed question')
            completed[qid] = item
        for item in read_checkpoint(folder / 'model_calls.jsonl'):
            if item['key'] in calls:
                raise ValueError('Duplicate cached model call across workers')
            calls[item['key']] = item
    if common is None:
        return 0, 0
    manifest = {**common, 'questions': len(common['all_ids']), 'ids': common['all_ids'],
                'execution': 'isolated-gpu-workers'}
    target = output / 'manifest.json'
    if target.exists() and json.loads(target.read_text(encoding='utf-8')) != manifest:
        raise ValueError('Merged manifest changed; choose a new output folder')
    atomic_write(target, json.dumps(manifest, ensure_ascii=False, indent=2))
    ordered = {qid: completed[qid] for qid in manifest['ids'] if qid in completed}
    atomic_write(output / 'questions.jsonl', ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in ordered.values()))
    atomic_write(output / 'model_calls.jsonl', ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in calls.values()))
    from benchmark_model_trial import save_report
    from utils.retrieval_metrics import RAGEvaluator
    save_report(output, manifest, ordered, RAGEvaluator.__new__(RAGEvaluator))
    return len(ordered), manifest['questions']


def worker_environment(index, url):
    return {**os.environ, 'CUDA_VISIBLE_DEVICES': str(index),
            'OLLAMA_BASE_URL': url, 'QDRANT_PATH': ''}


def stop_workers(processes):
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='results/qwen-gpu-workers-top5')
    parser.add_argument('--ollama-port', type=int, default=11435)
    parser.add_argument('--merge-only', action='store_true')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('limit must be nonnegative')
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        done, total = merge_checkpoints(output, args.workers)
        print(f'Merged {done}/{total} completed questions')
        return
    processes, handles = [], []
    try:
        for index in range(args.workers):
            url = f'http://127.0.0.1:{args.ollama_port + index}'
            command = [sys.executable, '-u', str(ROOT / 'benchmark_model_trial.py'),
                       '--limit', str(args.limit), '--seed', str(args.seed),
                       '--worker-index', str(index), '--worker-count', str(args.workers),
                       '--ollama-url', url, '--output', str(output / f'worker-{index}')]
            handle = (output / f'worker-{index}.log').open('ab')
            handles.append(handle)
            processes.append(subprocess.Popen(command, cwd=ROOT, env=worker_environment(index, url),
                                               stdout=handle, stderr=subprocess.STDOUT))
        last_update = 0
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError(f'A GPU worker failed; inspect logs under {output}')
            if time.monotonic() - last_update >= 30:
                progress = [len(read_checkpoint(output / f'worker-{i}' / 'questions.jsonl')) for i in range(args.workers)]
                print(f'Completed per GPU: {progress}', flush=True)
                last_update = time.monotonic()
            time.sleep(2)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError(f'A GPU worker failed; inspect logs under {output}')
    finally:
        stop_workers(processes)
        for handle in handles:
            handle.close()
        done, total = merge_checkpoints(output, args.workers)
        print(f'Merged {done}/{total} completed questions', flush=True)
    if done != total or not total:
        raise RuntimeError('Run incomplete; worker checkpoints are preserved for resume')


if __name__ == '__main__':
    main()
