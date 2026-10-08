"""Top-5 retrieval ablations anchored to the completed Qwen/BGE run. No LLM or BGE."""
import argparse
import ast
import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import time
from importlib.metadata import version

from utils.ablation import METHODS, rank_candidates, summarize_records

ROOT = Path(__file__).resolve().parent


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def baseline_rows(folder):
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in (folder / 'questions.jsonl').read_text(encoding='utf-8').splitlines()]
    ids = [r['question_id'] for r in rows]
    if len(set(ids)) != len(ids) or ids != manifest['ids'] or len(rows) != manifest['questions']:
        raise ValueError('Baseline is incomplete, duplicated, or out of order')
    if manifest.get('protocol') != 'qwen-only-original-query-top5-v3':
        raise ValueError('Unexpected baseline protocol')
    if manifest['top_k'] != 5 or manifest['refinement'] or manifest['candidate_k'] != 25:
        raise ValueError('Expected original-query top-5 baseline with 25+25 candidates')
    compact = []
    for r in rows:
        record = r['arms']['original_query']['records']
        if len(record) != 1 or record[0]['k'] != 5:
            raise ValueError('Baseline ranking is not top-5 only')
        if set(record[0]['expected_ids']) != set(r['expected_ids']):
            raise ValueError('Baseline record labels are inconsistent')
        if len(record[0]['retrieved_ids']) != len(set(record[0]['retrieved_ids'])):
            raise ValueError('Baseline contains duplicate retrieved IDs')
        compact.append({'question_id': r['question_id'], 'question': r['question'],
                        'expected_ids': r['expected_ids'], 'record': record[0]})
    fingerprint = hashlib.sha256(json.dumps(compact, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return manifest, compact, fingerprint


def validate_dataset(rows, path, baseline):
    if digest(path) != baseline['dataset_sha256']:
        raise ValueError('Dataset differs from the baseline')
    with path.open(encoding='utf-8-sig', newline='') as handle:
        dataset = {r['question_id']: r for r in csv.DictReader(handle)}
    for row in rows:
        source = dataset[row['question_id']]
        expected = {f"{x['law_id']}_{x['article_id']}" for x in ast.literal_eval(source['relevant_articles'])}
        if row['question'] != source['question'] or set(row['expected_ids']) != expected:
            raise ValueError('Question or gold labels differ from the baseline')


def strict_retrieve(function, query, k):
    capture = io.StringIO()
    started = time.perf_counter()
    with contextlib.redirect_stdout(capture):
        docs = function(query, top_k=k)
    seconds = time.perf_counter() - started
    if 'error' in capture.getvalue().casefold():
        raise RuntimeError('Retriever reported an error; refusing to score a failed call: ' + capture.getvalue()[-400:])
    if any(not d.get('id') or 'score' not in d for d in docs):
        raise ValueError('Malformed retrieval candidate')
    return docs, seconds


def write_report(output, manifest, completed, rows):
    metrics = []
    for method in manifest['methods']:
        records = [r for r in completed.values() if r['method'] == method]
        summary = summarize_records(records, method)
        if summary:
            metrics.append(summary)
    # Compare on the exact question subset for each method, including partial runs.
    baselines = []
    by_id = {r['question_id']: r for r in rows}
    for metric in metrics:
        ids = {r['question_id'] for r in completed.values() if r['method'] == metric['method']}
        base = summarize_records([by_id[qid]['record'] for qid in by_id if qid in ids], 'hybrid_bge')
        base['compared_to'] = metric['method']
        baselines.append(base)
    payload = {'manifest': manifest, 'completed_records': len(completed), 'metrics': metrics,
               'matched_baseline_metrics': baselines}
    (output / 'summary.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# Top-5 retrieval ablations — pending review', '',
             'README results are unchanged. No generation, refinement, BGE inference, or re-embedding.', '',
             '| Method | n | P@5 | R@5 | F1@5 | Hit@5 | MAP@5 | MRR@5 | nDCG@5 | Mean s | P95 s |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    display = metrics + baselines
    if baselines and all(b['questions'] == len(rows) for b in baselines):
        display = metrics + [baselines[0]]
    for r in display:
        label = r['method']
        if r['method'] == 'hybrid_bge' and r['questions'] != len(rows):
            label += ' (matched to ' + r['compared_to'] + ')'
        values = [label, str(r['questions'])] + [f"{r[k]:.4f}" for k in ('precision','recall','f1','hit_rate','map','mrr','ndcg','mean_seconds','p95_seconds')]
        lines.append('| ' + ' | '.join(values) + ' |')
    lines += ['', '## Changes relative to the matched BGE baseline', '',
              '| Ablation | Baseline minus ablation recall (pp) | Baseline minus ablation nDCG |',
              '|---|---:|---:|']
    for r, b in zip(metrics, baselines):
        lines.append(f"| {r['method']} | {100*(b['recall']-r['recall']):.2f} | {b['ndcg']-r['ndcg']:.4f} |")
    lines += ['', '## Protocol', '',
              '- Fixed top-25 sparse/dense candidates, matching the existing BGE candidate pools; final output is top-5.',
              '- Existing query preprocessing and per-list score normalization are retained. Hybrid deduplicates and uses the maximum normalized score, matching pre-BGE ranking. Ties retain sparse-first insertion order.',
              '- Qdrant uses the same raw cosine threshold as the baseline. No additional final-score threshold is applied: removing BGE does not also change final filtering.',
              '- Dense and BM25 timings include their candidate search and final ranking. Hybrid timing is the sum of both searches plus fusion; candidate searches are shared when multiple ablations run together.',
              '- Metrics use unique exact article IDs, macro averaging and binary nDCG. Training-set results are not held-out evidence or answer-accuracy scores.',
              '- Baseline timing comes from the earlier dual-T4 run. Ablation hardware is recorded separately; cross-hardware latency is not a causal speedup claim.',
              '- Partial reports use the same completed question IDs for each baseline comparison. Resume only with matching manifests.', '']
    (output / 'comparison.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--baseline', default='results/qwen-gpu-workers-top5')
    parser.add_argument('--output', default='results/ablation-top5')
    parser.add_argument('--limit', type=int, default=0, help='0 uses all saved baseline questions')
    parser.add_argument('--require-gpu', action='store_true')
    args = parser.parse_args()
    if args.limit < 0 or len(set(args.methods)) != len(args.methods):
        parser.error('Use a nonnegative limit and unique methods')
    os.chdir(ROOT)
    from config import Config
    baseline, rows, fingerprint = baseline_rows(ROOT / args.baseline)
    if args.limit:
        rows = rows[:args.limit]
    validate_dataset(rows, ROOT / 'data/train/train_qna.csv', baseline)
    for key, path in (('corpus_sha256', Config.CORPUS_PATH), ('bm25_sha256', 'index/bm25_index.pkl')):
        if digest(ROOT / path) != baseline[key]:
            raise ValueError(f'{key} differs from baseline')
    needs_sparse = any(m in args.methods for m in ('bm25', 'hybrid'))
    needs_dense = any(m in args.methods for m in ('dense', 'hybrid'))
    sparse, vector = None, None
    hardware = {'platform': platform.platform(), 'cpu': platform.processor(), 'gpu': None}
    try:
        if needs_sparse:
            from main.bm25_retriever import BM25Retriever
            sparse = BM25Retriever()
            if not sparse.load_index() or len(sparse.documents) != 61068:
                raise RuntimeError('Complete existing BM25 index required')
        if needs_dense:
            import torch
            torch.set_num_threads(2)
            if args.require_gpu and not torch.cuda.is_available():
                raise RuntimeError('GPU requested but unavailable')
            if Config.EMBEDDING_MODEL != baseline['embedding'] or Config.COLLECTION_NAME not in ('bkai_biencoder_vietnamese_legal_corpus',):
                raise ValueError('Embedding/collection differs from verified baseline')
            if Config.SIMILARITY_THRESHOLD != baseline['threshold']:
                raise ValueError('Dense cosine threshold differs from baseline')
            from main.vector_store import QdrantVectorStore
            vector = QdrantVectorStore()
            if not vector.client or vector.embedding_model is None or vector.get_collection_info().get('points_count') != 61068:
                raise RuntimeError('Complete existing vector index and embedding model required')
            from qdrant_client import QdrantClient
            options = {**vector.client.init_options, 'timeout': 60}
            vector.client.close()
            vector.client = QdrantClient(**options)
            if args.require_gpu and vector.embedding_model.device.type != 'cuda':
                raise RuntimeError('Query embeddings must run on GPU')
            hardware['gpu'] = torch.cuda.get_device_name(0) if vector.embedding_model.device.type == 'cuda' else None
            vector.embedding_model.encode(rows[0]['question'])
            if hardware['gpu']:
                torch.cuda.synchronize()
        manifest = {'protocol': 'fixed-candidates-top5-ablation-v1', 'methods': args.methods,
                    'questions': len(rows), 'ids': [r['question_id'] for r in rows], 'baseline_fingerprint': fingerprint,
                    'baseline_dataset_sha256': baseline['dataset_sha256'], 'baseline_corpus_sha256': baseline['corpus_sha256'],
                    'baseline_bm25_sha256': baseline['bm25_sha256'], 'embedding': baseline['embedding'] if needs_dense else None,
                    'candidate_k': 25, 'top_k': 5, 'hardware': hardware,
                    'threshold': baseline['threshold'] if needs_dense else None,
                    'stopwords_sha256': digest(ROOT / Config.STOPWORDS_PATH),
                    'versions': {n: version(n) for n in ('numpy','rank-bm25','underthesea') + (('torch','sentence-transformers','qdrant-client') if needs_dense else ())},
                    'code_hashes': {n: digest(ROOT/n) for n in ('benchmark_ablation.py','utils/ablation.py','main/bm25_retriever.py','utils/text_processor.py') + (('main/vector_store.py',) if needs_dense else ())}}
        output = ROOT / args.output
        output.mkdir(parents=True, exist_ok=True)
        path = output / 'manifest.json'
        if path.exists() and json.loads(path.read_text(encoding='utf-8')) != manifest:
            raise ValueError('Ablation settings changed; use a new output folder')
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        from benchmark_multi_gpu import repair_checkpoint
        checkpoint = output / 'records.jsonl'
        completed = {}
        gold = {r['question_id']: set(r['expected_ids']) for r in rows}
        for r in repair_checkpoint(checkpoint):
            key = (r['question_id'], r['method'])
            if key in completed or r['question_id'] not in manifest['ids'] or r['method'] not in args.methods:
                raise ValueError('Unexpected or duplicate checkpoint record')
            if r['k'] != 5 or set(r['expected_ids']) != gold[r['question_id']] or len(r['retrieved_ids']) > 5 or len(r['retrieved_ids']) != len(set(r['retrieved_ids'])):
                raise ValueError('Malformed checkpoint ranking or labels')
            completed[key] = r
        write_report(output, manifest, completed, rows)
        for index, row in enumerate(rows):
            pending = [m for m in args.methods if (row['question_id'], m) not in completed]
            if not pending:
                continue
            sparse_docs, sparse_s = strict_retrieve(sparse.get_relevant_documents, row['question'], 25) if any(m in pending for m in ('bm25','hybrid')) else ([], 0)
            dense_docs, dense_s = strict_retrieve(vector.search_similar_documents, row['question'], 25) if any(m in pending for m in ('dense','hybrid')) else ([], 0)
            for method in pending:
                started = time.perf_counter()
                docs = rank_candidates(sparse_docs, dense_docs, method)
                elapsed = time.perf_counter() - started
                elapsed += (sparse_s if method in ('bm25','hybrid') else 0) + (dense_s if method in ('dense','hybrid') else 0)
                record = {'question_id': row['question_id'], 'method': method, 'k': 5,
                          'expected_ids': row['expected_ids'], 'retrieved_ids': [d['id'] for d in docs], 'retrieval_seconds': elapsed}
                with checkpoint.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                completed[(row['question_id'], method)] = record
            if (index+1) % 25 == 0:
                write_report(output, manifest, completed, rows)
                print(f'Processed through {index+1}/{len(rows)} questions; {len(completed)} method records', flush=True)
        write_report(output, manifest, completed, rows)
        print(f'Complete: {len(completed)} records. Reports: {output}', flush=True)
    finally:
        if vector and vector.client:
            vector.client.close()


if __name__ == '__main__':
    main()
