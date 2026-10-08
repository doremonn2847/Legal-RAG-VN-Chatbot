"""Reproducible retrieval-only 5/10 comparison; never edits README.md."""
import argparse
import ast
import contextlib
import csv
import hashlib
import io
import json
import platform
import random
import functools
import statistics
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
METHODS = ('bm25', 'vector', 'hybrid', 'hybrid_rerank')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(records, evaluator):
    rows = []
    for method in METHODS:
        for k in (5, 10):
            group = [r for r in records if r['method'] == method and r['k'] == k]
            if not group:
                continue
            measurements = []
            for r in group:
                ids, expected = r['retrieved_ids'], r['expected_ids']
                p = evaluator.calculate_precision_at_k(ids, expected, k)
                recall = evaluator.calculate_recall_at_k(ids, expected, k)
                measurements.append({
                    'precision': p, 'recall': recall,
                    'f1': evaluator.calculate_f1_at_k(p, recall),
                    'hit_rate': float(bool(set(ids).intersection(expected))),
                    'map': evaluator.calculate_average_precision(ids, expected),
                    'mrr': evaluator.calculate_reciprocal_rank(ids, expected),
                    'ndcg': evaluator.calculate_ndcg_at_k(ids, expected, k),
                })
            rows.append({'method': method, 'k': k, 'questions': len(group),
                         **{name: statistics.mean(m[name] for m in measurements) for name in measurements[0]},
                         'mean_seconds': statistics.mean(r['retrieval_seconds'] for r in group),
                         'p95_seconds': sorted(r['retrieval_seconds'] for r in group)[max(0, int(.95 * len(group) + .999) - 1)]})
    return rows


def write_report(manifest, records, evaluator, output):
    rows = summarize(records, evaluator)
    (output / 'summary.json').write_text(json.dumps({'manifest': manifest, 'metrics': rows}, indent=2, ensure_ascii=False), encoding='utf-8')
    completed = len({r['question_id'] for r in records})
    lines = ['# Retrieval benchmark comparison — pending review', '',
             f"Status: {'complete' if completed == manifest['question_count'] else 'in progress'}. {completed}/{manifest['question_count']} questions scored.", '',
             'README.md and its historical Results table have not been changed.', '',
             '## Proposed results table', '',
             '| Method | Final k | Questions | P@k | R@k | F1@k | Hit@k / coverage | MAP@k | MRR@k | nDCG@k | Mean retrieval s | P95 s |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        values = [r['method'], str(r['k']), str(r['questions'])] + [f"{r[x]:.4f}" for x in ('precision', 'recall', 'f1', 'hit_rate', 'map', 'mrr', 'ndcg', 'mean_seconds', 'p95_seconds')]
        lines.append('| ' + ' | '.join(values) + ' |')
    lines += ['', '## Protocol and reproducibility', '',
              f"- Dataset: labeled train_qna.csv; {manifest['question_count']} of {manifest['dataset_total']} questions; selection seed {manifest['seed']}. Public test labels are unavailable.",
              f"- Query refinement: {manifest['refine']}. The same saved query is used for every method and both cutoffs. Generation, domain detection and web fallback are excluded.",
              '- Existing full-corpus BM25 and Qdrant indexes; no re-embedding. Article IDs must match exactly. Repeated IDs receive no additional relevance credit.',
              '- BGE reranks up to 25 sparse + 25 dense candidates after deduplication; fused score weight 0.8. Excerpts are limited to 1,000 characters. Final top 5 is the prefix of the same top 10 ranking.',
              '- Non-reranked modes are separately run at 5 and 10 because their candidate pool depends on k. BGE runs once per question; shared inference time is reported for both cutoffs, excluding refinement and initialization.',
              '- Benchmark-only Qdrant timeout: 60 seconds. An initial attempt hit the default timeout (the server took 9.86 seconds); no failed questions are counted as relevance misses.',
              '- Metrics are macro averages. Precision denominator is k; recall denominator is all unique labeled articles. MAP denominator is all labeled articles (matching the upstream convention). MRR is truncated at the output cutoff.',
              '- nDCG uses binary exact-ID relevance and discount 1/log2(rank+1), normalized by an ideal ranking of min(k, number of labeled articles). Coverage equals hit rate at the specified cutoff.',
              '- This measures retrieval relevance, not answer accuracy, faithfulness or legal currency. A training sample is not a held-out test; do not treat it as directly comparable to the historical full-training scores.',
              f"- Hardware: {manifest['hardware']}; PyTorch threads: {manifest['threads']}; BGE inference batch size: {manifest['reranker_batch_size']}.",
              f"- Raw records and metadata: `{output.relative_to(ROOT).as_posix()}` (local, ignored by Git).", '',
              'Reproduce from the project root:', '', '```powershell', manifest['command'], '```', '',
              'Reuse the same output directory to resume. Use a new directory for a fresh timed run. The manifest records dataset/code hashes, selected IDs, model names, versions and configuration.', '',
              '## Historical README table', '',
              '| Method | MRR | Coverage | R@1 | R@10 | R@20 | MAP@20 |',
              '|---|---:|---:|---:|---:|---:|---:|',
              '| Sparse TellOnly | 0.5545 | 0.7894 | 0.430 | 0.768 | 0.783 | 0.565 |',
              '| Dense Only | 0.4691 | 0.6809 | 0.364 | 0.666 | 0.673 | 0.471 |',
              '| Hybrid (Sparse + Dense) | 0.5801 | 0.8820 | 0.431 | 0.833 | 0.875 | 0.592 |',
              '| Hybrid + Reranking | 0.6082 | 0.8899 | 0.482 | 0.827 | 0.884 | 0.624 |', '',
              'Historical results used the former MS MARCO reranker and the full training set. nDCG was not reported; the historical rankings are unavailable, so it cannot be reconstructed. Old scoring accepted substring IDs; these new scores use exact IDs.', '']
    report = ROOT / 'docs' / 'benchmark-comparison.md'
    report.parent.mkdir(exist_ok=True)
    report.write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=50, help='0 for all labeled training questions')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--reranker-batch-size', type=int, default=8)
    parser.add_argument('--require-gpu', action='store_true')
    parser.add_argument('--no-refine', action='store_true', default=True, help='Original queries; refinement is disabled')
    parser.add_argument('--output', default='results/benchmark-5-10')
    args = parser.parse_args()
    if args.limit < 0 or args.threads < 1 or args.reranker_batch_size < 1:
        parser.error('limit must be nonnegative; threads and batch size must be positive')
    import os
    os.chdir(ROOT)
    import torch
    torch.set_num_threads(args.threads)
    if args.require_gpu and not torch.cuda.is_available():
        raise RuntimeError('A GPU runtime is required for this run')
    from importlib.metadata import version
    from config import Config
    from main.chatbot import VietnameseLegalRAG
    from utils.retrieval_metrics import RAGEvaluator

    dataset = ROOT / 'data/train/train_qna.csv'
    with dataset.open(encoding='utf-8-sig', newline='') as handle:
        all_questions = list(csv.DictReader(handle))
    questions = all_questions if args.limit == 0 else random.Random(args.seed).sample(all_questions, min(args.limit, len(all_questions)))
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    command = f'python -u benchmark_retrieval.py --limit {args.limit} --seed {args.seed} --threads {args.threads} --reranker-batch-size {args.reranker_batch_size}' + (' --require-gpu' if args.require_gpu else '') + (' --no-refine' if args.no_refine else '') + f' --output {args.output}'
    manifest = {
        'dataset_sha256': digest(dataset), 'dataset_total': len(all_questions), 'question_count': len(questions),
        'selected_ids': [q['question_id'] for q in questions], 'seed': args.seed,
        'refine': not args.no_refine, 'threads': args.threads, 'command': command,
        'reranker_batch_size': args.reranker_batch_size,
        'qdrant_timeout_seconds': 60,
        'embedding_model': Config.EMBEDDING_MODEL, 'reranker_model': Config.RERANKER_MODEL,
        'refinement_model': Config.MODEL_REFINE, 'collection': Config.COLLECTION_NAME,
        'refinement_settings': {name: getattr(Config, name) for name in ('ENABLE_CHAIN_OF_THOUGHT', 'ENABLE_ITERATIVE_REFINEMENT', 'ENABLE_LLM_VALIDATION', 'MAX_REFINEMENT_ITERATIONS', 'MIN_CONFIDENCE_SCORE')},
        'candidate_k': Config.RERANK_BEFORE_RETRIEVAL_TOP_K, 'fusion_alpha': Config.RERANKER_FUSION_ALPHA,
        'use_score_fusion': Config.USE_SCORE_FUSION, 'max_characters': Config.MAX_ARTICLE_CHARACTERS,
        'similarity_threshold': Config.SIMILARITY_THRESHOLD, 'minimum_similarity': Config.MIN_SIMILARITY_FOR_LEGAL_DOCS,
        'hardware': f'{platform.system()} {platform.machine()} {platform.processor()}; device {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}',
        'versions': {name: version(name) for name in ('torch', 'sentence-transformers', 'qdrant-client', 'langchain-ollama')},
        'code_hashes': {name: digest(ROOT / name) for name in ('main/chatbot.py', 'main/reranker.py', 'utils/question_refiner.py', 'utils/retrieval_metrics.py', 'benchmark_retrieval.py')},
        'corpus_sha256': digest(ROOT / Config.CORPUS_PATH),
        'bm25_sha256': digest(ROOT / 'index/bm25_index.pkl'),
        'readme_sha256_before_run': digest(ROOT / 'README.md'),
    }
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
            raise ValueError('Existing run has different inputs/configuration. Choose a new --output directory.')
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    checkpoint = output / 'questions.jsonl'
    query_checkpoint = output / 'queries.jsonl'
    queries = {}
    if query_checkpoint.exists():
        for line in query_checkpoint.read_text(encoding='utf-8').splitlines():
            item = json.loads(line)
            queries[item['question_id']] = item
    completed = {}
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding='utf-8').splitlines():
            item = json.loads(line)
            completed[item['question_id']] = item
    evaluator = RAGEvaluator.__new__(RAGEvaluator)
    records = [record for item in completed.values() for record in item['records']]
    write_report(manifest, records, evaluator, output)
    if len(completed) == len(questions):
        print('All questions already complete; report regenerated.', flush=True)
        return
    rag = VietnameseLegalRAG()
    evaluator.rag = rag
    try:
        if rag.vector_store:
            from qdrant_client import QdrantClient
            options = dict(rag.vector_store.client.init_options)
            options['timeout'] = 60
            rag.vector_store.client.close()
            rag.vector_store.client = QdrantClient(**options)
        if not rag.vector_store or not rag.bm25_retriever or not rag.bm25_retriever.load_index():
            raise RuntimeError('Required existing indexes are unavailable')
        if not rag.reranker or rag.reranker.model is None:
            raise RuntimeError('BGE is unavailable; refusing to mislabel an unreranked benchmark')
        if args.require_gpu and (rag.reranker.model.model.device.type != 'cuda' or rag.vector_store.embedding_model.device.type != 'cuda'):
            raise RuntimeError('Both neural models must be on GPU')
        rag.reranker.model.predict = functools.partial(rag.reranker.model.predict, batch_size=args.reranker_batch_size)
        if not args.no_refine and (not rag.question_refiner or not rag.question_refiner.llm):
            raise RuntimeError('Local refinement is unavailable; use --no-refine for an explicitly raw-query run')
        count = rag.vector_store.get_collection_info()['points_count']
        if count != len(rag.bm25_retriever.documents) or count != 61068:
            raise RuntimeError(f'Expected full matching 61,068-article indexes; Qdrant has {count}')
        for row in questions:
            if row['question_id'] in completed:
                continue
            query = row['question']
            refinement_seconds = 0.0
            refinement = None
            saved_query = queries.get(row['question_id'])
            if saved_query:
                query = saved_query['retrieval_query']
                refinement_seconds = saved_query['refinement_seconds']
                refinement = saved_query['refinement']
            elif not args.no_refine:
                start = time.perf_counter()
                refinement = rag.question_refiner.refine_question(query)
                query = refinement['refined_question']
                refinement_seconds = time.perf_counter() - start
            if not saved_query:
                saved_query = {'question_id': row['question_id'], 'retrieval_query': query,
                               'refinement_seconds': refinement_seconds, 'refinement': refinement}
                with query_checkpoint.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(saved_query, ensure_ascii=False) + '\n')
                queries[row['question_id']] = saved_query
            expected = sorted({f"{a['law_id']}_{a['article_id']}" for a in ast.literal_eval(row['relevant_articles'])})
            query_records = []
            for method in METHODS:
                shared = None
                for k in ((10, 5) if method == 'hybrid_rerank' else (5, 10)):
                    if shared is None:
                        start = time.perf_counter()
                        capture = io.StringIO()
                        with contextlib.redirect_stdout(capture):
                            docs = evaluator.retrieve_documents_with_method(query, method, k)
                        elapsed = time.perf_counter() - start
                        if 'Error ' in capture.getvalue() or 'Error during' in capture.getvalue():
                            raise RuntimeError(f'Retrieval failed for {method}: {capture.getvalue()}')
                        if method == 'hybrid_rerank':
                            if any('reranker_score' not in doc for doc in docs):
                                raise RuntimeError('BGE scoring failed')
                            shared = (docs, elapsed)
                    else:
                        docs, elapsed = shared
                    query_records.append({'question_id': row['question_id'], 'method': method, 'k': k,
                                          'expected_ids': expected, 'retrieved_ids': [d['id'] for d in docs[:k]],
                                          'scores': [float(d.get('score', 0)) for d in docs[:k]],
                                          'retrieval_seconds': elapsed})
            item = {'question_id': row['question_id'], 'original_query': row['question'], 'retrieval_query': query,
                    'refinement_seconds': refinement_seconds, 'refinement': refinement, 'records': query_records}
            with checkpoint.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(item, ensure_ascii=False) + '\n')
            completed[row['question_id']] = item
            records.extend(query_records)
            write_report(manifest, records, evaluator, output)
            print(f"Completed {len(completed)}/{len(questions)}; BGE retrieval {query_records[-1]['retrieval_seconds']:.1f}s", flush=True)
    finally:
        if rag.vector_store:
            rag.vector_store.client.close()


if __name__ == '__main__':
    main()
