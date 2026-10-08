"""Qwen-only answer benchmark on Kaggle; never edits README results."""
import argparse
import ast
import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import random
import statistics
import time
import urllib.request

from benchmark_retrieval import digest, summarize
from utils.model_trial import make_trial_llm, invoke_trial
from langchain_core.messages import HumanMessage

ROOT = Path(__file__).resolve().parent


def save_report(output, manifest, completed, evaluator):
    rows = []
    for arm in ('original_query',):
        records = [r for item in completed.values() for r in item['arms'][arm]['records']]
        rows.extend({'arm': arm, **r} for r in summarize(records, evaluator))
    summary = {'manifest': manifest, 'completed_questions': len(completed), 'metrics': rows}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# Qwen 3.5 9B trial — pending review', '',
             f"Completed {len(completed)}/{manifest['questions']} questions. Qwen-only run; README results are unchanged.", '',
             '| Query arm | k | n | Precision | Recall | F1 | Hit rate | MAP | MRR | nDCG | Retrieval seconds |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        values = [r['arm'], str(r['k']), str(r['questions'])] + [f"{r[key]:.4f}" for key in ('precision', 'recall', 'f1', 'hit_rate', 'map', 'mrr', 'ndcg', 'mean_seconds')]
        lines.append('| ' + ' | '.join(values) + ' |')
    lines += ['', '## Model calls', '',
              '| Model | Mean answer seconds | Rate-limit retries | Mean input tokens | Mean output tokens | Output-cap hits |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ('qwen',):
        answers = [item['arms'][arm]['answer'] for item in completed.values()]
        if answers:
            tokens = []
            for name in ('input_tokens', 'output_tokens'):
                values = [a['usage'][name] for a in answers if isinstance(a['usage'].get(name), (int, float))]
                tokens.append(f'{statistics.mean(values):.1f}' if values else 'unavailable')
            cap_hits = sum(a.get('finish_reason') in ('length', 'max_tokens') or a['usage'].get('output_tokens', 0) >= 2048 for a in answers)
            lines.append(f"| {arm} | {statistics.mean(a['seconds'] for a in answers):.2f} | {sum(a['rate_limit_retries'] for a in answers)} | {tokens[0]} | {tokens[1]} | {cap_hits} |")
    lines += ['', '## Interpretation and review', '',
              '- Same seeded training questions, existing vectors, BGE, and 25+25 candidates. Original-query retrieval only, final top-5.',
              '- No refinement calls. Qwen answers the original question with top-5 evidence.',
              '- Qwen: thinking disabled, context 8192, output cap 2048.',
              '- Retrieval metrics are shared; answer quality still requires human review of answer_review.csv.',
              '- Output usage, finish reasons when available, latency and retry counts are saved. Deployment timings are not model throughput comparisons.',
              '- Training-set scores are exploratory; previous experiment files and README results are preserved.', '']
    (output / 'comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    fields = ['question_id', 'question', 'expected_ids', 'shared_evidence_ids', 'shared_evidence_text', 'qwen_query',
              'qwen_answer', 'qwen_faithfulness', 'qwen_citation_accuracy', 'qwen_clarity', 'qwen_pass', 'notes']
    review_path = output / 'answer_review.csv'
    previous_reviews = {}
    if review_path.exists():
        with review_path.open(encoding='utf-8-sig', newline='') as handle:
            previous_reviews = {r['question_id']: r for r in csv.DictReader(handle)}
    with review_path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in completed.values():
            writer.writerow({**previous_reviews.get(item['question_id'], {}), 'question_id': item['question_id'], 'question': item['question'],
                             'expected_ids': json.dumps(item['expected_ids'], ensure_ascii=False),
                             'shared_evidence_ids': json.dumps(item['shared_evidence_ids'], ensure_ascii=False),
                             'shared_evidence_text': item.get('shared_context', ''),
                             'qwen_query': item['question'],
                             'qwen_answer': item['arms']['qwen']['answer']['text']})


def shard_questions(rows, index, count):
    if count < 1 or not 0 <= index < count:
        raise ValueError('Invalid worker index/count')
    return rows[index::count]


def check_gpu_residency(models):
    model = next((m for m in models if m['name'] == 'qwen3.5:9b'), None)
    if not model or model.get('size_vram', 0) <= 0:
        raise RuntimeError('Qwen is not using GPU')
    if model.get('size', 0) and model['size_vram'] / model['size'] < 0.9:
        raise RuntimeError('Qwen is substantially offloaded to CPU; models do not fit on this GPU')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=0, help='0 runs all training questions')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='results/qwen-only-top5')
    parser.add_argument('--ollama-url', default='http://localhost:11434')
    parser.add_argument('--worker-index', type=int, default=0)
    parser.add_argument('--worker-count', type=int, default=1)
    args = parser.parse_args()
    if args.worker_count < 1 or not 0 <= args.worker_index < args.worker_count:
        parser.error('Invalid worker index/count')
    if args.limit < 0:
        parser.error('limit must be nonnegative')
    os.chdir(ROOT)
    import torch
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('The Qwen benchmark requires a Kaggle GPU runtime')
    from config import Config
    from main.chatbot import VietnameseLegalRAG
    from utils.retrieval_metrics import RAGEvaluator
    from importlib.metadata import version
    qwen = make_trial_llm('ollama', 'qwen3.5:9b', base_url=args.ollama_url)
    dataset = ROOT / 'data/train/train_qna.csv'
    with dataset.open(encoding='utf-8-sig', newline='') as handle:
        all_rows = list(csv.DictReader(handle))
    selected = random.Random(args.seed).sample(all_rows, min(args.limit, len(all_rows)) if args.limit else len(all_rows))
    all_ids = [r['question_id'] for r in selected]
    selected = shard_questions(selected, args.worker_index, args.worker_count)
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(args.ollama_url + '/api/tags', timeout=15) as response:
        models = json.load(response)['models']
    with urllib.request.urlopen(args.ollama_url + '/api/version', timeout=15) as response:
        ollama_version = json.load(response)['version']
    model = next(m for m in models if m['name'] == 'qwen3.5:9b')
    manifest = {'protocol': 'qwen-only-original-query-top5-v3', 'top_k': 5, 'refinement': False, 'worker_index': args.worker_index, 'worker_count': args.worker_count, 'all_ids': all_ids, 'questions': len(selected), 'seed': args.seed, 'ids': [r['question_id'] for r in selected],
                'dataset_sha256': digest(dataset), 'corpus_sha256': digest(ROOT / Config.CORPUS_PATH),
                'bm25_sha256': digest(ROOT / 'index/bm25_index.pkl'), 'qwen_model_digest': model['digest'],
                'qwen_model_details': model.get('details', {}), 'ollama_version': ollama_version,
                'qwen_model': 'qwen3.5:9b',
                'embedding': Config.EMBEDDING_MODEL, 'reranker': Config.RERANKER_MODEL,
                'candidate_k': Config.RERANK_BEFORE_RETRIEVAL_TOP_K, 'fusion_alpha': Config.RERANKER_FUSION_ALPHA,
                'use_fusion': Config.USE_SCORE_FUSION, 'characters': Config.MAX_ARTICLE_CHARACTERS,
                'threshold': Config.SIMILARITY_THRESHOLD, 'minimum_threshold': Config.MIN_SIMILARITY_FOR_LEGAL_DOCS,
                'reranker_batch_size': 4, 'torch_gpu': torch.cuda.get_device_name(0),
                'versions': {n: version(n) for n in ('torch', 'sentence-transformers', 'langchain-ollama', 'qdrant-client')},
                'code_hashes': {n: digest(ROOT / n) for n in ('benchmark_model_trial.py', 'utils/model_trial.py', 'main/chatbot.py', 'main/reranker.py', 'utils/retrieval_metrics.py', 'benchmark_multi_gpu.py')},
                'settings': {'temperature': 0.1, 'qwen_context': 8192, 'qwen_output_cap': 2048, 'qwen_thinking': False},
                'readme_sha256': digest(ROOT / 'README.md'), 'system_prompt_sha256': hashlib.sha256(Config.SYSTEM_PROMPT.encode()).hexdigest()}
    manifest_path = output / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Trial inputs/runtime changed. Use a new output folder; do not mix experiments.')
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    cache_file = output / 'model_calls.jsonl'
    from benchmark_multi_gpu import repair_checkpoint
    cache = {}
    if cache_file.exists():
        for entry in repair_checkpoint(cache_file):
            cache[entry['key']] = entry['value']

    def cached_call(key, function):
        if key not in cache:
            value = function()
            with cache_file.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps({'key': key, 'value': value}, ensure_ascii=False) + '\n')
            cache[key] = value
        return cache[key]

    completed = {}
    checkpoint = output / 'questions.jsonl'
    if checkpoint.exists():
        for item in repair_checkpoint(checkpoint):
            completed[item['question_id']] = item
    evaluator = RAGEvaluator.__new__(RAGEvaluator)
    save_report(output, manifest, completed, evaluator)
    if len(completed) == len(selected):
        print('Trial already complete; reports refreshed.')
        return
    rag = VietnameseLegalRAG()
    try:
        if not rag.vector_store or not rag.bm25_retriever or not rag.bm25_retriever.load_index():
            raise RuntimeError('Existing full indexes are required')
        from qdrant_client import QdrantClient
        options = {**rag.vector_store.client.init_options, 'timeout': 60}
        rag.vector_store.client.close()
        rag.vector_store.client = QdrantClient(**options)
        if rag.vector_store.get_collection_info()['points_count'] != 61068 or len(rag.bm25_retriever.documents) != 61068:
            raise RuntimeError('Indexes must contain all 61,068 articles')
        if not rag.reranker or not rag.reranker.model or rag.reranker.model.model.device.type != 'cuda':
            raise RuntimeError('BGE must be on GPU')
        if rag.vector_store.embedding_model.device.type != 'cuda':
            raise RuntimeError('The query embedding model must be on GPU')
        import functools
        rag.reranker.model.predict = functools.partial(rag.reranker.model.predict, batch_size=4)
        # Load Qwen with the full context allocation while both retrieval models
        # are resident. This tests each worker's combined memory footprint.
        request = urllib.request.Request(args.ollama_url + '/api/generate',
            data=json.dumps({'model': 'qwen3.5:9b', 'prompt': 'Reply OK.', 'think': False,
                             'stream': False, 'keep_alive': -1,
                             'options': {'num_ctx': 8192, 'num_predict': 1}}).encode(),
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=180) as response:
            json.load(response)
        with urllib.request.urlopen(args.ollama_url + '/api/ps', timeout=15) as response:
            check_gpu_residency(json.load(response)['models'])
        # Exercise retrieval allocations while Qwen remains resident.
        probe_question = max((r['question'] for r in selected), key=len, default='Kiểm tra')
        rag.vector_store.embedding_model.encode(probe_question)
        probe_evidence = ('quy định ' * 200)[:Config.MAX_ARTICLE_CHARACTERS]
        rag.reranker.model.predict([(probe_question, probe_evidence)] * 4)
        torch.cuda.synchronize()
        free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes < 256 * 1024**2:
            raise RuntimeError('Less than 256 MiB GPU memory headroom; reduce context or BGE batch size')
        print(f'Worker {args.worker_index}: combined models on GPU, free memory {free_bytes / 1024**2:.0f} MiB', flush=True)
        for row in selected:
            qid, question = row['question_id'], row['question']
            if qid in completed:
                continue
            expected = sorted({f"{r['law_id']}_{r['article_id']}" for r in ast.literal_eval(row['relevant_articles'])})
            started = time.perf_counter()
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture):
                shared_docs = rag.retrieve_documents(question, top_k=5)
            elapsed = time.perf_counter() - started
            if 'Error ' in capture.getvalue() or any('reranker_score' not in d for d in shared_docs):
                raise RuntimeError('Retrieval failure; trial stopped instead of scoring an error.')
            arms = {'original_query': {'records': [{'method': 'hybrid_rerank', 'k': 5,
                    'expected_ids': expected, 'retrieved_ids': [d['id'] for d in shared_docs],
                    'retrieval_seconds': elapsed}]}, 'qwen': {}}
            context = rag.format_context(shared_docs)
            prompt = Config.SYSTEM_PROMPT.format(context=context, question=question)
            for arm, llm in (('qwen', qwen),):
                key = qid + ':' + arm + ':answer:' + hashlib.sha256(prompt.encode()).hexdigest()
                arms[arm]['answer'] = cached_call(key, lambda llm=llm: invoke_trial(llm, [HumanMessage(content=prompt)]))
            with urllib.request.urlopen(args.ollama_url + '/api/ps', timeout=15) as response:
                running = json.load(response)['models']
            check_gpu_residency(running)
            item = {'question_id': qid, 'question': question, 'expected_ids': expected,
                    'shared_evidence_ids': [d['id'] for d in shared_docs], 'shared_context': context, 'arms': arms}
            with checkpoint.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(item, ensure_ascii=False) + '\n')
            completed[qid] = item
            save_report(output, manifest, completed, evaluator)
            print(f'Qwen benchmark completed {len(completed)}/{len(selected)}', flush=True)
    finally:
        if rag.vector_store:
            rag.vector_store.client.close()


if __name__ == '__main__':
    main()
