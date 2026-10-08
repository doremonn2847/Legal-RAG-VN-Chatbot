# Matched top-5 retrieval ablations

These scripts evaluate BM25-only, dense-only, and hybrid without BGE against the verified 3,196-question hybrid+BGE baseline. The application and historical README Results table are unchanged.

Completed and verified October 8, 2026: all 9,588 method/question records. See the [results review](ablation-results-review.md) for the matched table, per-question changes, latency limits, and a supported CV claim.

## Local BM25

From the project root, with the existing dataset/index and extracted full-run results:

```powershell
.venv/Scripts/python.exe -X utf8 benchmark_ablation.py --methods bm25 --output results/ablation-bm25-local
```

This loads only BM25 and query preprocessing: no Qwen, BGE, embedding model, or Qdrant connection. An existing BM25 pickle is still required, as verified against the full-run index hash. No indexes are rebuilt.

For a three-question smoke run, add `--limit 3` and use a separate output folder. The limit takes the first questions in the saved baseline order. Reusing a folder with changed methods, limit, source or runtime fails rather than mixing settings.

## Kaggle: all three methods

Import `notebooks/kaggle_retrieval_ablation.ipynb`, enable GPU and internet, attach the existing `colab_benchmark_inputs.zip` Dataset, and Run All. One GPU is sufficient. It includes the verified baseline questions and retrieval rankings, without bundling generated answers; no result ZIP or API key is needed. Qdrant starts with the existing vectors, and only the Vietnamese query embedding model is loaded. No Ollama or reranker is started.

Download `/kaggle/working/ablation_top5_results.zip` after completion. Partial method records are checkpointed after each call and can be exported before discarding a runtime. Attach that ZIP as a Dataset to resume with matching source/runtime/methods. Both ZIP and automatically unpacked checkpoint layouts are supported.

The local equivalent, if query embeddings and Qdrant are available, is:

```powershell
.venv/Scripts/python.exe -X utf8 benchmark_ablation.py --methods bm25 dense hybrid --output results/ablation-top5
```

Add `--require-gpu` to reject CPU query embeddings. BM25 alone needs no GPU. The vector index must contain all 61,068 articles and use the same embedding model; dataset, corpus, and BM25 hashes are checked against the full baseline.

## Comparison rules

All methods use original questions, the same exact gold article IDs, and final top-5. Each enabled retriever gathers 25 candidates. This deliberately differs from the older generic evaluator, which gathered only five candidates without reranking; keeping 25 isolates removal of BGE from candidate-pool changes.

Candidate score normalization and dense cosine filtering follow the existing implementation. Hybrid merges by article ID and uses the maximum normalized score, with stable sparse-first ties, matching the current pre-BGE ranking. There is no additional normalized-score threshold after ranking, matching the BGE branch's final selection policy. Existing normalization edge cases are retained rather than silently corrected during the ablation.

Reports include precision, recall, F1, hit rate, MAP, MRR, binary nDCG, mean/p95 latency, and BGE-baseline-minus-ablation recall/nDCG differences. Partial reports compare each method only against the same completed question IDs. Searches are shared between methods, but hybrid time includes both search durations plus fusion. This is component timing within a shared run, not separately measured isolated-service throughput. Query embeddings are warmed once before timed dense calls.

`summary.json` and `comparison.md` live beside `manifest.json` and raw `records.jsonl`. BM25 CPU hardware and Kaggle GPU hardware are labeled separately; do not claim cross-machine latency differences as the causal effect of reranking. Baseline latency also came from two concurrent T4 workers, whereas this ablation notebook uses one query-embedding worker.

These are training-set retrieval measurements, not generated-answer accuracy or held-out quality estimates. Review the ablation results before proposing a new README table or CV improvement claim.

Regenerate the packaged notebook after source changes:

```powershell
.venv/Scripts/python.exe -X utf8 tools/create_kaggle_ablation_notebook.py
```
