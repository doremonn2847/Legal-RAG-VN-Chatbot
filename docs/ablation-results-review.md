# Verified retrieval ablation results

Reviewed October 8, 2026. All three ablations completed **3,196 questions each**, with **9,588 unique method/question records**. The current hybrid+BGE pipeline has the highest retrieval scores in this matched training-set comparison. The README Results section remains unchanged pending user review.

## Final comparison table

All methods use original queries, fixed top-25 candidates per enabled retriever, exact gold article IDs, and final top-5. Metrics are macro averages. The BGE baseline comes from the independently verified full Qwen run; Qwen does not affect the retrieval rankings.

| Method | P@5 | R@5 | F1@5 | Hit@5 | MAP@5 | MRR@5 | nDCG@5 | Mean retrieval s | P95 retrieval s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 only | 0.1430 | 0.7020 | 0.2370 | 0.7093 | 0.5371 | 0.5429 | 0.5800 | 0.4205 | 0.9473 |
| Dense only | 0.1210 | 0.5922 | 0.2004 | 0.6001 | 0.4489 | 0.4544 | 0.4862 | 0.0219 | 0.0246 |
| Hybrid without BGE | 0.1524 | 0.7476 | 0.2526 | 0.7547 | 0.5576 | 0.5633 | 0.6069 | 0.4423 | 0.9690 |
| **Hybrid + BGE** | **0.1681** | **0.8236** | **0.2785** | **0.8314** | **0.6715** | **0.6781** | **0.7113** | 2.7192 | 3.4635 |

MAP divides by the total number of unique gold articles. nDCG uses binary relevance and logarithmic discount. Dense retrieval retains the existing raw cosine threshold of 0.25. Hybrid deduplicates candidates and ranks by the maximum per-list normalized score with stable sparse-first ties. It retains existing normalization edge cases and applies no additional final-score threshold. This measures the current implementation, not every possible hybrid-fusion strategy.

## What the ablations establish

Hybrid+BGE improves Recall@5 over BM25 by **12.17 percentage points**, from 70.20% to 82.36%. Its nDCG@5 increases from 0.5800 to 0.7113, an absolute gain of 0.1313.

Combining dense and sparse retrieval improves recall over BM25 by **4.56 points**. Adding BGE to hybrid adds another **7.60 points** and increases nDCG by 0.1044. Dense retrieval alone underperforms BM25 on this benchmark; the hybrid pipeline still benefits from combining both candidate sources.

The aggregate gains do not mean reranking helps every query. Per-question Recall@5 comparisons with the BGE baseline are:

| Compared with | BGE baseline better | BGE baseline worse | Unchanged |
|---|---:|---:|---:|
| BM25 | 505 | 107 | 2,584 |
| Dense | 795 | 46 | 2,355 |
| Hybrid without BGE | 322 | 72 | 2,802 |

Per-question deltas are saved in `results/ablation-top5-kaggle/paired_comparison.csv`; the 72 hybrid recall regressions are useful candidates for targeted retrieval review.

## Verification and evidence

The original `results/ablation_top5_results.zip` is preserved. Its four raw exported files are extracted into `results/ablation-top5-kaggle/`. Derived `verification.json` and `paired_comparison.csv` are separate from the archived evidence.

- Every method covers exactly the saved 3,196 baseline question IDs, without duplicate records or duplicate ranked IDs.
- Gold labels match both the baseline and the local training CSV. The baseline fingerprint, dataset/corpus/BM25 hashes, stopword hash, candidate limit and cosine threshold match the expected inputs.
- All seven retrieval metrics, mean latency and p95 latency were independently recomputed from raw records and match the exported values within 1e-10.
- The three matched baseline summaries reproduce the independently computed hybrid+BGE metrics.
- All five ablation source hashes match the current local files.
- The local BM25 run also completed all 3,196 questions. Its ordered top-5 article IDs are identical to Kaggle's for every question.

The ablation runtime records a Tesla T4, PyTorch 2.11.0+cu128, Sentence Transformers 5.7.0, Qdrant client 1.19.1, rank-bm25 0.2.2, and underthesea 9.5.0. It makes no LLM calls, does not load BGE, and does not re-embed the corpus.

## Latency and quality limits

The table uses Kaggle ablation timings and the prior Kaggle BGE baseline timings. The ablations share candidate searches across methods and run one query-embedding worker; the baseline ran two concurrent workers, each with Qwen and BGE resident. The observed latency trade-off is useful, but it is not a controlled measurement of BGE's isolated incremental time or a deployment throughput benchmark.

Local BM25 has a median retrieval time of 0.4145 seconds and p95 of 0.9541 seconds, but its mean is 1.6816 seconds because the saved timings include a maximum of 3,610.02 seconds. The raw evidence does not establish the cause of that outlier; a pause or resource stall is possible. Do not use that local mean as steady-state throughput or quietly replace it with a filtered mean.

These are **training-set retrieval results**, not held-out generalization or answer-accuracy measurements. Generated answers still need a citation/faithfulness review. Increasing Qwen's output cap resolved reported cap hits in the full run, but this does not establish semantic correctness.

## A supported CV statement

Built and evaluated a Vietnamese legal RAG pipeline over 61,068 articles, achieving 82.36% Recall@5 on 3,196 labeled training queries—12.17 percentage points above BM25—using hybrid retrieval and BGE reranking; implemented resumable dual-GPU evaluation and local Qwen inference.

Keep the dataset scope and metric name in the claim. Do not describe Recall@5 as answer accuracy, attribute the retrieval gain to Qwen, or claim a two-GPU speedup without a matched timing experiment. README publication remains pending user review.
