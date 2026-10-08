# Retrieval benchmark comparison — pending review

Status: stopped at user request. 2/50 questions scored; these are partial CPU results, not a final benchmark.

README.md and its historical Results table have not been changed.

## Proposed results table

| Method | Final k | Questions | P@k | R@k | F1@k | Hit@k / coverage | MAP@k | MRR@k | nDCG@k | Mean retrieval s | P95 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 5 | 2 | 0.1000 | 0.5000 | 0.1667 | 0.5000 | 0.2500 | 0.2500 | 0.3155 | 12.6236 | 20.9586 |
| bm25 | 10 | 2 | 0.1000 | 1.0000 | 0.1818 | 1.0000 | 0.3500 | 0.3500 | 0.5089 | 2.4316 | 3.0882 |
| vector | 5 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.3095 | 8.7817 |
| vector | 10 | 2 | 0.0500 | 0.5000 | 0.0909 | 0.5000 | 0.0833 | 0.0833 | 0.1781 | 2.6352 | 2.7395 |
| hybrid | 5 | 2 | 0.1000 | 0.5000 | 0.1667 | 0.5000 | 0.1667 | 0.1667 | 0.2500 | 4.0408 | 4.2238 |
| hybrid | 10 | 2 | 0.1000 | 1.0000 | 0.1818 | 1.0000 | 0.2292 | 0.2292 | 0.4077 | 4.5989 | 5.0367 |
| hybrid_rerank | 5 | 2 | 0.2000 | 1.0000 | 0.3333 | 1.0000 | 0.2667 | 0.2667 | 0.4434 | 160.8123 | 171.8250 |
| hybrid_rerank | 10 | 2 | 0.1000 | 1.0000 | 0.1818 | 1.0000 | 0.2667 | 0.2667 | 0.4434 | 160.8123 | 171.8250 |

## Protocol and reproducibility

- Dataset: labeled train_qna.csv; 50 of 3196 questions; selection seed 42. Public test labels are unavailable.
- Query refinement: True. The same saved query is used for every method and both cutoffs. Generation, domain detection and web fallback are excluded.
- Existing full-corpus BM25 and Qdrant indexes; no re-embedding. Article IDs must match exactly. Repeated IDs receive no additional relevance credit.
- BGE reranks up to 25 sparse + 25 dense candidates after deduplication; fused score weight 0.8. Excerpts are limited to 1,000 characters. Final top 5 is the prefix of the same top 10 ranking.
- Non-reranked modes are separately run at 5 and 10 because their candidate pool depends on k. BGE runs once per question; shared inference time is reported for both cutoffs, excluding refinement and initialization.
- Benchmark-only Qdrant timeout: 60 seconds. An initial attempt hit the default timeout (the server took 9.86 seconds); no failed questions are counted as relevance misses.
- Metrics are macro averages. Precision denominator is k; recall denominator is all unique labeled articles. MAP denominator is all labeled articles (matching the upstream convention). MRR is truncated at the output cutoff.
- nDCG uses binary exact-ID relevance and discount 1/log2(rank+1), normalized by an ideal ranking of min(k, number of labeled articles). Coverage equals hit rate at the specified cutoff.
- This measures retrieval relevance, not answer accuracy, faithfulness or legal currency. A training sample is not a held-out test; do not treat it as directly comparable to the historical full-training scores.
- Hardware: Windows AMD64 AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD; device cpu; PyTorch threads: 4.
- Raw records and metadata: `results/benchmark-50-refined-v2` (local, ignored by Git).

Reproduce from the project root:

```powershell
.venv/Scripts/python.exe -X utf8 benchmark_retrieval.py --limit 50 --seed 42 --threads 4 --output results/benchmark-50-refined-v2
```

Reuse the same output directory to resume. Use a new directory for a fresh timed run. The manifest records dataset/code hashes, selected IDs, model names, versions and configuration.

## Historical README table

| Method | MRR | Coverage | R@1 | R@10 | R@20 | MAP@20 |
|---|---:|---:|---:|---:|---:|---:|
| Sparse TellOnly | 0.5545 | 0.7894 | 0.430 | 0.768 | 0.783 | 0.565 |
| Dense Only | 0.4691 | 0.6809 | 0.364 | 0.666 | 0.673 | 0.471 |
| Hybrid (Sparse + Dense) | 0.5801 | 0.8820 | 0.431 | 0.833 | 0.875 | 0.592 |
| Hybrid + Reranking | 0.6082 | 0.8899 | 0.482 | 0.827 | 0.884 | 0.624 |

Historical results used the former MS MARCO reranker and the full training set. nDCG was not reported; the historical rankings are unavailable, so it cannot be reconstructed. Old scoring accepted substring IDs; these new scores use exact IDs.
