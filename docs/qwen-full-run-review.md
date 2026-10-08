# Full Qwen top-5 run: verified results

Reviewed October 8, 2026. All **3,196/3,196 training questions** completed. This is the hybrid + BGE baseline for the planned retrieval ablations. The historical README Results section remains unchanged pending user review.

## Protocol and integrity

The original ZIP is preserved at `results/qwen_gpu_workers_top5_results.zip`. Its 20 exported files are extracted under `results/qwen-gpu-workers-top5/`. Derived verification results are saved separately as `verification.json` in that folder.

- Both workers completed 1,598 questions. Their partitions are disjoint and cover all selected questions exactly once.
- Global order matches the seed-42 selection of all local training questions. Question text, gold article IDs, and the dataset hash match the local dataset.
- Merged question records exactly match worker records. Every answer matches its cached model call; there are 3,196 unique cached calls and 3,196 nonempty answers.
- Rankings contain no duplicate IDs, use final k=5, and match the evidence IDs supplied to answer generation.
- Precision, recall, F1, hit rate, MAP, MRR, and binary nDCG were independently recomputed from exact article IDs. All match the exported summary within 1e-10.
- All six benchmark source hashes in the manifest match the current local code.

The run used original questions without refinement, the existing Vietnamese bi-encoder vectors, 25 dense + 25 sparse candidates, BGE v2-m3 reranking with score fusion (alpha 0.8), and final top-5 evidence. Qwen 3.5 9B ran through Ollama 0.40.0 with Q4_K_M quantization, thinking disabled, context 8192, and output cap 2048. Two independent Tesla T4 workers processed separate questions.

## Verified retrieval table

These are macro averages over the full training set. Relevance uses exact gold article-ID matches. MAP uses the total number of unique gold articles as its denominator; nDCG uses binary relevance. Times describe individual retrieval calls in this concurrent Kaggle deployment.

| Method | Questions | k | Precision | Recall | F1 | Hit rate | MAP | MRR | nDCG | Mean retrieval s | P95 retrieval s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hybrid + BGE, original query | 3,196 | 5 | 0.1681 | 0.8236 | 0.2785 | 0.8314 | 0.6715 | 0.6781 | 0.7113 | 2.7192 | 3.4635 |

Recall@5 is **82.36%**; hit rate is **83.14%**. Precision at a fixed cutoff of five should be interpreted against the number of gold articles per question, rather than as generated-answer accuracy.

## Generation measurements

| Model | Answers | Mean answer s | P95 answer s | Mean input tokens | Mean output tokens | Maximum output tokens | Output-cap hits | Rate-limit retries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen 3.5 9B | 3,196 | 15.9055 | 31.9675 | 1,685.65 | 343.10 | 1,920 | 0 | 0 |

All 3,196 saved finish reasons are `stop`. The output-cap criterion checks both a length-related finish reason and output-token usage reaching 2048. None met that criterion. This supports successful completion under the higher token cap, but does not establish answer correctness, citation fidelity, or appropriate abstention.

The sum of recorded model-call durations is approximately 14.12 worker-hours. It is **not elapsed wall-clock runtime**: two workers ran concurrently, and the experiment was resumed across sessions. The export does not establish an end-to-end two-GPU speedup against a matched single-GPU run.

No new inference was performed during this review. A full human semantic review of the generated answers has not been completed. The raw `answer_review.csv` is preserved for that work.

## Ablation comparison to prepare next

| Configuration | Status | Purpose |
|---|---|---|
| BM25 only | Pending | Keyword-search baseline |
| Dense only | Pending | Embedding-search baseline |
| Hybrid without BGE | Pending | Measure fusion without the cross-encoder |
| Hybrid + BGE | Verified full run | Current pipeline |

The ablations should use the exact saved question IDs, dataset, corpus, embeddings, gold labels, and top-5 cutoff. They require retrieval only: no Qwen calls, no refinement, and no re-embedding. Candidate pools, score normalization/fusion, and similarity filtering must be documented so differences are attributable to the intended ablation. Reuse the verified current baseline only when relevant source and index hashes still match.

Report recall/nDCG/MRR changes against each baseline and the corresponding latency trade-off. Local BM25 quality can be compared directly on matched inputs, but local CPU latency must be labeled separately from Kaggle measurements. The old README table and 50-question pilot are historical context, not matched ablation baselines.

These are training-set results, not held-out generalization evidence. A separate labeled held-out evaluation and reviewed answer sample would support stronger CV claims. Do not attribute these retrieval scores to Qwen: the generator receives evidence after retrieval and does not determine these rankings.
