# Hosted retrieval benchmark and previous-stack comparison

All 3,196 labeled training questions completed. The evaluation uses original queries, 25 BM25 and 25 dense candidates, exact gold article IDs, and final top-5. It makes no answer-generation, classification, refinement or web-search calls. Per-question checkpoints and manifests remain in `results/hosted-full-article-top5/` locally. The README Results tables now present this hosted benchmark following user approval; the previous stack is retained below for comparison.

## Current stack

| Method | P@5 | R@5 | F1@5 | Hit@5 | MAP@5 | MRR@5 | nDCG@5 | Mean s | P95 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.1430 | 0.7020 | 0.2370 | 0.7093 | 0.5371 | 0.5429 | 0.5800 | 0.4856 | 0.9370 |
| Dense BGE-M3 | 0.1544 | 0.7567 | 0.2559 | 0.7650 | 0.5885 | 0.5957 | 0.6324 | 0.5340 | 0.7470 |
| Hybrid | 0.1609 | 0.7887 | 0.2666 | 0.7963 | 0.5794 | 0.5851 | 0.6338 | 1.0196 | 1.4931 |
| Hybrid + BGE | 0.1854 | 0.9081 | 0.3071 | 0.9149 | 0.7843 | 0.7906 | 0.8173 | 1.6646 | 2.3744 |

## Previous stack

| Method | P@5 | R@5 | F1@5 | Hit@5 | MAP@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.1430 | 0.7020 | 0.2370 | 0.7093 | 0.5371 | 0.5429 | 0.5800 |
| Dense Vietnamese bi-encoder | 0.1210 | 0.5922 | 0.2004 | 0.6001 | 0.4489 | 0.4544 | 0.4862 |
| Hybrid | 0.1524 | 0.7476 | 0.2526 | 0.7547 | 0.5576 | 0.5633 | 0.6069 |
| Hybrid + BGE | 0.1681 | 0.8236 | 0.2785 | 0.8314 | 0.6715 | 0.6781 | 0.7113 |

The final reranked stack improves Recall@5 by **8.45 percentage points**, Hit@5 by **8.35 percentage points**, and nDCG@5 by **0.1060**. BM25 reproduces the previous scores exactly, supporting a consistent dataset/scoring protocol.

This compares whole stacks: dense embeddings changed to hosted BGE-M3 (1,024 dimensions), full articles now reach hosted reranking, and Qdrant/provider/compute locations changed. Hybrid merging uses maximum normalized retrieval score and BGE score fusion weight 0.8. Provider token limits still apply. This does not isolate which change caused the gain; network timings cannot be directly compared with the previous Kaggle hardware.

These are training-set retrieval results, not held-out accuracy, generated-answer quality, current-law validity or upload-document evaluation. The 1.6646-second mean measures the retrieval/reranking stages, not complete chatbot response time.
