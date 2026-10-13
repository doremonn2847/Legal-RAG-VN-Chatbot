# Hosted pipeline smoke check

**Historical check:** the settings and timings below describe the October 12
development-branch smoke test. The branch is now merged into main; the current
demo uses Qdrant Cloud, 2,000-character excerpts, private uploads and optional OCR.
See [current deployment](hosted-demo-deployment.md),
[completed benchmark](hosted-benchmark-comparison.md) and
[upload/OCR verification](document-upload-verification.md).

Tested October 12, 2026 on `codex/hosted-retrieval`, using a temporary local API
on port 8001. The main demo was temporarily stopped to free memory, then restored.

## Configuration and index

- Hosted BGE-M3 embeddings: 1,024 dimensions.
- Qdrant collection `hosted_baai_bge_m3_1024`: verified complete at 61,068 articles.
- Existing BM25 index: 61,068 articles; API startup verified matching counts.
- Hosted reranking at `https://api.dek.io.vn/rerank`; the owner confirms
  `BAAI/bge-reranker-v2-m3`.
- DeepSeek Flash through `langchain-deepseek` for generation and legal detection.
- Original query, 25 sparse + 25 dense candidates, deduplication, alpha-0.8 score
  fusion, final top-5, 1,000-character excerpts, no web fallback.

## Results

| Check | Result |
|---|---|
| Hosted API readiness | HTTP 200 |
| Traffic-police authority question | HTTP 200; five sources; 12.359 seconds |
| Unused annual leave payment for public employees | HTTP 200; five sources; 11.765 seconds |
| Dinner recommendation question | Rejected as non-legal; no sources; 0.766 seconds |
| Empty question | HTTP 422 |
| Original query preserved, no web fallback, source limit | Passed |
| Langfuse export | Seven observations per legal-question trace; no ERROR observations |
| Main demo restored | API readiness and Gradio page both HTTP 200 |

The leave-payment question retrieved the training dataset's gold article,
`58/2010/qh12_13`. The traffic-police question did not retrieve its gold article,
`47/2011/tt-bca_7`, in the final five. Its answer explicitly said the supplied
evidence was insufficient, rather than claiming the requested authority was
established. The retrieved excerpt for another potentially relevant article was
cut at the configured character limit.

The leave answer quoted the work-requirement condition from Article 13, but its
closing summary omitted that condition. This is an answer-quality issue to
review before presenting the output as authoritative legal guidance.

This is a functional smoke check on two selected training questions, not a full
benchmark, an unbiased quality estimate, or validation against current law.
The existing README result table was not changed.

Detailed local artifacts:

- `C:/Users/Lenovo/vietnamese-legal-chatbot/results/hosted-pipeline-smoke.json`
- `C:/Users/Lenovo/vietnamese-legal-chatbot/results/hosted-pipeline-trace-audit.json`

The temporary hosted API was stopped after this historical test. The current
main demo has since migrated to hosted models and Qdrant Cloud.
