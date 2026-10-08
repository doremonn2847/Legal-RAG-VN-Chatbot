# Qwen 3.5 9B Kaggle trial

The notebook now prepares the [Qwen-only top-5 follow-up](qwen-answer-only-run.md), with no refinement, a 2048-token output cap, and no API key requirement. The protocol below describes the completed first pilot, whose results are preserved.

Status: completed on Kaggle for all 50 paired questions. Results were imported into `results/qwen-trial-kaggle/` and checked against the labeled sample. See [the pilot review](qwen-trial-review.md) for verified metrics, answer findings, and limitations. At the time of this review, Qwen was not approved for promotion. The user subsequently selected Qwen globally for cost and availability; see the follow-up protocol. The original review and historical metrics are preserved.

The first pilot used `notebooks/kaggle_qwen_trial.ipynb`. Enable GPU and internet in Kaggle settings for the follow-up. Attach `results/colab_benchmark_inputs.zip` as a private Kaggle Dataset and add a Kaggle Secret named `GROQ_API_KEY`. No Google Drive mounting is required.

The pilot selects the same 50 labeled training questions with seed 42. It compares three query arms: original question, single-call Groq refinement, and single-call Qwen refinement. Both model arms use the same refinement instruction. This is a new experiment; it does not mix earlier multi-call Groq checkpoints into the pilot.

All arms use the existing 61,068 vectors and BM25 index, 25 dense + 25 sparse candidates, BGE reranking, and final cutoffs of 5/10. No embeddings are rebuilt. Metrics include precision, recall, F1, hit rate, MAP, MRR, binary nDCG, retrieval time, model response time, token usage when available, and rate-limit retries.

Both models also answer the original question using the same top-5 evidence from the original-query arm. Review these paired answers for faithfulness, precise citations, Vietnamese clarity, unsupported legal claims, and overall acceptability. A numeric-reference guard rejects newly introduced article, clause, and document numbers during refinement; it does not replace human review of changed facts or named legislation.

The trial runs Qwen through `langchain-ollama`, with thinking disabled, an 8,192-token context, and a 512-token output cap. The Groq reference uses low reasoning effort and a 1,024-token completion cap, which includes reasoning tokens. Its SDK retries are disabled; rate-limit responses receive explicit cooldowns with counted attempts. These are task-oriented settings, not identical decoding budgets.

On a dual-T4 runtime, Ollama uses the second GPU and embeddings/BGE use the first. On a single GPU they share memory; the runner verifies GPU use and uses a BGE batch size of four. Neither local CPU latency nor throughput parity is promised.

Every successful model call is cached; complete paired questions are checkpointed. Results live in `/kaggle/working/vietnamese-legal-chatbot/results/qwen-trial`. Kaggle working files are not guaranteed to survive a discarded session. Download the result ZIP or preserve it through saved notebook outputs. To resume, attach the result ZIP as a Kaggle Dataset and use the same source, model digest, package versions, and GPU type. The notebook restores that checkpoint separately from the original Colab benchmark.

Expected exported files:

- `comparison.md`: metrics and interpretation.
- `answer_review.csv`: paired answers and editable review fields.
- `manifest.json`: reproducibility settings and hashes.
- `questions.jsonl`: completed paired questions and retrieval rankings.
- `model_calls.jsonl`: cached responses, usage, timings, and retry counts.
- `summary.json`: machine-readable metrics.

Keep Qwen experimental until retrieval quality and reviewed answers are acceptable. After the pilot passes, run the full retrieval evaluation and consider changing the application provider/default. No promotion is automatic.
