# Qwen-only top-5 Kaggle run

Completed on Kaggle and imported October 8, 2026: all 3,196 questions, 1,598 per worker. See the [verified full-run review](qwen-full-run-review.md). The instructions below describe the reproducible run; this protocol supersedes the earlier paired follow-up.

Import `notebooks/kaggle_qwen_trial.ipynb` into Kaggle, enable GPU and internet, attach the existing `colab_benchmark_inputs.zip` Dataset, and select Run All. No secret or API key is needed. The notebook installs Ollama and pulls `qwen3.5:9b` inside Kaggle.

On a dual-T4 runtime, two independent workers process disjoint questions concurrently. Each GPU holds its own Qwen, Vietnamese embedding model, and BGE reranker. The weights are downloaded once; two dedicated Ollama servers use ports 11435 and 11436. A single-GPU runtime automatically uses one worker. The preflight loads Qwen with context 8192, exercises embeddings and a four-pair BGE batch, checks GPU residency, and requires 256 MiB of free GPU memory. It stops instead of silently accepting substantial CPU offload. The completed export verifies both workers finished on the recorded T4 runtime; a matched single-GPU speed comparison has not been performed.

The default is all training questions (`--limit 0`), shuffled with seed 42. Set `--limit 50` for the same seeded pilot sample. Retrieval uses the original question, 25 dense + 25 sparse candidates, BGE reranking, and final top-5. Qwen alone generates answers. Refinement is disabled. Output cap: 2048 tokens; context: 8192; thinking: disabled.

Results checkpoint under `/kaggle/working/vietnamese-legal-chatbot/results/qwen-gpu-workers-top5`. Download `/kaggle/working/qwen_gpu_workers_top5_results.zip` after the export cell. The ZIP includes merged reports plus worker directories and logs. Reuse the same worker count, source, and runtime to resume. On failure or interruption, the launcher stops its child processes and merges completed records; the export cell also refreshes the partial merged report. Only this worker-protocol checkpoint can resume this run; earlier Groq/Qwen pilot files are preserved separately. Manifests prevent mixing incompatible settings.

Exports contain retrieval metrics including nDCG, answers and human review fields, token usage, latency, and finish reasons where available. Manual review fields survive report regeneration. Calls and completed questions are cached.

The application also uses Qwen through `langchain-ollama`, configured with `OLLAMA_BASE_URL` (default `http://localhost:11434`) and `OLLAMA_MODEL` (default `qwen3.5:9b`). Run Ollama and `ollama pull qwen3.5:9b` locally before using the application. Groq dependencies, code paths, configuration and key requirements are removed. Existing legal-domain filtering also uses the local model; the controlled notebook bypasses domain filtering and web fallback.

Historical experiment files and the README Results section are preserved. Choosing Qwen for cost and availability does not resolve the citation problems identified in the first pilot; review the new answers for faithfulness and completeness.
