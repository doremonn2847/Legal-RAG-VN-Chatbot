# Hosted demo deployment on this laptop

The live API/UI now build from the hosted-retrieval checkout. They use DeepSeek Flash, hosted BGE-M3 embeddings, hosted BGE reranking, and the completed `hosted_baai_bge_m3_1024` collection (61,068 articles). The final context remains five articles, with query refinement disabled. Langfuse remains enabled through the private environment file.

The original main checkout, BM25 file, Qdrant server/storage, and Cloudflare tunnel are retained. Demo credentials are unchanged. The public quick-tunnel URL still depends on this laptop and the existing tunnel process; it is not permanent hosting.

## Restart the hosted demo

Run from `C:/Users/Lenovo/vietnamese-legal-chatbot` in PowerShell:

```powershell
$dockerExe = 'C:/Users/Lenovo/AppData/Local/Programs/DockerDesktop/resources/bin/docker.exe'
$hosted = 'C:/Users/Lenovo/.codex/worktrees/hosted-retrieval/vietnamese-legal-chatbot'
& $dockerExe compose --project-directory $PWD --env-file "$hosted/.env" -p vietnamese-legal-chatbot -f "$hosted/compose.yaml" -f results/hosted-demo.override.yaml up -d --no-deps api ui
```

The override points image builds to the hosted checkout while the project directory keeps data binds in the original checkout. Do not run the hosted Compose file alone against its worktree: that would select a different Qdrant storage folder. Add `--build` only when source changes require rebuilt images.

## Roll back to the previous local demo

Old images are tagged before migration. Run from the original checkout:

```powershell
$dockerExe = 'C:/Users/Lenovo/AppData/Local/Programs/DockerDesktop/resources/bin/docker.exe'
& $dockerExe compose stop -t 30 ui api
& $dockerExe tag legal-demo-rollback-api:20261012 vietnamese-legal-chatbot-api:latest
& $dockerExe tag legal-demo-rollback-ui:20261012 vietnamese-legal-chatbot-ui:latest
& $dockerExe compose up -d --no-build --no-deps api ui
```

This changes only API/UI containers. Both Qdrant collections remain intact. The hosted source is still on its development branch; deployment does not merge it into main or alter Git history.

## Limits

The earlier hosted smoke check verified all stages and Langfuse export, but one of two selected legal questions missed its gold reference. A representative evaluation is still needed before claiming quality gains. Hosted services require network availability and provider quota; public usage can incur API charges.

## Verification on 2026-10-12

The hosted API is healthy; authenticated Gradio chat completed through the API. Both local and public `/config` reject unauthenticated requests. A legal request returned five sources and included gold article `58/2010/qh12_13`. Langfuse exported seven observations, including two `deepseek-flash` generations with token usage and no error observations. The image now explicitly installs `langchain`, required by Langfuse's callback.

All 15 Gradio unit checks passed. The broader local test discovery stalled during loading and was stopped to free memory; it was not counted as a pass. Verification records are saved under the original checkout's `results/hosted-demo-*-verification.json` (plus `hosted-demo-verification.json`). No full quality benchmark was run for this deployment.
