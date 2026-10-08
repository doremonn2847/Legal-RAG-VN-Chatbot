# Live tracing verification — 2026-10-08

The official Langfuse skill was installed from `langfuse/skills` and its
instrumentation and best-practices guidance was applied. Langfuse Cloud project
authentication succeeded without displaying keys.

A legal question sent to the Docker API returned HTTP 200, five source excerpts,
and a trace ID. The current `.env` model was preserved: **qwen2.5:3b**. This
verification does not change the configured default or benchmark results.

The trace was fetched through the official Langfuse CLI's modern observations
endpoint. The audit confirmed six observations:

| Operation | Type | Parent |
| --- | --- | --- |
| answer-legal-question | Chain | Logical root |
| classify-legal-domain | Guardrail | Request |
| retrieve-context | Retriever | Request |
| rerank-context | Span | Retrieval |
| generate-response | Chain | Request |
| ChatOllama | Generation | Generation chain |

The question was accepted by keyword-based domain detection, so this trace has
one actual LLM call. Domain detection uses the same LangChain callback when its
LLM path is needed. The generation recorded **1,647 input tokens, 25 output
tokens**, and the model name. Root input/output, article IDs, feature settings,
session grouping and stage nesting were verified. No error observations were
present. The API request took **88.74 seconds** on this local CPU deployment;
this is one smoke test, not a latency benchmark or answer-quality evaluation.

[Open the audited trace](https://jp.cloud.langfuse.com/project/cmuz4zd1c00szad0ga7kdlpiu/traces/e26ed7f80c028f47583054d9febd6db5)
(requires your project access).

Privacy masking and telemetry failure handling passed automated tests. Pattern
masking is not complete anonymization; see [the deployment guide](docker-langfuse.md).
The full regression suite passed **94 tests**. Existing README Results remained
byte-for-byte unchanged.

Artifacts: `results/docker-langfuse-live-response.json`,
`results/langfuse-live-audit.json`, and `results/container-tracing-tests.log`.
