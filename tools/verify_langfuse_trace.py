"""Fetch a live trace through the official CLI and audit the RAG observations."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_id")
    parser.add_argument("--output", default="results/langfuse-live-audit.json")
    args = parser.parse_args()
    load_dotenv()
    environment = os.environ.copy()
    environment["LANGFUSE_HOST"] = environment["LANGFUSE_BASE_URL"]
    npx = shutil.which("npx.cmd") or shutil.which("npx")
    if not npx:
        raise RuntimeError("Node.js/npx is required for the official Langfuse CLI")
    response = subprocess.run([
        npx, "--yes", "langfuse-cli", "api", "observations", "list",
        "--trace-id", args.trace_id, "--limit", "100", "--json",
        "--fields", "core,basic,io,model,usage,metadata,metrics,trace_context",
    ], env=environment, capture_output=True, text=True, encoding="utf-8", timeout=120, check=True)
    envelope = json.loads(response.stdout)
    if "body" in envelope:
        assert envelope["status"] == 200, "Langfuse observations request failed"
    payload = envelope.get("body", envelope)
    rows = payload["data"]
    names = {row["name"] for row in rows}
    required = {"answer-legal-question", "retrieve-context", "rerank-context", "generate-response"}
    assert required <= names, f"Missing stages: {required - names}"
    root = next(row for row in rows if row["name"] == "answer-legal-question")
    assert root.get("input") and root.get("output"), "Root input/output missing"
    assert root.get("isRootObservation"), "Request is not a logical root observation"
    assert root.get("sessionId"), "Conversation ID missing"
    assert all(row.get("level") != "ERROR" for row in rows), "Trace contains errors"
    by_id = {row["id"]: row for row in rows}
    generations = [row for row in rows if row["type"] == "GENERATION"]
    assert generations, "LLM generation observation missing"
    for row in generations:
        assert row.get("model"), "Generation model missing"
        usage = row.get("usageDetails") or {}
        assert usage.get("input", 0) > 0 and usage.get("output", 0) > 0, "Token usage missing"
        assert row.get("input") and row.get("output"), "Generation input/output missing"
        assert row.get("parentObservationId") in by_id, "Generation is not nested in this trace"
    for row in rows:
        if row["id"] != root["id"]:
            assert row.get("parentObservationId") in by_id, "Pipeline step is orphaned"
    assert next(row for row in rows if row["name"] == "retrieve-context")["type"] == "RETRIEVER"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"trace_id": args.trace_id, "observations": len(rows),
                      "generations": [{"model": row["model"], "usage": row["usageDetails"]}
                                      for row in generations], "audit": "passed"}, indent=2))


if __name__ == "__main__":
    main()
