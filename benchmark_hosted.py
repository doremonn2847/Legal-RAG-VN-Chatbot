"""Checkpointed, retrieval-only top-five evaluation; never rewrites README results."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import pickle
import time

from utils.ablation import rank_candidates, summarize_records


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def dataset_rows(path, limit=0):
    rows = json.loads(Path(path).read_text(encoding="utf-8-sig"))["items"]
    if limit:
        rows = rows[:limit]
    if len({r["question_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate question IDs")
    for row in rows:
        row["expected_ids"] = sorted({f"{a['law_id']}_{a['article_id']}" for a in row["relevant_articles"]})
        if not row["question"].strip() or not row["expected_ids"]:
            raise ValueError("Question or gold article labels are missing")
    return rows


def measured(function, *args, **kwargs):
    capture = io.StringIO()
    started = time.perf_counter()
    with contextlib.redirect_stdout(capture):
        result = function(*args, **kwargs)
    if "error" in capture.getvalue().lower():
        raise RuntimeError("Retrieval failed; refusing to score it as a miss")
    if any(not d.get("id") for d in result):
        raise ValueError("Retrieved article ID is missing")
    return result, time.perf_counter() - started


def report(output, manifest):
    completed = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((output / "questions").glob("*.json"))]
    records = [r for row in completed for r in row["records"]]
    metrics = [summarize_records([r for r in records if r["method"] == m], m)
               for m in ("bm25", "dense", "hybrid", "hybrid_bge")]
    metrics = [m for m in metrics if m]
    data = {"manifest": manifest, "completed_questions": len(completed), "metrics": metrics}
    (output / "summary.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Hosted top-five retrieval comparison — pending review", "",
             f"Completed: {len(completed)}/{len(manifest['question_ids'])} training questions.", "",
             "README Results remain unchanged. No generation, legal classification, refinement, or web search.", "",
             "| Method | n | P@5 | R@5 | F1@5 | Hit@5 | MAP@5 | MRR@5 | nDCG@5 | Mean s | P95 s |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in metrics:
        values = [m["method"], str(m["questions"])] + [f"{m[k]:.4f}" for k in
                  ("precision", "recall", "f1", "hit_rate", "map", "mrr", "ndcg", "mean_seconds", "p95_seconds")]
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "Protocol: original questions, exact gold IDs, 25 candidates per enabled retriever, final five.",
              "Hybrid uses the existing maximum normalized-score merge. BGE fusion weight is 0.8; full articles are submitted.",
              "Provider token truncation still applies. Each question shares candidate searches across methods; timings include hosted network calls.",
              "Historical README baselines used local embeddings/reranking on Kaggle. Changes in model, input length, host, and provider are confounded; this is not a single-factor ablation.",
              "Training-set retrieval scores do not establish held-out accuracy, answer faithfulness, or current-law validity."]
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--bm25", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="First n questions for a smoke check; zero means all")
    parser.add_argument("--run", action="store_true", help="Make hosted embedding/reranking calls")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("limit must be nonnegative")
    rows = dataset_rows(args.dataset, args.limit)
    print(f"Scope: {len(rows):,} labeled training questions, four retrieval methods, final top-five.", flush=True)
    if not args.run:
        print("Dry run. Add --run to evaluate with hosted embeddings and reranking.")
        return
    from config import Config
    from main.bm25_retriever import BM25Retriever
    from main.reranker import DocumentReranker
    from main.vector_store import QdrantVectorStore
    if Config.EMBEDDING_PROVIDER != "hosted" or Config.RERANK_PROVIDER != "hosted":
        raise ValueError("Select hosted embeddings and reranking before this evaluation")
    manifest = {"protocol": "hosted-full-article-original-query-top5-v1", "question_ids": [r["question_id"] for r in rows],
                "dataset_sha256": digest(args.dataset), "bm25_sha256": digest(args.bm25),
                "embedding_model": Config.EMBEDDING_MODEL, "reranker_model": Config.RERANKER_MODEL,
                "collection": Config.COLLECTION_NAME, "candidate_k": 25, "k": 5, "alpha": 0.8,
                "refinement": False, "article_character_cutoff": None, "qdrant_url": Config.QDRANT_URL}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "questions").mkdir(exist_ok=True)
    path = args.output / "manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Existing checkpoint has a different evaluation configuration")
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    sparse = BM25Retriever()
    with args.bm25.open("rb") as handle:
        index = pickle.load(handle)
    for key in ("bm25", "documents", "tokenized_corpus"):
        setattr(sparse, key, index[key])
    del index
    article_ids = {d["id"] for d in sparse.documents}
    if not all(set(r["expected_ids"]) <= article_ids for r in rows):
        raise ValueError("Gold labels refer to articles missing from the index")
    dense = QdrantVectorStore()
    dense.raise_errors = True
    reranker = DocumentReranker()
    reranker.raise_errors = True
    try:
        info = dense.client.get_collection(Config.COLLECTION_NAME)
        if info.points_count != len(sparse.documents) or str(info.status) != "green":
            raise ValueError("Cloud collection must be healthy and match the full BM25 corpus")
        report(args.output, manifest)
        for number, row in enumerate(rows, 1):
            checkpoint = args.output / "questions" / f"{row['question_id']}.json"
            if checkpoint.exists():
                saved = json.loads(checkpoint.read_text(encoding="utf-8"))
                if saved["question_id"] != row["question_id"] or saved["expected_ids"] != row["expected_ids"] or len(saved["records"]) != 4:
                    raise ValueError("Invalid question checkpoint")
                continue
            query = row["question"]
            bm25, sparse_s = measured(sparse.get_relevant_documents, query, top_k=25)
            vectors, dense_s = measured(dense.search_similar_documents, query, top_k=25)
            candidates = rank_candidates(bm25, vectors, "hybrid", k=len(bm25) + len(vectors))
            bge, rerank_s = measured(reranker.rerank_with_fusion, query, candidates, alpha=0.8, top_k=5)
            records = []
            for method, docs, seconds in [("bm25", rank_candidates(bm25, [], "bm25"), sparse_s),
                                           ("dense", rank_candidates([], vectors, "dense"), dense_s),
                                           ("hybrid", candidates[:5], sparse_s + dense_s),
                                           ("hybrid_bge", bge, sparse_s + dense_s + rerank_s)]:
                records.append({"question_id": row["question_id"], "method": method, "k": 5,
                                "expected_ids": row["expected_ids"], "retrieved_ids": [d["id"] for d in docs],
                                "retrieval_seconds": seconds})
            temp = checkpoint.with_suffix(".tmp")
            temp.write_text(json.dumps({"question_id": row["question_id"], "question": query,
                "expected_ids": row["expected_ids"], "records": records}, ensure_ascii=False), encoding="utf-8")
            temp.replace(checkpoint)
            print(f"Evaluated {number}/{len(rows)}", flush=True)
            if number % 25 == 0:
                report(args.output, manifest)
    finally:
        dense.client.close()
        report(args.output, manifest)


if __name__ == "__main__":
    main()
