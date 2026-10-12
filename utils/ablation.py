"""Lightweight ranking and metrics for fixed-candidate retrieval ablations."""
import math
import statistics

METHODS = ('bm25', 'dense', 'hybrid')


def rank_candidates(sparse, dense, method, k=5):
    if method not in METHODS:
        raise ValueError('Unknown ablation method')
    documents = {}
    if method in ('bm25', 'hybrid'):
        for doc in sparse:
            if doc.get('id'):
                documents[doc['id']] = dict(doc)
    if method in ('dense', 'hybrid'):
        for doc in dense:
            if doc.get('id'):
                if doc['id'] in documents:
                    documents[doc['id']]['score'] = max(doc['score'], documents[doc['id']]['score'])
                else:
                    documents[doc['id']] = dict(doc)
    # Matches the current pipeline's pre-BGE ranking and stable tie order.
    return sorted(documents.values(), key=lambda d: d['score'], reverse=True)[:k]


def score_ranking(ids, expected, k=5):
    relevant = set(expected)
    seen = set()
    hits, ap, dcg, rr = 0, 0.0, 0.0, 0.0
    for rank, doc in enumerate(ids[:k], 1):
        if doc in relevant and doc not in seen:
            hits += 1
            ap += hits / rank
            dcg += 1 / math.log2(rank + 1)
            rr = rr or 1 / rank
            seen.add(doc)
    p = hits / k
    recall = hits / len(relevant) if relevant else 0.0
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(k, len(relevant)) + 1))
    return {'precision': p, 'recall': recall, 'f1': 2*p*recall/(p+recall) if p+recall else 0.0,
            'hit_rate': float(hits > 0), 'map': ap/len(relevant) if relevant else 0.0,
            'mrr': rr, 'ndcg': dcg/ideal if ideal else 0.0}


def summarize_records(records, method):
    scores = [score_ranking(r['retrieved_ids'], r['expected_ids'], r['k']) for r in records]
    if not scores:
        return None
    durations = sorted(r['retrieval_seconds'] for r in records)
    return {'method': method, 'k': 5, 'questions': len(records),
            **{name: statistics.mean(s[name] for s in scores) for name in scores[0]},
            'mean_seconds': statistics.mean(durations),
            'p95_seconds': durations[math.ceil(.95 * len(durations)) - 1]}
