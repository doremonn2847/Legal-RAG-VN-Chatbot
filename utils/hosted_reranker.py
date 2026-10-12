"""TEI rerank endpoint adapter returning scores in original document order."""
import math
import time
from urllib.parse import urlsplit

import requests


class HostedReranker:
    def __init__(self, url, api_key):
        parsed = urlsplit(url)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("RERANK_URL must be an HTTPS URL without credentials or query")
        if not api_key:
            raise ValueError("RERANK_API_KEY or EMBEDDING_API_KEY is required")
        self.url = url
        self.headers = {"Authorization": "Bearer " + api_key}

    def predict(self, pairs):
        if not pairs:
            return []
        query = pairs[0][0]
        if not isinstance(query, str) or not query.strip() or any(
            pair[0] != query or not isinstance(pair[1], str) or not pair[1].strip()
            for pair in pairs
        ):
            raise ValueError("Rerank requires one nonempty query and nonempty document texts")
        body = {"query": query, "texts": [pair[1] for pair in pairs],
                "truncate": True, "raw_scores": True, "return_text": False}
        for attempt in range(3):
            try:
                response = requests.post(self.url, headers=self.headers, json=body, timeout=(10, 120))
            except requests.RequestException:
                raise RuntimeError("Hosted rerank connection failed") from None
            if response.status_code not in (429, 502, 503, 504) or attempt == 2:
                break
            time.sleep(2 ** attempt)
        if response.status_code != 200:
            raise RuntimeError(f"Hosted rerank returned HTTP {response.status_code}")
        try:
            rows = response.json()
            if not isinstance(rows, list) or len(rows) != len(pairs):
                raise ValueError("Incorrect score count")
            scores = {}
            for row in rows:
                index, score = row["index"], row["score"]
                if (type(index) is not int or index in scores or type(score) not in (int, float)
                        or not math.isfinite(score)):
                    raise ValueError("Invalid index or score")
                scores[index] = float(score)
            if set(scores) != set(range(len(pairs))):
                raise ValueError("Missing or out-of-range indexes")
            return [scores[i] for i in range(len(pairs))]
        except (ValueError, KeyError, TypeError):
            raise RuntimeError("Hosted rerank returned an invalid response") from None
