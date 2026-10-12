"""OpenAI-compatible embedding endpoint with the existing encoder interface."""
import math
import time
from urllib.parse import urlsplit

import numpy as np
import requests


class HostedEmbeddings:
    def __init__(self, base_url, model, dimensions, api_key):
        url = urlsplit(base_url)
        if url.scheme != "https" or not url.hostname or url.username or url.query or url.fragment:
            raise ValueError("EMBEDDING_BASE_URL must be an HTTPS URL without credentials or query parameters")
        if not api_key:
            raise ValueError("EMBEDDING_API_KEY is required")
        if dimensions < 1:
            raise ValueError("EMBEDDING_DIMENSIONS must be positive")
        self.url = base_url.rstrip("/") + "/embeddings"
        self.model, self.dimensions = model, dimensions
        self.headers = {"Authorization": "Bearer " + api_key}

    def get_sentence_embedding_dimension(self):
        return self.dimensions

    def _embed_batch(self, texts):
        body = {"model": self.model, "input": texts, "dimensions": self.dimensions, "encoding_format": "float"}
        for attempt in range(5):
            try:
                response = requests.post(self.url, headers=self.headers, json=body, timeout=(10, 120))
            except requests.RequestException:
                raise RuntimeError("Hosted embedding connection failed; rerun to resume indexing") from None
            if response.status_code not in (429, 502, 503, 504) or attempt == 4:
                break
            try:
                delay = float(response.headers.get("Retry-After", 2 ** attempt))
            except ValueError:
                delay = 2 ** attempt
            time.sleep(min(120, max(1, delay)))
        if response.status_code != 200:
            raise RuntimeError(f"Hosted embeddings returned HTTP {response.status_code}")
        try:
            data = response.json()["data"]
            if len(data) != len(texts):
                raise ValueError("Response count differs from input count")
            by_index = {row["index"]: row["embedding"] for row in data}
            if set(by_index) != set(range(len(texts))):
                raise ValueError("Embedding indexes are missing or duplicated")
            vectors = [by_index[i] for i in range(len(texts))]
            if any(len(v) != self.dimensions or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in v) for v in vectors):
                raise ValueError("Invalid embedding dimensions or values")
            return vectors
        except (ValueError, KeyError, TypeError):
            raise RuntimeError("Hosted embeddings returned an invalid response") from None

    def encode(self, texts, batch_size=16, **kwargs):
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        if batch_size < 1 or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("Embedding inputs must be nonempty text and batch_size must be positive")
        vectors = []
        for start in range(0, len(texts), batch_size):
            vectors.extend(self._embed_batch(texts[start:start + batch_size]))
        array = np.asarray(vectors, dtype=float).reshape(len(texts), self.dimensions)
        return array[0] if single else array
