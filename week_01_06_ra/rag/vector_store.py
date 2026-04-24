"""
In-memory vector store (FAISS-free placeholder).

Stores chunk texts aligned with embedding rows; search uses cosine similarity via NumPy.
"""

from __future__ import annotations

import numpy as np


class VectorStore:
    """Holds embeddings and the original chunk strings for similarity search."""

    def __init__(self) -> None:
        self._vectors: np.ndarray | None = None  # (n, dim)
        self._texts: list[str] = []

    @property
    def size(self) -> int:
        return len(self._texts)

    def add(self, embeddings: np.ndarray, texts: list[str]) -> None:
        """Append rows to the index; `embeddings` shape (n, dim), `texts` length n."""
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2-D")
        if embeddings.shape[0] != len(texts):
            raise ValueError("embeddings and texts length mismatch")
        if self._vectors is None:
            self._vectors = embeddings.astype(np.float64, copy=True)
        else:
            if embeddings.shape[1] != self._vectors.shape[1]:
                raise ValueError("embedding dimension mismatch")
            self._vectors = np.vstack([self._vectors, embeddings])
        self._texts.extend(texts)

    def search(self, query_vector: np.ndarray, k: int = 4) -> list[tuple[str, float]]:
        """
        Return up to `k` (chunk_text, score) pairs, sorted by descending cosine similarity.
        """
        if self._vectors is None or len(self._texts) == 0:
            return []
        q = query_vector.astype(np.float64, copy=False).ravel()
        if q.shape[0] != self._vectors.shape[1]:
            raise ValueError("query_vector dimension mismatch")

        mat = self._vectors
        qn = q / (np.linalg.norm(q) + 1e-12)
        mn = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
        scores = mn @ qn
        k = max(1, min(k, len(scores)))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(self._texts[i], float(scores[i])) for i in idx]
