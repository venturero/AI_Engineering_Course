"""
Create dense vectors for chunks and queries.

Placeholder implementation: deterministic hashed bag-of-words vectors (NumPy only).
Swap this module for sentence-transformers or an API client when you need semantic quality.
"""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def embed_texts(texts: Iterable[str], *, dim: int = 256) -> np.ndarray:
    """
    Map each string to an L2-normalized `dim`-dimensional vector (hashing trick).

    Same input text always yields the same vector; different texts spread across dims.
    Good enough to wire the RAG pipeline; replace for real retrieval quality.
    """
    rows: list[np.ndarray] = []
    for raw in texts:
        vec = np.zeros(dim, dtype=np.float64)
        for tok in _tokenize(str(raw)):
            h = hash(tok) % dim
            vec[h] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        rows.append(vec)
    if not rows:
        return np.zeros((0, dim), dtype=np.float64)
    return np.vstack(rows)
