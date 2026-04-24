"""
Retrieve the most relevant chunks for a query and format them as context for the LM.
"""

from __future__ import annotations

from .embeddings import embed_texts
from .vector_store import VectorStore


def retrieve_context(
    query: str,
    store: VectorStore,
    *,
    k: int = 4,
    separator: str = "\n\n---\n\n",
) -> str:
    """
    Embed `query`, run top-k similarity search, concatenate chunk texts.

    The returned string is injected into the generation prompt as ### Context: ...
    """
    if store.size == 0:
        return ""
    qv = embed_texts([query])[0]
    hits = store.search(qv, k=k)
    parts = [text for text, _score in hits if text.strip()]
    return separator.join(parts)
