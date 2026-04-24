"""
Retrieval-augmented generation (RAG) building blocks.

Data flow (typical):
    ingest.load_document(path) -> raw text
    -> chunking.chunk_text(text) -> list[str]
    -> embeddings.embed_texts(chunks) -> ndarray (n, dim)
    -> vector_store.VectorStore.add(embeddings, chunks)
    -> retrieve.retrieve_context(query, store) -> str for the prompt
"""
