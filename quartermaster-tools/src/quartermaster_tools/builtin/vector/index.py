"""
Document indexing tool.

Reads a text file, splits it into overlapping chunks, embeds each chunk,
and stores them in a vector collection for later retrieval.
"""

from __future__ import annotations

import os

from quartermaster_tools.builtin.vector.store import vector_store
from quartermaster_tools.decorator import tool


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping chunks by character count."""
    if chunk_size <= 0:
        return [text] if text else []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
        if chunk_overlap >= chunk_size:
            # Prevent infinite loop
            start = end
    return chunks


@tool()
def document_index(
    file_path: str,
    collection: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    store_path: str = None,
) -> dict:
    """Index a document for vector search.

    Reads a text file, splits it into overlapping chunks, generates
    embeddings for each chunk, and stores them in a vector collection.
    The indexed chunks can then be searched with vector_search or
    hybrid_search.

    Args:
        file_path: Path to the text file to index.
        collection: Name of the collection to store chunks in.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.
        store_path: Path to a JSON file for persistent storage. In-memory if omitted.
    """
    if not file_path:
        raise ValueError("Parameter 'file_path' is required")
    if not collection:
        raise ValueError("Parameter 'collection' is required")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    chunk_size = int(chunk_size)
    chunk_overlap = int(chunk_overlap)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        raise RuntimeError(f"Failed to read file: {e}")

    chunks = _chunk_text(content, chunk_size, chunk_overlap)
    if not chunks:
        return {"chunks_indexed": 0, "collection": collection}

    for i, chunk in enumerate(chunks):
        metadata = {
            "source": file_path,
            "chunk_index": i,
            "chunk_total": len(chunks),
        }
        vector_store(
            collection=collection,
            text=chunk,
            metadata=metadata,
            store_path=store_path,
        )

    return {"chunks_indexed": len(chunks), "collection": collection}
