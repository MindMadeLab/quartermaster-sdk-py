"""
Vector store tool for persisting text with embeddings.

Supports in-memory storage or JSON-file-backed persistence.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

from quartermaster_tools.builtin.vector.embed import _builtin_embed
from quartermaster_tools.decorator import tool

# Module-level in-memory store shared across instances
_memory_store: dict[str, list[dict[str, Any]]] = {}


def _load_store(store_path: str | None) -> dict[str, list[dict[str, Any]]]:
    """Load the vector store from disk or return the in-memory store."""
    if store_path is None:
        return _memory_store
    if os.path.exists(store_path):
        with open(store_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_store(store_path: str | None, store: dict[str, list[dict[str, Any]]]) -> None:
    """Persist the store to disk if a path is provided."""
    if store_path is not None:
        with open(store_path, "w", encoding="utf-8") as f:
            json.dump(store, f, ensure_ascii=False)


@tool()
def vector_store(
    collection: str,
    text: str,
    metadata: dict = None,
    embedding: list = None,
    store_path: str = None,
) -> dict:
    """Store text with vector embeddings.

    Stores a text document along with its embedding vector in a named
    collection. Supports in-memory storage or JSON-file persistence.
    If no embedding is provided, one is auto-generated using the
    built-in hash-based embedder.

    Args:
        collection: Name of the collection to store the document in.
        text: The text content to store.
        metadata: Optional metadata dict to attach to the document.
        embedding: Pre-computed embedding vector. Auto-generated if omitted.
        store_path: Path to a JSON file for persistent storage. In-memory if omitted.
    """
    if not collection:
        raise ValueError("Parameter 'collection' is required")
    if not text:
        raise ValueError("Parameter 'text' is required")

    if embedding is None:
        embedding = _builtin_embed(text)

    store = _load_store(store_path)
    if collection not in store:
        store[collection] = []

    doc_id = str(uuid.uuid4())
    doc = {
        "id": doc_id,
        "text": text,
        "embedding": embedding,
        "metadata": metadata or {},
    }
    store[collection].append(doc)
    _save_store(store_path, store)

    # Also update in-memory store if using file-backed mode
    # so that searches in the same session see the data
    if store_path is not None:
        _memory_store.update(store)

    return {"id": doc_id, "collection": collection, "stored": True}
