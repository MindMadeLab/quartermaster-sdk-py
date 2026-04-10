"""
Vector store tool for persisting text with embeddings.

Supports in-memory storage or JSON-file-backed persistence.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

from quartermaster_tools.base import AbstractTool
from quartermaster_tools.builtin.vector.embed import _builtin_embed
from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolResult

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


class VectorStoreTool(AbstractTool):
    """Store text with embeddings in a vector collection."""

    def name(self) -> str:
        return "vector_store"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="collection",
                description="Name of the collection to store the document in.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="text",
                description="The text content to store.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="metadata",
                description="Optional metadata dict to attach to the document.",
                type="object",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="embedding",
                description="Pre-computed embedding vector. Auto-generated if omitted.",
                type="array",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="store_path",
                description="Path to a JSON file for persistent storage. In-memory if omitted.",
                type="string",
                required=False,
                default=None,
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Store text with vector embeddings.",
            long_description=(
                "Stores a text document along with its embedding vector in a named "
                "collection. Supports in-memory storage or JSON-file persistence. "
                "If no embedding is provided, one is auto-generated using the "
                "built-in hash-based embedder."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        collection: str = kwargs.get("collection", "")
        text: str = kwargs.get("text", "")
        metadata: dict[str, Any] | None = kwargs.get("metadata", None)
        embedding: list[float] | None = kwargs.get("embedding", None)
        store_path: str | None = kwargs.get("store_path", None)

        if not collection:
            return ToolResult(success=False, error="Parameter 'collection' is required")
        if not text:
            return ToolResult(success=False, error="Parameter 'text' is required")

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

        return ToolResult(
            success=True,
            data={"id": doc_id, "collection": collection, "stored": True},
        )
