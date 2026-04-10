"""
Document indexing tool.

Reads a text file, splits it into overlapping chunks, embeds each chunk,
and stores them in a vector collection for later retrieval.
"""

from __future__ import annotations

import os
from typing import Any

from quartermaster_tools.base import AbstractTool
from quartermaster_tools.builtin.vector.store import VectorStoreTool
from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolResult


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


class DocumentIndexTool(AbstractTool):
    """Index a text document by chunking and storing embeddings."""

    def name(self) -> str:
        return "document_index"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="file_path",
                description="Path to the text file to index.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="collection",
                description="Name of the collection to store chunks in.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="chunk_size",
                description="Maximum number of characters per chunk.",
                type="number",
                required=False,
                default=500,
            ),
            ToolParameter(
                name="chunk_overlap",
                description="Number of overlapping characters between consecutive chunks.",
                type="number",
                required=False,
                default=50,
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
            short_description="Index a document for vector search.",
            long_description=(
                "Reads a text file, splits it into overlapping chunks, generates "
                "embeddings for each chunk, and stores them in a vector collection. "
                "The indexed chunks can then be searched with VectorSearchTool or "
                "HybridSearchTool."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        file_path: str = kwargs.get("file_path", "")
        collection: str = kwargs.get("collection", "")
        chunk_size: int = int(kwargs.get("chunk_size", 500))
        chunk_overlap: int = int(kwargs.get("chunk_overlap", 50))
        store_path: str | None = kwargs.get("store_path", None)

        if not file_path:
            return ToolResult(success=False, error="Parameter 'file_path' is required")
        if not collection:
            return ToolResult(success=False, error="Parameter 'collection' is required")
        if not os.path.exists(file_path):
            return ToolResult(success=False, error=f"File not found: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to read file: {e}")

        chunks = _chunk_text(content, chunk_size, chunk_overlap)
        if not chunks:
            return ToolResult(
                success=True,
                data={"chunks_indexed": 0, "collection": collection},
                metadata={"message": "File is empty, no chunks to index"},
            )

        store_tool = VectorStoreTool()
        for i, chunk in enumerate(chunks):
            metadata = {
                "source": file_path,
                "chunk_index": i,
                "chunk_total": len(chunks),
            }
            result = store_tool.run(
                collection=collection,
                text=chunk,
                metadata=metadata,
                store_path=store_path,
            )
            if not result.success:
                return ToolResult(
                    success=False,
                    error=f"Failed to store chunk {i}: {result.error}",
                )

        return ToolResult(
            success=True,
            data={"chunks_indexed": len(chunks), "collection": collection},
        )
