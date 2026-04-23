"""
Vector/RAG tools for embedding, storing, and searching text.

Provides tools for:
- embed_text: Generate text embeddings (built-in hash-based or sentence-transformers)
- vector_store: Store text with embeddings in memory or JSON files
- vector_search: Cosine-similarity search over stored embeddings
- document_index: Chunk and index documents for retrieval
- hybrid_search: Combined semantic + keyword search
"""

from quartermaster_tools.builtin.vector.embed import embed_text
from quartermaster_tools.builtin.vector.index import document_index
from quartermaster_tools.builtin.vector.search import (
    hybrid_search,
    vector_search,
)
from quartermaster_tools.builtin.vector.store import vector_store

__all__ = [
    "document_index",
    "embed_text",
    "hybrid_search",
    "vector_search",
    "vector_store",
]
