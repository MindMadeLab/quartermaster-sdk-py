"""
Vector/RAG tools for embedding, storing, and searching text.

Provides tools for:
- embed_text: Generate text embeddings (built-in hash-based or sentence-transformers)
- vector_store: Store text with embeddings in memory or JSON files
- vector_search: Cosine-similarity search over stored embeddings
- document_index: Chunk and index documents for retrieval
- hybrid_search: Combined semantic + keyword search
"""

from quartermaster_tools.builtin.vector.embed import EmbedTextTool, embed_text
from quartermaster_tools.builtin.vector.index import DocumentIndexTool, document_index
from quartermaster_tools.builtin.vector.search import (
    HybridSearchTool,
    VectorSearchTool,
    hybrid_search,
    vector_search,
)
from quartermaster_tools.builtin.vector.store import VectorStoreTool, vector_store

__all__ = [
    "document_index",
    "DocumentIndexTool",
    "embed_text",
    "EmbedTextTool",
    "hybrid_search",
    "HybridSearchTool",
    "vector_search",
    "VectorSearchTool",
    "vector_store",
    "VectorStoreTool",
]
