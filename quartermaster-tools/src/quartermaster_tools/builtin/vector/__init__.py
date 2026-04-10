"""
Vector/RAG tools for embedding, storing, and searching text.

Provides tools for:
- EmbedTextTool: Generate text embeddings (built-in hash-based or sentence-transformers)
- VectorStoreTool: Store text with embeddings in memory or JSON files
- VectorSearchTool: Cosine-similarity search over stored embeddings
- DocumentIndexTool: Chunk and index documents for retrieval
- HybridSearchTool: Combined semantic + keyword search
"""

from quartermaster_tools.builtin.vector.embed import EmbedTextTool
from quartermaster_tools.builtin.vector.store import VectorStoreTool
from quartermaster_tools.builtin.vector.search import HybridSearchTool, VectorSearchTool
from quartermaster_tools.builtin.vector.index import DocumentIndexTool

__all__ = [
    "DocumentIndexTool",
    "EmbedTextTool",
    "HybridSearchTool",
    "VectorSearchTool",
    "VectorStoreTool",
]
