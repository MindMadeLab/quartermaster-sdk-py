"""Tests for the vector/RAG tools."""

from __future__ import annotations

import json
import os

import pytest

from quartermaster_tools.builtin.vector.embed import EmbedTextTool, _builtin_embed
from quartermaster_tools.builtin.vector.index import DocumentIndexTool, _chunk_text
from quartermaster_tools.builtin.vector.search import (
    HybridSearchTool,
    VectorSearchTool,
    _cosine_similarity,
    _keyword_score,
)
from quartermaster_tools.builtin.vector.store import VectorStoreTool, _memory_store


@pytest.fixture(autouse=True)
def _clear_memory_store():
    """Clear the in-memory vector store before each test."""
    _memory_store.clear()
    yield
    _memory_store.clear()


# ---------------------------------------------------------------------------
# EmbedTextTool tests
# ---------------------------------------------------------------------------


class TestEmbedTextTool:
    def test_deterministic_embeddings(self):
        """Same text always produces the same embedding."""
        tool = EmbedTextTool
        r1 = tool.run(text="hello world")
        r2 = tool.run(text="hello world")
        assert r1.success
        assert r2.success
        assert r1.data["embedding"] == r2.data["embedding"]

    def test_correct_dimensions(self):
        """Embedding has the requested number of dimensions."""
        tool = EmbedTextTool
        result = tool.run(text="test", dimensions=128)
        assert result.success
        assert len(result.data["embedding"]) == 128
        assert result.data["dimensions"] == 128

    def test_default_dimensions(self):
        """Default dimensions is 384."""
        tool = EmbedTextTool
        result = tool.run(text="test")
        assert result.success
        assert len(result.data["embedding"]) == 384

    def test_different_texts_different_vectors(self):
        """Different texts produce different embeddings."""
        tool = EmbedTextTool
        r1 = tool.run(text="the cat sat on the mat")
        r2 = tool.run(text="quantum physics is fascinating")
        assert r1.data["embedding"] != r2.data["embedding"]

    def test_model_field_returned(self):
        """Result includes the model name."""
        tool = EmbedTextTool
        result = tool.run(text="test")
        assert result.data["model"] == "builtin"

    def test_missing_text_error(self):
        """Missing text parameter returns an error."""
        tool = EmbedTextTool
        result = tool.run()
        assert not result.success
        assert "required" in result.error.lower()

    def test_nonexistent_model_error(self):
        """Non-builtin model without sentence-transformers returns error."""
        tool = EmbedTextTool
        result = tool.run(text="test", model="all-MiniLM-L6-v2")
        assert not result.success
        assert "sentence-transformers" in result.error.lower()

    def test_normalized_embedding(self):
        """Built-in embedding is L2-normalized (unit length)."""
        import math

        embedding = _builtin_embed("some text", 256)
        norm = math.sqrt(sum(v * v for v in embedding))
        assert abs(norm - 1.0) < 1e-6

    def test_tool_metadata(self):
        """Tool reports correct name and version."""
        tool = EmbedTextTool
        assert tool.name() == "embed_text"
        assert tool.version() == "1.0.0"
        info = tool.info()
        assert info.name == "embed_text"
        assert len(tool.parameters()) == 3


# ---------------------------------------------------------------------------
# VectorStoreTool tests
# ---------------------------------------------------------------------------


class TestVectorStoreTool:
    def test_store_in_memory(self):
        """Store a document in memory and verify the result."""
        tool = VectorStoreTool
        result = tool.run(collection="docs", text="hello world")
        assert result.success
        assert result.data["stored"] is True
        assert result.data["collection"] == "docs"
        assert "id" in result.data

    def test_store_with_metadata(self):
        """Store a document with metadata."""
        tool = VectorStoreTool
        result = tool.run(
            collection="docs",
            text="hello",
            metadata={"source": "test", "page": 1},
        )
        assert result.success
        # Verify metadata is actually stored
        doc = _memory_store["docs"][0]
        assert doc["metadata"]["source"] == "test"
        assert doc["metadata"]["page"] == 1

    def test_store_with_precomputed_embedding(self):
        """Store with a pre-computed embedding vector."""
        tool = VectorStoreTool
        emb = [0.1, 0.2, 0.3]
        result = tool.run(collection="docs", text="test", embedding=emb)
        assert result.success
        doc = _memory_store["docs"][0]
        assert doc["embedding"] == emb

    def test_auto_generates_embedding(self):
        """Embedding is auto-generated when not provided."""
        tool = VectorStoreTool
        tool.run(collection="docs", text="auto embed me")
        doc = _memory_store["docs"][0]
        assert len(doc["embedding"]) == 384

    def test_persistence_to_file(self, tmp_path):
        """Store persists to a JSON file."""
        store_file = str(tmp_path / "vectors.json")
        tool = VectorStoreTool
        tool.run(collection="test_col", text="persistent doc", store_path=store_file)

        assert os.path.exists(store_file)
        with open(store_file, "r") as f:
            data = json.load(f)
        assert "test_col" in data
        assert len(data["test_col"]) == 1
        assert data["test_col"][0]["text"] == "persistent doc"

    def test_missing_collection_error(self):
        tool = VectorStoreTool
        result = tool.run(text="hello")
        assert not result.success

    def test_missing_text_error(self):
        tool = VectorStoreTool
        result = tool.run(collection="docs")
        assert not result.success


# ---------------------------------------------------------------------------
# VectorSearchTool tests
# ---------------------------------------------------------------------------


class TestVectorSearchTool:
    def _populate(self, texts: list[str], collection: str = "docs"):
        store = VectorStoreTool
        for t in texts:
            store.run(collection=collection, text=t)

    def test_finds_relevant_docs(self):
        """Search returns documents, most similar first."""
        self._populate(["the cat sat on the mat", "dogs are loyal animals", "cat food is expensive"])
        tool = VectorSearchTool
        result = tool.run(collection="docs", query="cat")
        assert result.success
        assert result.data["count"] > 0

    def test_top_k_limits_results(self):
        """top_k parameter limits the number of results."""
        self._populate([f"document number {i}" for i in range(10)])
        tool = VectorSearchTool
        result = tool.run(collection="docs", query="document", top_k=3)
        assert result.success
        assert result.data["count"] <= 3

    def test_threshold_filters_results(self):
        """High threshold filters out low-similarity results."""
        self._populate(["hello world", "completely unrelated xyz"])
        tool = VectorSearchTool
        result = tool.run(collection="docs", query="hello world", threshold=0.99)
        assert result.success
        # The exact match should score very high
        for r in result.data["results"]:
            assert r["score"] >= 0.99

    def test_empty_collection(self):
        """Search on empty collection returns empty results."""
        tool = VectorSearchTool
        result = tool.run(collection="nonexistent", query="test")
        assert result.success
        assert result.data["count"] == 0

    def test_results_have_score_and_metadata(self):
        """Each result has text, score, and metadata fields."""
        store = VectorStoreTool
        store.run(collection="docs", text="test doc", metadata={"key": "val"})
        tool = VectorSearchTool
        result = tool.run(collection="docs", query="test")
        assert result.success
        r = result.data["results"][0]
        assert "text" in r
        assert "score" in r
        assert "metadata" in r
        assert r["metadata"]["key"] == "val"

    def test_file_backed_search(self, tmp_path):
        """Search works with file-backed store."""
        store_file = str(tmp_path / "store.json")
        store = VectorStoreTool
        store.run(collection="docs", text="hello world", store_path=store_file)
        store.run(collection="docs", text="goodbye world", store_path=store_file)

        tool = VectorSearchTool
        result = tool.run(collection="docs", query="hello", store_path=store_file)
        assert result.success
        assert result.data["count"] > 0


# ---------------------------------------------------------------------------
# DocumentIndexTool tests
# ---------------------------------------------------------------------------


class TestDocumentIndexTool:
    def test_chunks_file_correctly(self, tmp_path):
        """Indexes a file into the correct number of chunks."""
        doc = tmp_path / "doc.txt"
        doc.write_text("a" * 1000)
        tool = DocumentIndexTool
        result = tool.run(file_path=str(doc), collection="idx", chunk_size=500, chunk_overlap=0)
        assert result.success
        assert result.data["chunks_indexed"] == 2

    def test_chunks_with_overlap(self, tmp_path):
        """Overlap produces more chunks than without."""
        doc = tmp_path / "doc.txt"
        doc.write_text("a" * 1000)
        tool = DocumentIndexTool
        result = tool.run(file_path=str(doc), collection="idx", chunk_size=500, chunk_overlap=100)
        assert result.success
        assert result.data["chunks_indexed"] == 3  # 0-500, 400-900, 800-1000

    def test_stores_all_chunks(self, tmp_path):
        """All chunks are stored in the collection."""
        doc = tmp_path / "doc.txt"
        doc.write_text("word " * 200)  # 1000 chars
        tool = DocumentIndexTool
        tool.run(file_path=str(doc), collection="idx", chunk_size=250, chunk_overlap=0)
        assert len(_memory_store.get("idx", [])) == 4

    def test_chunk_metadata_has_source(self, tmp_path):
        """Each chunk's metadata includes the source file path."""
        doc = tmp_path / "doc.txt"
        doc.write_text("short text")
        tool = DocumentIndexTool
        tool.run(file_path=str(doc), collection="idx")
        doc_entry = _memory_store["idx"][0]
        assert doc_entry["metadata"]["source"] == str(doc)
        assert doc_entry["metadata"]["chunk_index"] == 0

    def test_nonexistent_file_error(self):
        """Error when file does not exist."""
        tool = DocumentIndexTool
        result = tool.run(file_path="/nonexistent/file.txt", collection="idx")
        assert not result.success
        assert "not found" in result.error.lower()

    def test_empty_file(self, tmp_path):
        """Empty file produces zero chunks."""
        doc = tmp_path / "empty.txt"
        doc.write_text("")
        tool = DocumentIndexTool
        result = tool.run(file_path=str(doc), collection="idx")
        assert result.success
        assert result.data["chunks_indexed"] == 0

    def test_searchable_after_indexing(self, tmp_path):
        """Indexed documents can be found via vector search."""
        doc = tmp_path / "doc.txt"
        doc.write_text("Python is a programming language. It is used for web development.")
        tool = DocumentIndexTool
        tool.run(file_path=str(doc), collection="idx", chunk_size=40, chunk_overlap=10)

        search = VectorSearchTool
        result = search.run(collection="idx", query="Python programming")
        assert result.success
        assert result.data["count"] > 0


# ---------------------------------------------------------------------------
# HybridSearchTool tests
# ---------------------------------------------------------------------------


class TestHybridSearchTool:
    def _populate(self, texts: list[str], collection: str = "hybrid"):
        store = VectorStoreTool
        for t in texts:
            store.run(collection=collection, text=t)

    def test_combines_scores(self):
        """Results contain both vector_score and keyword_score."""
        self._populate(["python programming", "java development", "python snake"])
        tool = HybridSearchTool
        result = tool.run(collection="hybrid", query="python")
        assert result.success
        for r in result.data["results"]:
            assert "vector_score" in r
            assert "keyword_score" in r
            assert "score" in r

    def test_keyword_weight_affects_results(self):
        """Changing keyword_weight changes the final scores."""
        self._populate(["python programming", "java development"])
        tool = HybridSearchTool
        r1 = tool.run(collection="hybrid", query="python", keyword_weight=0.0)
        r2 = tool.run(collection="hybrid", query="python", keyword_weight=1.0)
        assert r1.success and r2.success
        # With keyword_weight=0, score equals vector_score
        for r in r1.data["results"]:
            assert abs(r["score"] - r["vector_score"]) < 1e-5
        # With keyword_weight=1, score equals keyword_score
        for r in r2.data["results"]:
            assert abs(r["score"] - r["keyword_score"]) < 1e-5

    def test_empty_collection(self):
        """Hybrid search on empty collection returns empty."""
        tool = HybridSearchTool
        result = tool.run(collection="empty", query="test")
        assert result.success
        assert result.data["count"] == 0

    def test_top_k_respected(self):
        """top_k limits the number of results."""
        self._populate([f"doc {i}" for i in range(10)])
        tool = HybridSearchTool
        result = tool.run(collection="hybrid", query="doc", top_k=2)
        assert result.data["count"] <= 2

    def test_tool_metadata(self):
        """Tool reports correct name and version."""
        tool = HybridSearchTool
        assert tool.name() == "hybrid_search"
        assert tool.version() == "1.0.0"


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_cosine_similarity_identical(self):
        """Identical vectors have cosine similarity of 1.0."""
        v = [1.0, 2.0, 3.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-9

    def test_cosine_similarity_orthogonal(self):
        """Orthogonal vectors have cosine similarity of 0.0."""
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(_cosine_similarity(a, b)) < 1e-9

    def test_cosine_similarity_opposite(self):
        """Opposite vectors have cosine similarity of -1.0."""
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(_cosine_similarity(a, b) - (-1.0)) < 1e-9

    def test_keyword_score_basic(self):
        """Keyword score counts query terms in text."""
        score = _keyword_score("python java", "python is great and python rocks")
        assert score > 0

    def test_keyword_score_no_match(self):
        """No matching terms gives score of 0."""
        score = _keyword_score("xyz", "abc def ghi")
        assert score == 0.0

    def test_chunk_text(self):
        """Chunks text correctly with overlap."""
        chunks = _chunk_text("abcdefghij", 5, 2)
        assert chunks[0] == "abcde"
        assert chunks[1] == "defgh"
        assert chunks[2] == "ghij"

    def test_chunk_text_no_overlap(self):
        """Chunks text correctly without overlap."""
        chunks = _chunk_text("abcdefghij", 5, 0)
        assert len(chunks) == 2
        assert chunks[0] == "abcde"
        assert chunks[1] == "fghij"
