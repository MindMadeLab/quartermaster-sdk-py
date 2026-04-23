"""
Vector search tools: semantic similarity and hybrid (semantic + keyword).

Implements cosine-similarity search over stored embeddings, and a hybrid
search that combines cosine similarity with TF-based keyword scoring.
"""

from __future__ import annotations

import math
import re

from quartermaster_tools.builtin.vector.embed import _builtin_embed
from quartermaster_tools.builtin.vector.store import _load_store
from quartermaster_tools.decorator import tool


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _keyword_score(query: str, text: str) -> float:
    """Compute a simple TF-based keyword score.

    Counts how many query terms appear in the document, normalized by
    the number of query terms.
    """
    query_terms = set(re.findall(r"\w+", query.lower()))
    if not query_terms:
        return 0.0
    text_lower = text.lower()
    text_words = re.findall(r"\w+", text_lower)
    text_word_counts: dict[str, int] = {}
    for w in text_words:
        text_word_counts[w] = text_word_counts.get(w, 0) + 1

    score = 0.0
    for term in query_terms:
        score += text_word_counts.get(term, 0)
    # Normalize by document length to avoid bias toward longer docs
    if text_words:
        score = score / len(text_words)
    return score


@tool()
def vector_search(
    collection: str,
    query: str,
    top_k: int = 5,
    threshold: float = 0.0,
    store_path: str = None,
) -> dict:
    """Search documents by semantic similarity.

    Performs cosine-similarity search over a vector collection.
    Embeds the query text and compares against stored embeddings,
    returning the most similar documents ranked by score.

    Args:
        collection: Name of the collection to search.
        query: The query text to search for.
        top_k: Maximum number of results to return.
        threshold: Minimum similarity score to include in results.
        store_path: Path to the JSON store file. Uses in-memory store if omitted.
    """
    if not collection:
        raise ValueError("Parameter 'collection' is required")
    if not query:
        raise ValueError("Parameter 'query' is required")

    top_k = int(top_k)
    threshold = float(threshold)

    store = _load_store(store_path)
    docs = store.get(collection, [])

    if not docs:
        return {"results": [], "count": 0}

    query_embedding = _builtin_embed(query)

    scored = []
    for doc in docs:
        score = _cosine_similarity(query_embedding, doc["embedding"])
        if score >= threshold:
            scored.append(
                {
                    "text": doc["text"],
                    "metadata": doc.get("metadata", {}),
                    "score": round(score, 6),
                }
            )

    scored.sort(key=lambda x: x["score"], reverse=True)
    results = scored[:top_k]

    return {"results": results, "count": len(results)}


@tool()
def hybrid_search(
    collection: str,
    query: str,
    top_k: int = 5,
    keyword_weight: float = 0.3,
    store_path: str = None,
) -> dict:
    """Hybrid semantic + keyword search.

    Combines cosine-similarity vector search with TF-based keyword
    scoring. The final score is a weighted blend:
    (1 - keyword_weight) * vector_score + keyword_weight * keyword_score.

    Args:
        collection: Name of the collection to search.
        query: The query text to search for.
        top_k: Maximum number of results to return.
        keyword_weight: Weight for keyword scoring (0.0-1.0). Semantic weight is 1 - keyword_weight.
        store_path: Path to the JSON store file. Uses in-memory store if omitted.
    """
    if not collection:
        raise ValueError("Parameter 'collection' is required")
    if not query:
        raise ValueError("Parameter 'query' is required")

    top_k = int(top_k)
    keyword_weight = float(keyword_weight)

    store = _load_store(store_path)
    docs = store.get(collection, [])

    if not docs:
        return {"results": [], "count": 0}

    query_embedding = _builtin_embed(query)

    # Compute raw keyword scores for normalization
    raw_keyword_scores = [_keyword_score(query, doc["text"]) for doc in docs]
    max_kw = max(raw_keyword_scores) if raw_keyword_scores else 0.0

    scored = []
    for doc, raw_kw in zip(docs, raw_keyword_scores):
        vector_score = _cosine_similarity(query_embedding, doc["embedding"])
        # Normalize keyword score to [0, 1]
        kw_score = (raw_kw / max_kw) if max_kw > 0 else 0.0
        combined = (1.0 - keyword_weight) * vector_score + keyword_weight * kw_score
        scored.append(
            {
                "text": doc["text"],
                "metadata": doc.get("metadata", {}),
                "score": round(combined, 6),
                "vector_score": round(vector_score, 6),
                "keyword_score": round(kw_score, 6),
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    results = scored[:top_k]

    return {"results": results, "count": len(results)}
