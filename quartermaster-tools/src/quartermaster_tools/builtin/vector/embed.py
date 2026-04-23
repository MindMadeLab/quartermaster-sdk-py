"""
Text embedding tool.

Generates vector embeddings for text using either a built-in deterministic
hash-based approach (zero dependencies) or sentence-transformers if available.
"""

from __future__ import annotations

import hashlib
import math
import struct

from quartermaster_tools.decorator import tool


def _builtin_embed(text: str, dimensions: int = 384) -> list[float]:
    """Generate a deterministic embedding vector from text using hashing.

    Produces a consistent float vector by hashing overlapping character
    n-grams of the input text.  The result is L2-normalized so that
    cosine similarity works correctly.

    This is NOT a semantic embedding -- it is a deterministic fingerprint
    useful for testing and zero-dependency operation.
    """
    vector = [0.0] * dimensions

    # Hash the full text and overlapping 3-grams to populate the vector
    tokens = [text] + [text[i : i + 3] for i in range(max(1, len(text) - 2))]
    for token in tokens:
        h = hashlib.sha256(token.encode("utf-8")).digest()
        # Use 4-byte chunks from the hash to seed positions
        for offset in range(0, len(h) - 3, 4):
            raw = struct.unpack_from("<I", h, offset)[0]
            idx = raw % dimensions
            # Convert to a float in [-1, 1]
            value = (raw / 0xFFFFFFFF) * 2.0 - 1.0
            vector[idx] += value

    # L2-normalize
    norm = math.sqrt(sum(v * v for v in vector))
    if norm > 0:
        vector = [v / norm for v in vector]

    return vector


@tool()
def embed_text(text: str, model: str = "builtin", dimensions: int = 384) -> dict:
    """Generate vector embeddings for text.

    Produces a numerical vector representation of text.
    Uses a built-in hash-based approach by default (zero dependencies,
    deterministic but not semantic). If sentence-transformers is installed,
    can use real transformer models for semantic embeddings.

    Args:
        text: The text to generate an embedding for.
        model: Embedding model to use. 'builtin' for hash-based, or a sentence-transformers model name.
        dimensions: Number of embedding dimensions (only for builtin model).
    """
    if not text:
        raise ValueError("Parameter 'text' is required")

    dimensions = int(dimensions)

    if model == "builtin":
        embedding = _builtin_embed(text, dimensions)
        return {
            "embedding": embedding,
            "dimensions": dimensions,
            "model": "builtin",
        }

    # Try sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer

        st_model = SentenceTransformer(model)
        embedding = st_model.encode(text).tolist()
        return {
            "embedding": embedding,
            "dimensions": len(embedding),
            "model": model,
        }
    except ImportError:
        raise ImportError(
            f"Model '{model}' requires sentence-transformers. "
            "Install with: pip install sentence-transformers"
        )
    except Exception as e:
        raise RuntimeError(f"Embedding failed: {e}")
