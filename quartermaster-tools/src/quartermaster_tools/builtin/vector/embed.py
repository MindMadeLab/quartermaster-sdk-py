"""
Text embedding tool.

Generates vector embeddings for text using either a built-in deterministic
hash-based approach (zero dependencies) or sentence-transformers if available.
"""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Any

from quartermaster_tools.base import AbstractTool
from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolResult


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


class EmbedTextTool(AbstractTool):
    """Generate vector embeddings for a text string."""

    def name(self) -> str:
        return "embed_text"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="text",
                description="The text to generate an embedding for.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="model",
                description="Embedding model to use. 'builtin' for hash-based, or a sentence-transformers model name.",
                type="string",
                required=False,
                default="builtin",
            ),
            ToolParameter(
                name="dimensions",
                description="Number of embedding dimensions (only for builtin model).",
                type="number",
                required=False,
                default=384,
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Generate vector embeddings for text.",
            long_description=(
                "Produces a numerical vector representation of text. "
                "Uses a built-in hash-based approach by default (zero dependencies, "
                "deterministic but not semantic). If sentence-transformers is installed, "
                "can use real transformer models for semantic embeddings."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        text: str = kwargs.get("text", "")
        model: str = kwargs.get("model", "builtin")
        dimensions: int = int(kwargs.get("dimensions", 384))

        if not text:
            return ToolResult(success=False, error="Parameter 'text' is required")

        if model == "builtin":
            embedding = _builtin_embed(text, dimensions)
            return ToolResult(
                success=True,
                data={
                    "embedding": embedding,
                    "dimensions": dimensions,
                    "model": "builtin",
                },
            )

        # Try sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer

            st_model = SentenceTransformer(model)
            embedding = st_model.encode(text).tolist()
            return ToolResult(
                success=True,
                data={
                    "embedding": embedding,
                    "dimensions": len(embedding),
                    "model": model,
                },
            )
        except ImportError:
            return ToolResult(
                success=False,
                error=(
                    f"Model '{model}' requires sentence-transformers. "
                    "Install with: pip install sentence-transformers"
                ),
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Embedding failed: {e}")
