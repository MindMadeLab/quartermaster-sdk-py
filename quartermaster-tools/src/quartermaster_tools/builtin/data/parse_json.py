"""
parse_json: Parse JSON data from a file path or string.

Uses the stdlib ``json`` module. Optionally applies a JMESPath query
when the ``jmespath`` library is installed.
"""

from __future__ import annotations

import json
import os
from typing import Any

from quartermaster_tools.decorator import tool

try:
    import jmespath as _jmespath  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    _jmespath = None


def _read_source(source: str) -> str:
    """Return JSON text, reading from file if *source* is a file path."""
    if os.path.isfile(source):
        with open(source, "r", encoding="utf-8") as fh:
            return fh.read()
    return source


def _parse_json(source: str, query: str | None = None) -> Any:
    """Parse JSON from *source* and optionally apply a JMESPath query.

    Args:
        source: File path or raw JSON string.
        query: Optional JMESPath expression.

    Returns:
        Parsed (and optionally queried) data.

    Raises:
        ValueError: When the source cannot be parsed as JSON.
        RuntimeError: When a query is provided but jmespath is not installed.
    """
    text = _read_source(source)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc

    if query is not None:
        if _jmespath is None:
            raise RuntimeError("jmespath is not installed. Install it with: pip install jmespath")
        data = _jmespath.search(query, data)

    return data


@tool()
def parse_json(source: str, query: str = None) -> dict:
    """Parse JSON data from a file or string.

    Reads JSON content from a file path or inline string. Optionally
    applies a JMESPath query to extract or transform data.

    Args:
        source: File path or raw JSON string to parse.
        query: Optional JMESPath query to apply to the parsed data.
    """
    if not source:
        return {"error": "Parameter 'source' is required"}

    try:
        result = _parse_json(source, query=query)
    except Exception as exc:
        return {"error": str(exc)}

    return {"result": result}


# Alias used by convert_format
parse_json_data = _parse_json
