"""
parse_csv: Parse CSV data from a file path or string.

Uses the stdlib ``csv`` module to parse comma-separated (or custom-delimited)
values into structured Python objects.
"""

from __future__ import annotations

import csv
import io
import os

from quartermaster_tools.decorator import tool


def _read_source(source: str) -> str:
    """Return CSV text, reading from file if *source* is a file path."""
    if os.path.isfile(source):
        with open(source, "r", encoding="utf-8") as fh:
            return fh.read()
    return source


def _parse_csv(
    source: str,
    delimiter: str = ",",
    has_headers: bool = True,
) -> list[dict[str, str]] | list[list[str]]:
    """Parse CSV from *source* and return structured data.

    Args:
        source: File path or raw CSV string.
        delimiter: Column delimiter (default ``","``).
        has_headers: If ``True``, the first row is treated as column names.

    Returns:
        List of dicts when *has_headers* is True, otherwise list of lists.

    Raises:
        ValueError: When the source is empty or unreadable.
    """
    text = _read_source(source)
    if not text.strip():
        raise ValueError("CSV source is empty")

    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows = list(reader)

    if not rows:
        raise ValueError("CSV source contains no rows")

    if has_headers:
        headers = rows[0]
        return [dict(zip(headers, row)) for row in rows[1:]]
    return rows


@tool()
def parse_csv(source: str, delimiter: str = ",", has_headers: bool = True) -> dict:
    """Parse CSV data from a file or string.

    Reads CSV content from a file path or inline string. Supports custom
    delimiters and optional header rows. Returns list of dicts (with headers)
    or list of lists (without).

    Args:
        source: File path or raw CSV string to parse.
        delimiter: Column delimiter character.
        has_headers: Whether the first row contains column headers.
    """
    if not source:
        return {"error": "Parameter 'source' is required"}

    try:
        rows = _parse_csv(source, delimiter=delimiter, has_headers=has_headers)
    except Exception as exc:
        return {"error": str(exc)}

    return {"rows": rows, "count": len(rows)}


# Alias used by convert_format
parse_csv_data = _parse_csv
