"""
ParseCSVTool: Parse CSV data from a file path or string.

Uses the stdlib ``csv`` module to parse comma-separated (or custom-delimited)
values into structured Python objects.
"""

from __future__ import annotations

import csv
import io
import os
from typing import Any

from quartermaster_tools.base import AbstractTool
from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolResult


class ParseCSVTool(AbstractTool):
    """Parse CSV content from a file path or raw string.

    Returns a list of dicts when headers are present, or a list of lists
    when ``has_headers`` is ``False``.
    """

    def name(self) -> str:
        """Return the tool name."""
        return "parse_csv"

    def version(self) -> str:
        """Return the tool version."""
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        """Return parameter definitions for the tool."""
        return [
            ToolParameter(
                name="source",
                description="File path or raw CSV string to parse.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="delimiter",
                description="Column delimiter character.",
                type="string",
                required=False,
                default=",",
            ),
            ToolParameter(
                name="has_headers",
                description="Whether the first row contains column headers.",
                type="boolean",
                required=False,
                default=True,
            ),
        ]

    def info(self) -> ToolDescriptor:
        """Return metadata describing this tool."""
        return ToolDescriptor(
            name=self.name(),
            short_description="Parse CSV data from a file or string.",
            long_description=(
                "Reads CSV content from a file path or inline string. "
                "Supports custom delimiters and optional header rows. "
                "Returns list of dicts (with headers) or list of lists (without)."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    @staticmethod
    def _read_source(source: str) -> str:
        """Return CSV text, reading from file if *source* is a file path."""
        if os.path.isfile(source):
            with open(source, "r", encoding="utf-8") as fh:
                return fh.read()
        return source

    def parse(
        self,
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
        text = self._read_source(source)
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

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the CSV parse tool.

        Args:
            source: File path or raw CSV string.
            delimiter: Column delimiter (default ``","``).
            has_headers: Whether the first row is headers (default ``True``).

        Returns:
            ToolResult with parsed rows in ``data["rows"]``.
        """
        source: str = kwargs.get("source", "")
        delimiter: str = kwargs.get("delimiter", ",")
        has_headers: bool = kwargs.get("has_headers", True)

        if not source:
            return ToolResult(success=False, error="Parameter 'source' is required")

        try:
            rows = self.parse(source, delimiter=delimiter, has_headers=has_headers)
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

        return ToolResult(success=True, data={"rows": rows, "count": len(rows)})
