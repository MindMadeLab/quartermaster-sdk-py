"""
ParseJSONTool: Parse JSON data from a file path or string.

Uses the stdlib ``json`` module. Optionally applies a JMESPath query
when the ``jmespath`` library is installed.
"""

from __future__ import annotations

import json
import os
from typing import Any

from quartermaster_tools.base import AbstractTool
from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolResult

try:
    import jmespath as _jmespath  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    _jmespath = None


class ParseJSONTool(AbstractTool):
    """Parse JSON content from a file path or raw string.

    Supports optional JMESPath queries for extracting nested data.
    """

    def name(self) -> str:
        """Return the tool name."""
        return "parse_json"

    def version(self) -> str:
        """Return the tool version."""
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        """Return parameter definitions for the tool."""
        return [
            ToolParameter(
                name="source",
                description="File path or raw JSON string to parse.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="query",
                description="Optional JMESPath query to apply to the parsed data.",
                type="string",
                required=False,
                default=None,
            ),
        ]

    def info(self) -> ToolDescriptor:
        """Return metadata describing this tool."""
        return ToolDescriptor(
            name=self.name(),
            short_description="Parse JSON data from a file or string.",
            long_description=(
                "Reads JSON content from a file path or inline string. "
                "Optionally applies a JMESPath query to extract or transform data."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    @staticmethod
    def _read_source(source: str) -> str:
        """Return JSON text, reading from file if *source* is a file path."""
        if os.path.isfile(source):
            with open(source, "r", encoding="utf-8") as fh:
                return fh.read()
        return source

    def parse(self, source: str, query: str | None = None) -> Any:
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
        text = self._read_source(source)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc

        if query is not None:
            if _jmespath is None:
                raise RuntimeError(
                    "jmespath is not installed. Install it with: pip install jmespath"
                )
            data = _jmespath.search(query, data)

        return data

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the JSON parse tool.

        Args:
            source: File path or raw JSON string.
            query: Optional JMESPath query.

        Returns:
            ToolResult with parsed data in ``data["result"]``.
        """
        source: str = kwargs.get("source", "")
        query: str | None = kwargs.get("query")

        if not source:
            return ToolResult(success=False, error="Parameter 'source' is required")

        try:
            result = self.parse(source, query=query)
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

        return ToolResult(success=True, data={"result": result})
