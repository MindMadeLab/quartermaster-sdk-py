"""
ParseYAMLTool: Parse YAML data from a file path or string.

Uses the ``pyyaml`` library (``yaml`` package).
"""

from __future__ import annotations

import os
from typing import Any

from quartermaster_tools.base import AbstractTool
from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolResult

try:
    import yaml as _yaml  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    _yaml = None


class ParseYAMLTool(AbstractTool):
    """Parse YAML content from a file path or raw string."""

    def name(self) -> str:
        """Return the tool name."""
        return "parse_yaml"

    def version(self) -> str:
        """Return the tool version."""
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        """Return parameter definitions for the tool."""
        return [
            ToolParameter(
                name="source",
                description="File path or raw YAML string to parse.",
                type="string",
                required=True,
            ),
        ]

    def info(self) -> ToolDescriptor:
        """Return metadata describing this tool."""
        return ToolDescriptor(
            name=self.name(),
            short_description="Parse YAML data from a file or string.",
            long_description=(
                "Reads YAML content from a file path or inline string "
                "and returns the parsed Python data structure."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    @staticmethod
    def _read_source(source: str) -> str:
        """Return YAML text, reading from file if *source* is a file path."""
        if os.path.isfile(source):
            with open(source, "r", encoding="utf-8") as fh:
                return fh.read()
        return source

    def parse(self, source: str) -> Any:
        """Parse YAML from *source*.

        Args:
            source: File path or raw YAML string.

        Returns:
            Parsed Python data structure.

        Raises:
            RuntimeError: When pyyaml is not installed.
            ValueError: When the source cannot be parsed as YAML.
        """
        if _yaml is None:
            raise RuntimeError(
                "pyyaml is not installed. Install it with: pip install pyyaml"
            )

        text = self._read_source(source)
        try:
            return _yaml.safe_load(text)
        except _yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML: {exc}") from exc

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the YAML parse tool.

        Args:
            source: File path or raw YAML string.

        Returns:
            ToolResult with parsed data in ``data["result"]``.
        """
        source: str = kwargs.get("source", "")

        if not source:
            return ToolResult(success=False, error="Parameter 'source' is required")

        try:
            result = self.parse(source)
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

        return ToolResult(success=True, data={"result": result})
