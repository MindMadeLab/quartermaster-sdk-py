"""
ConvertFormatTool: Convert structured data between CSV, JSON, and YAML formats.

Delegates parsing to the individual parse tools and formats the output
for the target format.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any

from quartermaster_tools.base import AbstractTool
from quartermaster_tools.builtin.data.parse_csv import ParseCSVTool
from quartermaster_tools.builtin.data.parse_json import ParseJSONTool
from quartermaster_tools.builtin.data.parse_yaml import ParseYAMLTool
from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolParameterOption, ToolResult

try:
    import yaml as _yaml  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    _yaml = None

_SUPPORTED_FORMATS = ("csv", "json", "yaml")


class ConvertFormatTool(AbstractTool):
    """Convert data between CSV, JSON, and YAML formats.

    Uses the parse tools to read the source format and then serialises
    to the target format.
    """

    def __init__(self) -> None:
        """Initialise with internal parse tool instances."""
        self._csv_tool = ParseCSVTool()
        self._json_tool = ParseJSONTool()
        self._yaml_tool = ParseYAMLTool()

    def name(self) -> str:
        """Return the tool name."""
        return "convert_format"

    def version(self) -> str:
        """Return the tool version."""
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        """Return parameter definitions for the tool."""
        fmt_options = [ToolParameterOption(label=f, value=f) for f in _SUPPORTED_FORMATS]
        return [
            ToolParameter(
                name="source",
                description="Data string or file path to convert.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="from_format",
                description="Source data format.",
                type="string",
                required=True,
                options=fmt_options,
            ),
            ToolParameter(
                name="to_format",
                description="Target data format.",
                type="string",
                required=True,
                options=fmt_options,
            ),
        ]

    def info(self) -> ToolDescriptor:
        """Return metadata describing this tool."""
        return ToolDescriptor(
            name=self.name(),
            short_description="Convert data between CSV, JSON, and YAML.",
            long_description=(
                "Parses data from one format (CSV, JSON, YAML) and "
                "serialises it to another. Delegates to the individual "
                "parse tools for reading and uses stdlib/pyyaml for writing."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def _parse_input(self, source: str, from_format: str) -> Any:
        """Parse *source* according to *from_format*."""
        if from_format == "csv":
            return self._csv_tool.parse(source)
        if from_format == "json":
            return self._json_tool.parse(source)
        if from_format == "yaml":
            return self._yaml_tool.parse(source)
        raise ValueError(f"Unsupported source format: {from_format!r}")

    @staticmethod
    def _to_csv(data: Any) -> str:
        """Serialise a list-of-dicts to CSV string."""
        if not isinstance(data, list) or not data:
            raise ValueError("CSV output requires a non-empty list of dicts")

        # If items are dicts, use keys as headers
        if isinstance(data[0], dict):
            fieldnames = list(data[0].keys())
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
            return buf.getvalue()

        # list of lists
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerows(data)
        return buf.getvalue()

    @staticmethod
    def _to_json(data: Any) -> str:
        """Serialise data to a JSON string."""
        return json.dumps(data, indent=2, ensure_ascii=False)

    @staticmethod
    def _to_yaml(data: Any) -> str:
        """Serialise data to a YAML string."""
        if _yaml is None:
            raise RuntimeError(
                "pyyaml is not installed. Install it with: pip install pyyaml"
            )
        return _yaml.dump(data, default_flow_style=False, allow_unicode=True)

    def convert(self, source: str, from_format: str, to_format: str) -> str:
        """Convert *source* from one format to another.

        Args:
            source: Data string or file path.
            from_format: One of ``"csv"``, ``"json"``, ``"yaml"``.
            to_format: One of ``"csv"``, ``"json"``, ``"yaml"``.

        Returns:
            Serialised string in the target format.

        Raises:
            ValueError: On unsupported format or incompatible data shapes.
        """
        from_format = from_format.lower().strip()
        to_format = to_format.lower().strip()

        for fmt in (from_format, to_format):
            if fmt not in _SUPPORTED_FORMATS:
                raise ValueError(
                    f"Unsupported format: {fmt!r}. Must be one of {_SUPPORTED_FORMATS}"
                )

        data = self._parse_input(source, from_format)

        if to_format == "csv":
            return self._to_csv(data)
        if to_format == "json":
            return self._to_json(data)
        if to_format == "yaml":
            return self._to_yaml(data)

        raise ValueError(f"Unsupported target format: {to_format!r}")  # pragma: no cover

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the format conversion tool.

        Args:
            source: Data string or file path.
            from_format: Source format (csv, json, yaml).
            to_format: Target format (csv, json, yaml).

        Returns:
            ToolResult with converted string in ``data["output"]``.
        """
        source: str = kwargs.get("source", "")
        from_format: str = kwargs.get("from_format", "")
        to_format: str = kwargs.get("to_format", "")

        if not source:
            return ToolResult(success=False, error="Parameter 'source' is required")
        if not from_format:
            return ToolResult(success=False, error="Parameter 'from_format' is required")
        if not to_format:
            return ToolResult(success=False, error="Parameter 'to_format' is required")

        try:
            output = self.convert(source, from_format, to_format)
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

        return ToolResult(success=True, data={"output": output})
