"""
ParseXMLTool: Parse XML data from a file path or string.

Uses the stdlib ``xml.etree.ElementTree`` module with optional XPath queries.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Any

from quartermaster_tools.base import AbstractTool
from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolResult


def _element_to_dict(element: ET.Element) -> dict[str, Any]:
    """Recursively convert an ElementTree element to a dict.

    Attributes are stored under ``@attr``, text under ``#text``,
    and child elements are nested by tag name. Repeated tags become lists.
    """
    result: dict[str, Any] = {}

    # Attributes
    for key, value in element.attrib.items():
        result[f"@{key}"] = value

    # Text content
    text = (element.text or "").strip()
    if text:
        result["#text"] = text

    # Children
    for child in element:
        child_dict = _element_to_dict(child)
        tag = child.tag
        if tag in result:
            existing = result[tag]
            if isinstance(existing, list):
                existing.append(child_dict)
            else:
                result[tag] = [existing, child_dict]
        else:
            result[tag] = child_dict

    return result


class ParseXMLTool(AbstractTool):
    """Parse XML content from a file path or raw string.

    Converts XML elements into nested dicts. Supports optional XPath
    queries to select specific elements.
    """

    def name(self) -> str:
        """Return the tool name."""
        return "parse_xml"

    def version(self) -> str:
        """Return the tool version."""
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        """Return parameter definitions for the tool."""
        return [
            ToolParameter(
                name="source",
                description="File path or raw XML string to parse.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="xpath",
                description="Optional XPath expression to select elements.",
                type="string",
                required=False,
                default=None,
            ),
        ]

    def info(self) -> ToolDescriptor:
        """Return metadata describing this tool."""
        return ToolDescriptor(
            name=self.name(),
            short_description="Parse XML data from a file or string.",
            long_description=(
                "Reads XML content from a file path or inline string. "
                "Converts elements to nested dicts and supports optional XPath queries."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    @staticmethod
    def _read_source(source: str) -> str:
        """Return XML text, reading from file if *source* is a file path."""
        if os.path.isfile(source):
            with open(source, "r", encoding="utf-8") as fh:
                return fh.read()
        return source

    def parse(self, source: str, xpath: str | None = None) -> Any:
        """Parse XML from *source* and optionally select elements via XPath.

        Args:
            source: File path or raw XML string.
            xpath: Optional XPath expression to select elements.

        Returns:
            A dict (full document or single match) or list of dicts (multiple matches).

        Raises:
            ValueError: When the source cannot be parsed as XML.
        """
        text = self._read_source(source)
        try:
            root = ET.fromstring(text)
        except ET.ParseError as exc:
            raise ValueError(f"Invalid XML: {exc}") from exc

        if xpath is not None:
            elements = root.findall(xpath)
            results = [_element_to_dict(el) for el in elements]
            if len(results) == 1:
                return results[0]
            return results

        return _element_to_dict(root)

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the XML parse tool.

        Args:
            source: File path or raw XML string.
            xpath: Optional XPath expression.

        Returns:
            ToolResult with parsed data in ``data["result"]``.
        """
        source: str = kwargs.get("source", "")
        xpath: str | None = kwargs.get("xpath")

        if not source:
            return ToolResult(success=False, error="Parameter 'source' is required")

        try:
            result = self.parse(source, xpath=xpath)
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

        return ToolResult(success=True, data={"result": result})
