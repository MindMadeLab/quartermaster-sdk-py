"""
parse_xml: Parse XML data from a file path or string.

Uses the stdlib ``xml.etree.ElementTree`` module with optional XPath queries.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Any

from quartermaster_tools.decorator import tool


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


def _read_source(source: str) -> str:
    """Return XML text, reading from file if *source* is a file path."""
    if os.path.isfile(source):
        with open(source, "r", encoding="utf-8") as fh:
            return fh.read()
    return source


def _parse_xml(source: str, xpath: str | None = None) -> Any:
    """Parse XML from *source* and optionally select elements via XPath.

    Args:
        source: File path or raw XML string.
        xpath: Optional XPath expression to select elements.

    Returns:
        A dict (full document or single match) or list of dicts (multiple matches).

    Raises:
        ValueError: When the source cannot be parsed as XML.
    """
    text = _read_source(source)
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


@tool()
def parse_xml(source: str, xpath: str = None) -> dict:
    """Parse XML data from a file or string.

    Reads XML content from a file path or inline string. Converts elements
    to nested dicts and supports optional XPath queries.

    Args:
        source: File path or raw XML string to parse.
        xpath: Optional XPath expression to select elements.
    """
    if not source:
        return {"error": "Parameter 'source' is required"}

    try:
        result = _parse_xml(source, xpath=xpath)
    except Exception as exc:
        return {"error": str(exc)}

    return {"result": result}


# Backward-compatible alias
ParseXMLTool = parse_xml

# Alias used by tests
parse_xml_data = _parse_xml
