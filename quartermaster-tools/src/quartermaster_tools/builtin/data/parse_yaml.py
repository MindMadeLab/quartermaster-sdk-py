"""
parse_yaml: Parse YAML data from a file path or string.

Uses the ``pyyaml`` library (``yaml`` package).
"""

from __future__ import annotations

import os
from typing import Any

from quartermaster_tools.decorator import tool

try:
    import yaml as _yaml  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    _yaml = None


def _read_source(source: str) -> str:
    """Return YAML text, reading from file if *source* is a file path."""
    if os.path.isfile(source):
        with open(source, "r", encoding="utf-8") as fh:
            return fh.read()
    return source


def _parse_yaml(source: str) -> Any:
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
        raise RuntimeError("pyyaml is not installed. Install it with: pip install pyyaml")

    text = _read_source(source)
    try:
        return _yaml.safe_load(text)
    except _yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML: {exc}") from exc


@tool()
def parse_yaml(source: str) -> dict:
    """Parse YAML data from a file or string.

    Reads YAML content from a file path or inline string and returns
    the parsed Python data structure.

    Args:
        source: File path or raw YAML string to parse.
    """
    if not source:
        return {"error": "Parameter 'source' is required"}

    try:
        result = _parse_yaml(source)
    except Exception as exc:
        return {"error": str(exc)}

    return {"result": result}


# Backward-compatible alias
ParseYAMLTool = parse_yaml

# Alias used by convert_format
parse_yaml_data = _parse_yaml
