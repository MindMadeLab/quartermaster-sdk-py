"""
convert_format: Convert structured data between CSV, JSON, and YAML formats.

Delegates parsing to the individual parse helper functions and formats
the output for the target format.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any

from quartermaster_tools.builtin.data.parse_csv import _parse_csv
from quartermaster_tools.builtin.data.parse_json import _parse_json
from quartermaster_tools.builtin.data.parse_yaml import _parse_yaml
from quartermaster_tools.decorator import tool

try:
    import yaml as _yaml  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    _yaml = None

_SUPPORTED_FORMATS = ("csv", "json", "yaml")


def _parse_input(source: str, from_format: str) -> Any:
    """Parse *source* according to *from_format*."""
    if from_format == "csv":
        return _parse_csv(source)
    if from_format == "json":
        return _parse_json(source)
    if from_format == "yaml":
        return _parse_yaml(source)
    raise ValueError(f"Unsupported source format: {from_format!r}")


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


def _to_json(data: Any) -> str:
    """Serialise data to a JSON string."""
    return json.dumps(data, indent=2, ensure_ascii=False)


def _to_yaml(data: Any) -> str:
    """Serialise data to a YAML string."""
    if _yaml is None:
        raise RuntimeError(
            "pyyaml is not installed. Install it with: pip install pyyaml"
        )
    return _yaml.dump(data, default_flow_style=False, allow_unicode=True)


def _convert(source: str, from_format: str, to_format: str) -> str:
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

    data = _parse_input(source, from_format)

    if to_format == "csv":
        return _to_csv(data)
    if to_format == "json":
        return _to_json(data)
    if to_format == "yaml":
        return _to_yaml(data)

    raise ValueError(f"Unsupported target format: {to_format!r}")  # pragma: no cover


@tool()
def convert_format(source: str, from_format: str, to_format: str) -> dict:
    """Convert data between CSV, JSON, and YAML.

    Parses data from one format (CSV, JSON, YAML) and serialises it to
    another. Delegates to the individual parse helpers for reading and
    uses stdlib/pyyaml for writing.

    Args:
        source: Data string or file path to convert.
        from_format: Source data format (csv, json, yaml).
        to_format: Target data format (csv, json, yaml).
    """
    if not source:
        return {"error": "Parameter 'source' is required"}
    if not from_format:
        return {"error": "Parameter 'from_format' is required"}
    if not to_format:
        return {"error": "Parameter 'to_format' is required"}

    try:
        output = _convert(source, from_format, to_format)
    except Exception as exc:
        return {"error": str(exc)}

    return {"output": output}


# Backward-compatible alias
ConvertFormatTool = convert_format

# Public alias for the convert helper
convert_format_data = _convert
