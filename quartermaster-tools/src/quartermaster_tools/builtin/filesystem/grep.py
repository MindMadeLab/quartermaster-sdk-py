"""
GrepTool: Search file contents for a regex pattern.
"""

from __future__ import annotations

import os
import re
from typing import Any

from quartermaster_tools.decorator import tool

from ._security import validate_path


def _search_file(
    file_path: str,
    regex: re.Pattern[str],
    context_lines: int,
    results: list[dict[str, Any]],
) -> None:
    """Search a single file for regex matches."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except (OSError, PermissionError):
        return

    for i, line in enumerate(lines):
        if regex.search(line):
            start = max(0, i - context_lines)
            end = min(len(lines), i + context_lines + 1)
            context = [l.rstrip("\n\r") for l in lines[start:end]]
            results.append({
                "file": file_path,
                "line_number": i + 1,
                "line": line.rstrip("\n\r"),
                "context": context if context_lines > 0 else [],
            })


@tool()
def grep(path: str, pattern: str, recursive: bool = True, context_lines: int = 0) -> dict:
    """Search file contents for a regex pattern.

    Args:
        path: File or directory to search.
        pattern: Regex pattern to search for.
        recursive: Recurse into subdirectories.
        context_lines: Number of context lines around matches.
    """
    if not path:
        return {"error": "Parameter 'path' is required"}
    if not pattern:
        return {"error": "Parameter 'pattern' is required"}

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return {"error": f"Invalid regex: {e}"}

    error, real_path = validate_path(path, None)
    if error:
        return {"error": error}

    if not os.path.exists(real_path):
        return {"error": f"Path not found: {path}"}

    context_lines = int(context_lines)
    matches: list[dict[str, Any]] = []

    if os.path.isfile(real_path):
        _search_file(real_path, regex, context_lines, matches)
    elif os.path.isdir(real_path):
        if recursive:
            for dirpath, _dirnames, filenames in os.walk(real_path):
                for fname in sorted(filenames):
                    fpath = os.path.join(dirpath, fname)
                    _search_file(fpath, regex, context_lines, matches)
        else:
            for fname in sorted(os.listdir(real_path)):
                fpath = os.path.join(real_path, fname)
                if os.path.isfile(fpath):
                    _search_file(fpath, regex, context_lines, matches)

    return {"matches": matches, "total_matches": len(matches)}


# Backward-compatible alias
GrepTool = grep
