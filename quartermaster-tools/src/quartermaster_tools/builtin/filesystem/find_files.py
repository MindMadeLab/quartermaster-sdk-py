"""
FindFilesTool: Find files using glob patterns and optional regex name filtering.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from quartermaster_tools.decorator import tool

from ._security import validate_path


@tool()
def find_files(root_path: str, pattern: str, name_pattern: str = None) -> dict:
    """Find files using glob and optional regex filtering.

    Args:
        root_path: Root directory to search from.
        pattern: Glob pattern (e.g. '**/*.py').
        name_pattern: Optional regex to further filter file names.
    """
    if not root_path:
        return {"error": "Parameter 'root_path' is required"}
    if not pattern:
        return {"error": "Parameter 'pattern' is required"}

    error, real_root = validate_path(root_path, None)
    if error:
        return {"error": error}

    if not os.path.isdir(real_root):
        return {"error": f"Not a directory: {root_path}"}

    # Compile optional regex
    name_re = None
    if name_pattern:
        try:
            name_re = re.compile(name_pattern)
        except re.error as e:
            return {"error": f"Invalid regex: {e}"}

    root = Path(real_root)
    matches: list[str] = []
    try:
        for p in sorted(root.glob(pattern)):
            if name_re and not name_re.search(p.name):
                continue
            matches.append(str(p))
    except OSError as e:
        return {"error": f"Error scanning files: {e}"}

    return {"files": matches, "count": len(matches)}


# Backward-compatible alias
FindFilesTool = find_files
