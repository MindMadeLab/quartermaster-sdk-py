"""
ListDirectoryTool: List directory entries with optional recursion and pattern filtering.
"""

from __future__ import annotations

import fnmatch
import os
from typing import Any

from quartermaster_tools.decorator import tool

from ._security import validate_path


def _entry_info(full_path: str, base: str) -> dict[str, Any]:
    """Build an entry info dict for a single path."""
    try:
        stat = os.stat(full_path)
        return {
            "name": os.path.relpath(full_path, base),
            "type": "directory" if os.path.isdir(full_path) else "file",
            "size": stat.st_size,
            "modified": stat.st_mtime,
        }
    except OSError:
        return {
            "name": os.path.relpath(full_path, base),
            "type": "unknown",
            "size": 0,
            "modified": 0,
        }


@tool()
def list_directory(
    path: str, recursive: bool = False, pattern: str = "*", include_hidden: bool = False
) -> dict:
    """List directory entries with type, size, and modification time.

    Args:
        path: Directory path to list.
        recursive: Whether to list recursively.
        pattern: Glob pattern to filter entries.
        include_hidden: Include hidden files (dot-prefixed).
    """
    if not path:
        return {"error": "Parameter 'path' is required"}

    error, real_path = validate_path(path, None)
    if error:
        return {"error": error}

    if not os.path.exists(real_path):
        return {"error": f"Directory not found: {path}"}
    if not os.path.isdir(real_path):
        return {"error": f"Not a directory: {path}"}

    entries: list[dict[str, Any]] = []
    try:
        if recursive:
            for dirpath, dirnames, filenames in os.walk(real_path):
                if not include_hidden:
                    dirnames[:] = [d for d in dirnames if not d.startswith(".")]
                for name in dirnames + filenames:
                    if not include_hidden and name.startswith("."):
                        continue
                    if not fnmatch.fnmatch(name, pattern):
                        continue
                    full = os.path.join(dirpath, name)
                    entries.append(_entry_info(full, real_path))
        else:
            for name in sorted(os.listdir(real_path)):
                if not include_hidden and name.startswith("."):
                    continue
                if not fnmatch.fnmatch(name, pattern):
                    continue
                full = os.path.join(real_path, name)
                entries.append(_entry_info(full, real_path))
    except PermissionError as e:
        return {"error": f"Permission denied: {e}"}

    return {"entries": entries, "count": len(entries)}
