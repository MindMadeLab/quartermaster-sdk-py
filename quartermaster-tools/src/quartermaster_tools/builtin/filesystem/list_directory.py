"""
ListDirectoryTool: List directory entries with optional recursion and pattern filtering.
"""

from __future__ import annotations

import fnmatch
import os
from typing import Any

from quartermaster_tools.base import AbstractTool
from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolResult

from ._security import resolve_base_dir, validate_path


class ListDirectoryTool(AbstractTool):
    """List entries in a directory.

    Returns file names, types, sizes, and modification times.
    Supports recursive listing and glob pattern filtering.
    """

    def __init__(self, allowed_base_dir: str | None = None) -> None:
        """Initialise the ListDirectoryTool.

        Args:
            allowed_base_dir: If set, only paths under this directory are allowed.
        """
        self._allowed_base_dir = resolve_base_dir(allowed_base_dir)

    def name(self) -> str:
        """Return the tool name."""
        return "list_directory"

    def version(self) -> str:
        """Return the tool version."""
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        """Return parameter definitions."""
        return [
            ToolParameter(name="path", description="Directory path to list.", type="string", required=True),
            ToolParameter(name="recursive", description="Whether to list recursively.", type="boolean", default=False),
            ToolParameter(name="pattern", description="Glob pattern to filter entries.", type="string", default="*"),
            ToolParameter(name="include_hidden", description="Include hidden files (dot-prefixed).", type="boolean", default=False),
        ]

    def info(self) -> ToolDescriptor:
        """Return tool metadata."""
        return ToolDescriptor(
            name=self.name(),
            short_description="List directory entries with type, size, and modification time.",
            long_description="Lists entries in a directory, optionally recursing and filtering by glob pattern.",
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        """List directory contents.

        Args:
            path: Directory to list.
            recursive: Recurse into subdirectories (default False).
            pattern: Glob pattern for filtering (default '*').
            include_hidden: Include dot-prefixed entries (default False).

        Returns:
            ToolResult with data["entries"] as a list of entry dicts.
        """
        path: str = kwargs.get("path", "")
        recursive: bool = kwargs.get("recursive", False)
        pattern: str = kwargs.get("pattern", "*")
        include_hidden: bool = kwargs.get("include_hidden", False)

        if not path:
            return ToolResult(success=False, error="Parameter 'path' is required")

        error, real_path = validate_path(path, self._allowed_base_dir)
        if error:
            return ToolResult(success=False, error=error)

        if not os.path.exists(real_path):
            return ToolResult(success=False, error=f"Directory not found: {path}")
        if not os.path.isdir(real_path):
            return ToolResult(success=False, error=f"Not a directory: {path}")

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
                        entries.append(self._entry_info(full, real_path))
            else:
                for name in sorted(os.listdir(real_path)):
                    if not include_hidden and name.startswith("."):
                        continue
                    if not fnmatch.fnmatch(name, pattern):
                        continue
                    full = os.path.join(real_path, name)
                    entries.append(self._entry_info(full, real_path))
        except PermissionError as e:
            return ToolResult(success=False, error=f"Permission denied: {e}")

        return ToolResult(success=True, data={"entries": entries, "count": len(entries)})

    @staticmethod
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
