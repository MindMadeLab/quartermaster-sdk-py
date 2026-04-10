"""
FindFilesTool: Find files using glob patterns and optional regex name filtering.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from quartermaster_tools.base import AbstractTool
from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolResult

from ._security import resolve_base_dir, validate_path


class FindFilesTool(AbstractTool):
    """Find files matching a glob pattern and optional regex on file names."""

    def __init__(self, allowed_base_dir: str | None = None) -> None:
        """Initialise the FindFilesTool.

        Args:
            allowed_base_dir: If set, only paths under this directory are allowed.
        """
        self._allowed_base_dir = resolve_base_dir(allowed_base_dir)

    def name(self) -> str:
        """Return the tool name."""
        return "find_files"

    def version(self) -> str:
        """Return the tool version."""
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        """Return parameter definitions."""
        return [
            ToolParameter(name="root_path", description="Root directory to search from.", type="string", required=True),
            ToolParameter(name="pattern", description="Glob pattern (e.g. '**/*.py').", type="string", required=True),
            ToolParameter(name="name_pattern", description="Optional regex to further filter file names.", type="string", default=None),
        ]

    def info(self) -> ToolDescriptor:
        """Return tool metadata."""
        return ToolDescriptor(
            name=self.name(),
            short_description="Find files using glob and optional regex filtering.",
            long_description="Searches for files matching a glob pattern under a root directory, with optional regex filtering on file names.",
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        """Find files matching the given patterns.

        Args:
            root_path: Directory to search from.
            pattern: Glob pattern for matching.
            name_pattern: Optional regex to filter file names.

        Returns:
            ToolResult with data["files"] list of matching paths.
        """
        root_path: str = kwargs.get("root_path", "")
        pattern: str = kwargs.get("pattern", "")
        name_pattern: str | None = kwargs.get("name_pattern")

        if not root_path:
            return ToolResult(success=False, error="Parameter 'root_path' is required")
        if not pattern:
            return ToolResult(success=False, error="Parameter 'pattern' is required")

        error, real_root = validate_path(root_path, self._allowed_base_dir)
        if error:
            return ToolResult(success=False, error=error)

        if not os.path.isdir(real_root):
            return ToolResult(success=False, error=f"Not a directory: {root_path}")

        # Compile optional regex
        name_re = None
        if name_pattern:
            try:
                name_re = re.compile(name_pattern)
            except re.error as e:
                return ToolResult(success=False, error=f"Invalid regex: {e}")

        root = Path(real_root)
        matches: list[str] = []
        try:
            for p in sorted(root.glob(pattern)):
                if name_re and not name_re.search(p.name):
                    continue
                matches.append(str(p))
        except OSError as e:
            return ToolResult(success=False, error=f"Error scanning files: {e}")

        return ToolResult(success=True, data={"files": matches, "count": len(matches)})
