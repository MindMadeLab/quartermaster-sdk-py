"""
GrepTool: Search file contents for a regex pattern.
"""

from __future__ import annotations

import os
import re
from typing import Any

from quartermaster_tools.base import AbstractTool
from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolResult

from ._security import resolve_base_dir, validate_path


class GrepTool(AbstractTool):
    """Search file contents for a regex pattern, optionally recursive."""

    def __init__(self, allowed_base_dir: str | None = None) -> None:
        """Initialise the GrepTool.

        Args:
            allowed_base_dir: If set, only paths under this directory are allowed.
        """
        self._allowed_base_dir = resolve_base_dir(allowed_base_dir)

    def name(self) -> str:
        """Return the tool name."""
        return "grep"

    def version(self) -> str:
        """Return the tool version."""
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        """Return parameter definitions."""
        return [
            ToolParameter(name="path", description="File or directory to search.", type="string", required=True),
            ToolParameter(name="pattern", description="Regex pattern to search for.", type="string", required=True),
            ToolParameter(name="recursive", description="Recurse into subdirectories.", type="boolean", default=True),
            ToolParameter(name="context_lines", description="Number of context lines around matches.", type="number", default=0),
        ]

    def info(self) -> ToolDescriptor:
        """Return tool metadata."""
        return ToolDescriptor(
            name=self.name(),
            short_description="Search file contents for a regex pattern.",
            long_description="Searches file(s) for lines matching a regex pattern, with optional context lines and recursive directory scanning.",
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        """Search for a pattern in file contents.

        Args:
            path: File or directory to search.
            pattern: Regex pattern.
            recursive: Recurse into subdirectories (default True).
            context_lines: Lines of context around each match (default 0).

        Returns:
            ToolResult with data["matches"] list of match dicts.
        """
        path: str = kwargs.get("path", "")
        pattern: str = kwargs.get("pattern", "")
        recursive: bool = kwargs.get("recursive", True)
        context_lines: int = int(kwargs.get("context_lines", 0))

        if not path:
            return ToolResult(success=False, error="Parameter 'path' is required")
        if not pattern:
            return ToolResult(success=False, error="Parameter 'pattern' is required")

        try:
            regex = re.compile(pattern)
        except re.error as e:
            return ToolResult(success=False, error=f"Invalid regex: {e}")

        error, real_path = validate_path(path, self._allowed_base_dir)
        if error:
            return ToolResult(success=False, error=error)

        if not os.path.exists(real_path):
            return ToolResult(success=False, error=f"Path not found: {path}")

        matches: list[dict[str, Any]] = []

        if os.path.isfile(real_path):
            self._search_file(real_path, regex, context_lines, matches)
        elif os.path.isdir(real_path):
            if recursive:
                for dirpath, _dirnames, filenames in os.walk(real_path):
                    for fname in sorted(filenames):
                        fpath = os.path.join(dirpath, fname)
                        self._search_file(fpath, regex, context_lines, matches)
            else:
                for fname in sorted(os.listdir(real_path)):
                    fpath = os.path.join(real_path, fname)
                    if os.path.isfile(fpath):
                        self._search_file(fpath, regex, context_lines, matches)

        return ToolResult(success=True, data={"matches": matches, "total_matches": len(matches)})

    @staticmethod
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
