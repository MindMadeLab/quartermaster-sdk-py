"""
ReadFileTool: Read file content from a path with security validation.

Validates paths to prevent directory traversal and enforces size limits
to avoid reading excessively large files into memory.
"""

from __future__ import annotations

import os
from typing import Any

from qm_tools.base import AbstractTool
from qm_tools.types import ToolDescriptor, ToolParameter, ToolResult

# Default maximum file size: 10 MB
DEFAULT_MAX_FILE_SIZE = 10 * 1024 * 1024

# Paths that are never allowed to be read
_BLOCKED_PREFIXES = (
    "/etc/shadow",
    "/etc/passwd",
    "/private/etc/shadow",
    "/private/etc/passwd",
    "/proc/",
    "/sys/",
    "/dev/",
)


class ReadFileTool(AbstractTool):
    """Read file content from a given path.

    Security features:
    - Resolves symlinks and rejects directory traversal attempts
    - Blocks access to sensitive system paths
    - Enforces a configurable maximum file size
    - Optionally restricts reads to an allowed base directory
    """

    def __init__(
        self,
        max_file_size: int = DEFAULT_MAX_FILE_SIZE,
        allowed_base_dir: str | None = None,
    ) -> None:
        """Initialise the ReadFileTool.

        Args:
            max_file_size: Maximum file size in bytes that can be read.
            allowed_base_dir: If set, only files under this directory may be read.
        """
        self._max_file_size = max_file_size
        self._allowed_base_dir = (
            os.path.realpath(allowed_base_dir) if allowed_base_dir else None
        )

    def name(self) -> str:
        """Return the tool name."""
        return "read_file"

    def version(self) -> str:
        """Return the tool version."""
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        """Return parameter definitions for the tool."""
        return [
            ToolParameter(
                name="path",
                description="Absolute or relative path to the file to read.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="encoding",
                description="Text encoding to use when reading the file.",
                type="string",
                required=False,
                default="utf-8",
            ),
        ]

    def info(self) -> ToolDescriptor:
        """Return metadata describing this tool."""
        return ToolDescriptor(
            name=self.name(),
            short_description="Read content from a file.",
            long_description=(
                "Reads the text content of a file at the given path. "
                "Includes path validation, size limits, and optional "
                "base-directory restriction for security."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def _validate_path(self, path: str) -> str | None:
        """Validate the file path and return an error message if invalid, else None."""
        real_path = os.path.realpath(path)

        for prefix in _BLOCKED_PREFIXES:
            if real_path.startswith(prefix):
                return f"Access denied: reading from '{prefix}' is not allowed"

        if self._allowed_base_dir is not None:
            if not real_path.startswith(self._allowed_base_dir + os.sep) and real_path != self._allowed_base_dir:
                return (
                    f"Access denied: path must be under '{self._allowed_base_dir}'"
                )

        return None

    def run(self, **kwargs: Any) -> ToolResult:
        """Read the file and return its content.

        Args:
            path: The file path to read.
            encoding: Text encoding (default utf-8).

        Returns:
            ToolResult with file content in data["content"].
        """
        path: str = kwargs.get("path", "")
        encoding: str = kwargs.get("encoding", "utf-8")

        if not path:
            return ToolResult(success=False, error="Parameter 'path' is required")

        # Validate path security
        error = self._validate_path(path)
        if error:
            return ToolResult(success=False, error=error)

        real_path = os.path.realpath(path)

        if not os.path.exists(real_path):
            return ToolResult(success=False, error=f"File not found: {path}")

        if not os.path.isfile(real_path):
            return ToolResult(success=False, error=f"Not a file: {path}")

        # Check file size before reading
        try:
            file_size = os.path.getsize(real_path)
        except OSError as e:
            return ToolResult(success=False, error=f"Cannot stat file: {e}")

        if file_size > self._max_file_size:
            return ToolResult(
                success=False,
                error=(
                    f"File too large: {file_size} bytes "
                    f"(limit: {self._max_file_size} bytes)"
                ),
            )

        try:
            with open(real_path, "r", encoding=encoding) as f:
                content = f.read()
        except UnicodeDecodeError as e:
            return ToolResult(success=False, error=f"Encoding error: {e}")
        except OSError as e:
            return ToolResult(success=False, error=f"Failed to read file: {e}")

        return ToolResult(
            success=True,
            data={"content": content, "path": real_path, "size": file_size},
        )
