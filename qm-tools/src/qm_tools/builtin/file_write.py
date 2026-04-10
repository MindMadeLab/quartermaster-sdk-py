"""
WriteFileTool: Write content to a file with size limits.

Enforces a maximum content size to prevent writing excessively large files.
Optionally restricts writes to an allowed base directory.
"""

from __future__ import annotations

import os
from typing import Any

from qm_tools.base import AbstractTool
from qm_tools.types import ToolDescriptor, ToolParameter, ToolResult

# Default maximum content size: 10 MB
DEFAULT_MAX_CONTENT_SIZE = 10 * 1024 * 1024

# Allowed encodings
_ALLOWED_ENCODINGS = frozenset({
    "utf-8", "utf-16", "utf-32", "ascii", "latin-1", "iso-8859-1",
})

# Paths that are never allowed to be written to
_BLOCKED_PREFIXES = (
    "/etc/",
    "/private/etc/",
    "/proc/",
    "/sys/",
    "/dev/",
    "/boot/",
    "/sbin/",
    "/usr/bin/",
    "/usr/sbin/",
    "/var/run/secrets/",
)


class WriteFileTool(AbstractTool):
    """Write text content to a file.

    Security features:
    - Blocks writes to sensitive system directories
    - Enforces a configurable maximum content size
    - Checks cumulative file size in append mode
    - Optionally restricts writes to an allowed base directory
    - Validates encoding against an allowlist
    """

    def __init__(
        self,
        max_content_size: int = DEFAULT_MAX_CONTENT_SIZE,
        allowed_base_dir: str | None = None,
        create_dirs: bool = False,
    ) -> None:
        """Initialise the WriteFileTool.

        Args:
            max_content_size: Maximum content size in bytes that can be written.
            allowed_base_dir: If set, only files under this directory may be written.
            create_dirs: Whether to create parent directories automatically.

        Raises:
            ValueError: If allowed_base_dir is set but is not a directory.
        """
        self._max_content_size = max_content_size
        if allowed_base_dir is not None:
            resolved = os.path.realpath(allowed_base_dir)
            if os.path.exists(resolved) and not os.path.isdir(resolved):
                raise ValueError(f"allowed_base_dir must be a directory: {allowed_base_dir}")
            self._allowed_base_dir: str | None = resolved
        else:
            self._allowed_base_dir = None
        self._create_dirs = create_dirs

    def name(self) -> str:
        """Return the tool name."""
        return "write_file"

    def version(self) -> str:
        """Return the tool version."""
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        """Return parameter definitions for the tool."""
        return [
            ToolParameter(
                name="path",
                description="Path to the file to write.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="content",
                description="Text content to write to the file.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="encoding",
                description="Text encoding to use when writing the file.",
                type="string",
                required=False,
                default="utf-8",
            ),
            ToolParameter(
                name="append",
                description="If true, append to the file instead of overwriting.",
                type="boolean",
                required=False,
                default=False,
            ),
        ]

    def info(self) -> ToolDescriptor:
        """Return metadata describing this tool."""
        return ToolDescriptor(
            name=self.name(),
            short_description="Write content to a file.",
            long_description=(
                "Writes text content to a file at the given path. "
                "Supports overwrite and append modes. Includes content "
                "size limits and optional base-directory restriction."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def _validate_path(self, path: str) -> tuple[str | None, str]:
        """Validate the file path.

        Returns:
            Tuple of (error_message_or_None, resolved_real_path).
        """
        real_path = os.path.realpath(path)

        for prefix in _BLOCKED_PREFIXES:
            if real_path.startswith(prefix):
                return f"Access denied: writing to '{prefix}' is not allowed", real_path

        if self._allowed_base_dir is not None:
            if not real_path.startswith(self._allowed_base_dir + os.sep) and real_path != self._allowed_base_dir:
                return (
                    f"Access denied: path must be under '{self._allowed_base_dir}'",
                    real_path,
                )

        return None, real_path

    def run(self, **kwargs: Any) -> ToolResult:
        """Write content to the specified file.

        Args:
            path: The file path to write to.
            content: The text content to write.
            encoding: Text encoding (default utf-8).
            append: If True, append instead of overwrite (default False).

        Returns:
            ToolResult indicating success or failure.
        """
        path: str = kwargs.get("path", "")
        content: str = kwargs.get("content", "")
        encoding: str = kwargs.get("encoding", "utf-8")
        append: bool = kwargs.get("append", False)

        if not path:
            return ToolResult(success=False, error="Parameter 'path' is required")

        # Validate encoding
        if encoding.lower().replace("-", "") not in {
            e.lower().replace("-", "") for e in _ALLOWED_ENCODINGS
        }:
            return ToolResult(
                success=False,
                error=f"Unsupported encoding: {encoding!r}. Allowed: {sorted(_ALLOWED_ENCODINGS)}",
            )

        # Check content size
        content_bytes = len(content.encode(encoding, errors="replace"))
        if content_bytes > self._max_content_size:
            return ToolResult(
                success=False,
                error=(
                    f"Content too large: {content_bytes} bytes "
                    f"(limit: {self._max_content_size} bytes)"
                ),
            )

        # Validate path security — get resolved path once
        error, real_path = self._validate_path(path)
        if error:
            return ToolResult(success=False, error=error)

        # In append mode, check cumulative file size
        if append and os.path.exists(real_path):
            try:
                existing_size = os.path.getsize(real_path)
                if existing_size + content_bytes > self._max_content_size:
                    return ToolResult(
                        success=False,
                        error=(
                            f"Cumulative file size would exceed limit: "
                            f"{existing_size} existing + {content_bytes} new = "
                            f"{existing_size + content_bytes} bytes "
                            f"(limit: {self._max_content_size} bytes)"
                        ),
                    )
            except OSError:
                pass  # File may not exist yet, proceed

        # Create parent directories if needed
        parent_dir = os.path.dirname(real_path)
        if self._create_dirs and not os.path.exists(parent_dir):
            try:
                os.makedirs(parent_dir, exist_ok=True)
            except OSError as e:
                return ToolResult(
                    success=False, error=f"Failed to create directories: {e}"
                )

        mode = "a" if append else "w"
        try:
            with open(real_path, mode, encoding=encoding) as f:
                f.write(content)
        except OSError as e:
            return ToolResult(success=False, error=f"Failed to write file: {e}")

        return ToolResult(
            success=True,
            data={
                "path": path,
                "bytes_written": content_bytes,
                "mode": "append" if append else "overwrite",
            },
        )
