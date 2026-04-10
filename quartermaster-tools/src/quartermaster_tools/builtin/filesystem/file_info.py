"""
FileInfoTool: Return metadata about a file or directory.
"""

from __future__ import annotations

import mimetypes
import os
import stat
from typing import Any

from quartermaster_tools.base import AbstractTool
from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolResult

from ._security import resolve_base_dir, validate_path


class FileInfoTool(AbstractTool):
    """Return detailed metadata about a file or directory."""

    def __init__(self, allowed_base_dir: str | None = None) -> None:
        """Initialise the FileInfoTool.

        Args:
            allowed_base_dir: If set, only paths under this directory are allowed.
        """
        self._allowed_base_dir = resolve_base_dir(allowed_base_dir)

    def name(self) -> str:
        """Return the tool name."""
        return "file_info"

    def version(self) -> str:
        """Return the tool version."""
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        """Return parameter definitions."""
        return [
            ToolParameter(name="path", description="Path to the file or directory.", type="string", required=True),
        ]

    def info(self) -> ToolDescriptor:
        """Return tool metadata."""
        return ToolDescriptor(
            name=self.name(),
            short_description="Get file metadata (size, type, permissions, mime type).",
            long_description="Returns size, modification time, creation time, type, permissions, and MIME type for a path.",
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        """Return metadata about a file or directory.

        Args:
            path: Path to inspect.

        Returns:
            ToolResult with size, modified, created, type, permissions, mime_type.
        """
        path: str = kwargs.get("path", "")

        if not path:
            return ToolResult(success=False, error="Parameter 'path' is required")

        error, real_path = validate_path(path, self._allowed_base_dir)
        if error:
            return ToolResult(success=False, error=error)

        if not os.path.exists(real_path):
            return ToolResult(success=False, error=f"Path not found: {path}")

        try:
            st = os.stat(real_path)
        except OSError as e:
            return ToolResult(success=False, error=f"Cannot stat path: {e}")

        if os.path.isfile(real_path):
            file_type = "file"
        elif os.path.isdir(real_path):
            file_type = "directory"
        elif os.path.islink(real_path):
            file_type = "symlink"
        else:
            file_type = "other"

        mime_type, _ = mimetypes.guess_type(real_path)

        return ToolResult(
            success=True,
            data={
                "path": real_path,
                "size": st.st_size,
                "modified": st.st_mtime,
                "created": getattr(st, "st_birthtime", st.st_ctime),
                "type": file_type,
                "permissions": stat.filemode(st.st_mode),
                "mime_type": mime_type or "application/octet-stream",
            },
        )
