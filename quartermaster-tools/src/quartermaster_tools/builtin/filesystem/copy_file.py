"""
CopyFileTool: Copy a file or directory with path validation.
"""

from __future__ import annotations

import os
import shutil
from typing import Any

from quartermaster_tools.base import AbstractTool
from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolResult

from ._security import resolve_base_dir, validate_path


class CopyFileTool(AbstractTool):
    """Copy a file or directory from source to destination."""

    def __init__(self, allowed_base_dir: str | None = None) -> None:
        """Initialise the CopyFileTool.

        Args:
            allowed_base_dir: If set, both source and destination must be under this directory.
        """
        self._allowed_base_dir = resolve_base_dir(allowed_base_dir)

    def name(self) -> str:
        """Return the tool name."""
        return "copy_file"

    def version(self) -> str:
        """Return the tool version."""
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        """Return parameter definitions."""
        return [
            ToolParameter(name="source", description="Source path.", type="string", required=True),
            ToolParameter(name="destination", description="Destination path.", type="string", required=True),
        ]

    def info(self) -> ToolDescriptor:
        """Return tool metadata."""
        return ToolDescriptor(
            name=self.name(),
            short_description="Copy a file or directory.",
            long_description="Copies a file or directory from source to destination. Both paths are validated for security.",
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        """Copy a file or directory.

        Args:
            source: Path to copy from.
            destination: Path to copy to.

        Returns:
            ToolResult indicating success or failure.
        """
        source: str = kwargs.get("source", "")
        destination: str = kwargs.get("destination", "")

        if not source:
            return ToolResult(success=False, error="Parameter 'source' is required")
        if not destination:
            return ToolResult(success=False, error="Parameter 'destination' is required")

        error, real_source = validate_path(source, self._allowed_base_dir)
        if error:
            return ToolResult(success=False, error=error)

        error, real_dest = validate_path(destination, self._allowed_base_dir)
        if error:
            return ToolResult(success=False, error=error)

        if not os.path.exists(real_source):
            return ToolResult(success=False, error=f"Source not found: {source}")

        try:
            if os.path.isdir(real_source):
                shutil.copytree(real_source, real_dest)
            else:
                shutil.copy2(real_source, real_dest)
        except OSError as e:
            return ToolResult(success=False, error=f"Copy failed: {e}")

        return ToolResult(
            success=True,
            data={"source": source, "destination": destination},
        )
