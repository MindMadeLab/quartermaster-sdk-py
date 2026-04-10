"""
CreateDirectoryTool: Create a directory with optional parent creation.
"""

from __future__ import annotations

import os
from typing import Any

from quartermaster_tools.base import AbstractTool
from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolResult

from ._security import resolve_base_dir, validate_path


class CreateDirectoryTool(AbstractTool):
    """Create a directory, optionally creating parent directories."""

    def __init__(self, allowed_base_dir: str | None = None) -> None:
        """Initialise the CreateDirectoryTool.

        Args:
            allowed_base_dir: If set, only paths under this directory are allowed.
        """
        self._allowed_base_dir = resolve_base_dir(allowed_base_dir)

    def name(self) -> str:
        """Return the tool name."""
        return "create_directory"

    def version(self) -> str:
        """Return the tool version."""
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        """Return parameter definitions."""
        return [
            ToolParameter(name="path", description="Path of the directory to create.", type="string", required=True),
            ToolParameter(name="parents", description="Create parent directories as needed.", type="boolean", default=True),
        ]

    def info(self) -> ToolDescriptor:
        """Return tool metadata."""
        return ToolDescriptor(
            name=self.name(),
            short_description="Create a directory.",
            long_description="Creates a directory at the given path, optionally creating parent directories.",
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        """Create a directory.

        Args:
            path: Directory path to create.
            parents: Create parent directories if True (default True).

        Returns:
            ToolResult indicating success or failure.
        """
        path: str = kwargs.get("path", "")
        parents: bool = kwargs.get("parents", True)

        if not path:
            return ToolResult(success=False, error="Parameter 'path' is required")

        error, real_path = validate_path(path, self._allowed_base_dir)
        if error:
            return ToolResult(success=False, error=error)

        try:
            if parents:
                os.makedirs(real_path, exist_ok=True)
            else:
                os.mkdir(real_path)
        except FileExistsError:
            return ToolResult(success=False, error=f"Directory already exists: {path}")
        except OSError as e:
            return ToolResult(success=False, error=f"Failed to create directory: {e}")

        return ToolResult(
            success=True,
            data={"path": path, "created": True},
        )
