"""
DeleteFileTool: Delete a file or directory with safety confirmation.
"""

from __future__ import annotations

import os
import shutil
from typing import Any

from quartermaster_tools.base import AbstractTool
from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolResult

from ._security import resolve_base_dir, validate_path


class DeleteFileTool(AbstractTool):
    """Delete a file or directory.

    Requires explicit confirm=True to prevent accidental deletion.
    """

    def __init__(self, allowed_base_dir: str | None = None) -> None:
        """Initialise the DeleteFileTool.

        Args:
            allowed_base_dir: If set, only paths under this directory may be deleted.
        """
        self._allowed_base_dir = resolve_base_dir(allowed_base_dir)

    def name(self) -> str:
        """Return the tool name."""
        return "delete_file"

    def version(self) -> str:
        """Return the tool version."""
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        """Return parameter definitions."""
        return [
            ToolParameter(name="path", description="Path to the file or directory to delete.", type="string", required=True),
            ToolParameter(name="confirm", description="Must be True to confirm deletion.", type="boolean", required=True),
        ]

    def info(self) -> ToolDescriptor:
        """Return tool metadata."""
        return ToolDescriptor(
            name=self.name(),
            short_description="Delete a file or directory (requires confirmation).",
            long_description="Deletes a file or directory. The confirm parameter must be True to proceed, preventing accidental deletion.",
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        """Delete a file or directory.

        Args:
            path: Path to delete.
            confirm: Must be True to proceed.

        Returns:
            ToolResult indicating success or failure.
        """
        path: str = kwargs.get("path", "")
        confirm: bool = kwargs.get("confirm", False)

        if not path:
            return ToolResult(success=False, error="Parameter 'path' is required")

        if not confirm:
            return ToolResult(success=False, error="Deletion not confirmed: set confirm=True to proceed")

        error, real_path = validate_path(path, self._allowed_base_dir)
        if error:
            return ToolResult(success=False, error=error)

        if not os.path.exists(real_path):
            return ToolResult(success=False, error=f"Path not found: {path}")

        try:
            if os.path.isdir(real_path):
                shutil.rmtree(real_path)
            else:
                os.remove(real_path)
        except OSError as e:
            return ToolResult(success=False, error=f"Delete failed: {e}")

        return ToolResult(
            success=True,
            data={"path": path, "deleted": True},
        )
