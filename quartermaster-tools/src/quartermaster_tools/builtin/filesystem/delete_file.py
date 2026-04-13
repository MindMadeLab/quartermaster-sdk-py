"""
DeleteFileTool: Delete a file or directory with safety confirmation.
"""

from __future__ import annotations

import os
import shutil
from typing import Any

from quartermaster_tools.decorator import tool

from ._security import resolve_base_dir, validate_path


@tool()
def delete_file(path: str, confirm: bool) -> dict:
    """Delete a file or directory (requires confirmation).

    Args:
        path: Path to the file or directory to delete.
        confirm: Must be True to confirm deletion.
    """
    if not path:
        return {"error": "Parameter 'path' is required"}

    if not confirm:
        return {"error": "Deletion not confirmed: set confirm=True to proceed"}

    error, real_path = validate_path(path, None)
    if error:
        return {"error": error}

    if not os.path.exists(real_path):
        return {"error": f"Path not found: {path}"}

    try:
        if os.path.isdir(real_path):
            shutil.rmtree(real_path)
        else:
            os.remove(real_path)
    except OSError as e:
        return {"error": f"Delete failed: {e}"}

    return {"path": path, "deleted": True}


# Backward-compatible class wrapper supporting allowed_base_dir constructor arg
class DeleteFileTool:
    """Delete a file or directory.

    Wraps the delete_file function tool, adding optional allowed_base_dir
    restriction for backward compatibility.
    """

    def __init__(self, allowed_base_dir: str | None = None) -> None:
        self._allowed_base_dir = resolve_base_dir(allowed_base_dir)
        self._tool = delete_file

    def name(self) -> str:
        return self._tool.name()

    def version(self) -> str:
        return self._tool.version()

    def parameters(self):
        return self._tool.parameters()

    def info(self):
        return self._tool.info()

    def run(self, **kwargs: Any):
        path = kwargs.get("path", "")
        if path:
            error, _ = validate_path(path, self._allowed_base_dir)
            if error:
                from quartermaster_tools.types import ToolResult
                return ToolResult(success=False, error=error)
        return self._tool.run(**kwargs)
