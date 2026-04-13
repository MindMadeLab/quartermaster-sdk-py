"""
CreateDirectoryTool: Create a directory with optional parent creation.
"""

from __future__ import annotations

import os
from typing import Any

from quartermaster_tools.decorator import tool

from ._security import resolve_base_dir, validate_path


@tool()
def create_directory(path: str, parents: bool = True) -> dict:
    """Create a directory.

    Args:
        path: Path of the directory to create.
        parents: Create parent directories as needed.
    """
    if not path:
        return {"error": "Parameter 'path' is required"}

    error, real_path = validate_path(path, None)
    if error:
        return {"error": error}

    try:
        if parents:
            os.makedirs(real_path, exist_ok=True)
        else:
            os.mkdir(real_path)
    except FileExistsError:
        return {"error": f"Directory already exists: {path}"}
    except OSError as e:
        return {"error": f"Failed to create directory: {e}"}

    return {"path": path, "created": True}


# Backward-compatible class wrapper supporting allowed_base_dir constructor arg
class CreateDirectoryTool:
    """Create a directory, optionally creating parent directories.

    Wraps the create_directory function tool, adding optional allowed_base_dir
    restriction for backward compatibility.
    """

    def __init__(self, allowed_base_dir: str | None = None) -> None:
        self._allowed_base_dir = resolve_base_dir(allowed_base_dir)
        self._tool = create_directory

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
