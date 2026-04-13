"""
CopyFileTool: Copy a file or directory with path validation.
"""

from __future__ import annotations

import os
import shutil
from typing import Any

from quartermaster_tools.decorator import tool

from ._security import resolve_base_dir, validate_path


@tool()
def copy_file(source: str, destination: str) -> dict:
    """Copy a file or directory.

    Args:
        source: Source path.
        destination: Destination path.
    """
    if not source:
        return {"error": "Parameter 'source' is required"}
    if not destination:
        return {"error": "Parameter 'destination' is required"}

    error, real_source = validate_path(source, None)
    if error:
        return {"error": error}

    error, real_dest = validate_path(destination, None)
    if error:
        return {"error": error}

    if not os.path.exists(real_source):
        return {"error": f"Source not found: {source}"}

    try:
        if os.path.isdir(real_source):
            shutil.copytree(real_source, real_dest)
        else:
            shutil.copy2(real_source, real_dest)
    except OSError as e:
        return {"error": f"Copy failed: {e}"}

    return {"source": source, "destination": destination}


# Backward-compatible class wrapper supporting allowed_base_dir constructor arg
class CopyFileTool:
    """Copy a file or directory from source to destination.

    Wraps the copy_file function tool, adding optional allowed_base_dir
    restriction for backward compatibility.
    """

    def __init__(self, allowed_base_dir: str | None = None) -> None:
        self._allowed_base_dir = resolve_base_dir(allowed_base_dir)
        self._tool = copy_file

    def name(self) -> str:
        return self._tool.name()

    def version(self) -> str:
        return self._tool.version()

    def parameters(self):
        return self._tool.parameters()

    def info(self):
        return self._tool.info()

    def run(self, **kwargs: Any):
        from quartermaster_tools.types import ToolResult
        source = kwargs.get("source", "")
        destination = kwargs.get("destination", "")
        if source:
            error, _ = validate_path(source, self._allowed_base_dir)
            if error:
                return ToolResult(success=False, error=error)
        if destination:
            error, _ = validate_path(destination, self._allowed_base_dir)
            if error:
                return ToolResult(success=False, error=error)
        return self._tool.run(**kwargs)
