"""
FileInfoTool: Return metadata about a file or directory.
"""

from __future__ import annotations

import mimetypes
import os
import stat
from typing import Any

from quartermaster_tools.decorator import tool

from ._security import resolve_base_dir, validate_path


@tool()
def file_info(path: str) -> dict:
    """Get file metadata (size, type, permissions, mime type).

    Args:
        path: Path to the file or directory.
    """
    if not path:
        return {"error": "Parameter 'path' is required"}

    error, real_path = validate_path(path, None)
    if error:
        return {"error": error}

    if not os.path.exists(real_path):
        return {"error": f"Path not found: {path}"}

    try:
        st = os.stat(real_path)
    except OSError as e:
        return {"error": f"Cannot stat path: {e}"}

    if os.path.isfile(real_path):
        file_type = "file"
    elif os.path.isdir(real_path):
        file_type = "directory"
    elif os.path.islink(real_path):
        file_type = "symlink"
    else:
        file_type = "other"

    mime_type, _ = mimetypes.guess_type(real_path)

    return {
        "path": real_path,
        "size": st.st_size,
        "modified": st.st_mtime,
        "created": getattr(st, "st_birthtime", st.st_ctime),
        "type": file_type,
        "permissions": stat.filemode(st.st_mode),
        "mime_type": mime_type or "application/octet-stream",
    }


# Backward-compatible class wrapper supporting allowed_base_dir constructor arg
class FileInfoTool:
    """Return detailed metadata about a file or directory.

    Wraps the file_info function tool, adding optional allowed_base_dir
    restriction for backward compatibility.
    """

    def __init__(self, allowed_base_dir: str | None = None) -> None:
        self._allowed_base_dir = resolve_base_dir(allowed_base_dir)
        self._tool = file_info

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
