"""
FileInfoTool: Return metadata about a file or directory.
"""

from __future__ import annotations

import mimetypes
import os
import stat

from quartermaster_tools.decorator import tool

from ._security import validate_path


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


# Backward-compatible alias
FileInfoTool = file_info
