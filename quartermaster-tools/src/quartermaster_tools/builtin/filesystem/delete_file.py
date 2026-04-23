"""
DeleteFileTool: Delete a file or directory with safety confirmation.
"""

from __future__ import annotations

import os
import shutil

from quartermaster_tools.decorator import tool

from ._security import validate_path


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
