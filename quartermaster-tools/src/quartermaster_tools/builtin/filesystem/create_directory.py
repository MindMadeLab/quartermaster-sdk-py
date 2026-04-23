"""
CreateDirectoryTool: Create a directory with optional parent creation.
"""

from __future__ import annotations

import os

from quartermaster_tools.decorator import tool

from ._security import validate_path


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
