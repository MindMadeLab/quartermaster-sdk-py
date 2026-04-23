"""
MoveFileTool: Move or rename a file or directory with path validation.
"""

from __future__ import annotations

import os
import shutil

from quartermaster_tools.decorator import tool

from ._security import validate_path


@tool()
def move_file(source: str, destination: str) -> dict:
    """Move or rename a file or directory.

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
        shutil.move(real_source, real_dest)
    except OSError as e:
        return {"error": f"Move failed: {e}"}

    return {"source": source, "destination": destination}
