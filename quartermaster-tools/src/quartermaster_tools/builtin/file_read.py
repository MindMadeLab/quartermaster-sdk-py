"""
ReadFileTool: Read file content from a path with security validation.

Validates paths to prevent directory traversal and enforces size limits
to avoid reading excessively large files into memory.
"""

from __future__ import annotations

import os
from typing import Any

from quartermaster_tools.decorator import tool

# Default maximum file size: 10 MB
DEFAULT_MAX_FILE_SIZE = 10 * 1024 * 1024

# Allowed encodings
_ALLOWED_ENCODINGS = frozenset({
    "utf-8", "utf-16", "utf-32", "ascii", "latin-1", "iso-8859-1",
})

# Paths that are never allowed to be read
_BLOCKED_PREFIXES = (
    "/etc/",
    "/private/etc/",
    "/proc/",
    "/sys/",
    "/dev/",
    "/boot/",
    "/var/run/secrets/",
)


def _validate_read_path(path: str, allowed_base_dir: str | None = None) -> tuple[str | None, str]:
    """Validate the file path.

    Returns:
        Tuple of (error_message_or_None, resolved_real_path).
    """
    real_path = os.path.realpath(path)

    for prefix in _BLOCKED_PREFIXES:
        if real_path.startswith(prefix):
            return f"Access denied: reading from '{prefix}' is not allowed", real_path

    if allowed_base_dir is not None:
        if not real_path.startswith(allowed_base_dir + os.sep) and real_path != allowed_base_dir:
            return (
                f"Access denied: path must be under '{allowed_base_dir}'",
                real_path,
            )

    return None, real_path


def _read_file_impl(
    path: str,
    encoding: str = "utf-8",
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
    allowed_base_dir: str | None = None,
) -> dict:
    """Core implementation for reading a file."""
    if not path:
        return {"error": "Parameter 'path' is required"}

    # Validate encoding
    if encoding.lower().replace("-", "") not in {
        e.lower().replace("-", "") for e in _ALLOWED_ENCODINGS
    }:
        return {"error": f"Unsupported encoding: {encoding!r}. Allowed: {sorted(_ALLOWED_ENCODINGS)}"}

    # Validate path security
    error, real_path = _validate_read_path(path, allowed_base_dir)
    if error:
        return {"error": error}

    if not os.path.exists(real_path):
        return {"error": f"File not found: {path}"}

    if not os.path.isfile(real_path):
        return {"error": f"Not a file: {path}"}

    # Check file size before reading
    try:
        file_size = os.path.getsize(real_path)
    except OSError as e:
        return {"error": f"Cannot stat file: {e}"}

    if file_size > max_file_size:
        return {"error": f"File too large: {file_size} bytes (limit: {max_file_size} bytes)"}

    try:
        with open(real_path, "r", encoding=encoding) as f:
            content = f.read()
    except UnicodeDecodeError as e:
        return {"error": f"Encoding error: {e}"}
    except OSError as e:
        return {"error": f"Failed to read file: {e}"}

    return {"content": content, "path": path, "size": file_size}


@tool()
def read_file(path: str, encoding: str = "utf-8") -> dict:
    """Read content from a file.

    Args:
        path: Absolute or relative path to the file to read.
        encoding: Text encoding to use when reading the file.
    """
    return _read_file_impl(path, encoding=encoding)


# Backward-compatible class wrapper supporting constructor args
class ReadFileTool:
    """Read file content from a given path.

    Wraps the read_file function tool, adding optional max_file_size and
    allowed_base_dir restrictions for backward compatibility.
    """

    def __init__(
        self,
        max_file_size: int = DEFAULT_MAX_FILE_SIZE,
        allowed_base_dir: str | None = None,
    ) -> None:
        self._max_file_size = max_file_size
        if allowed_base_dir is not None:
            resolved = os.path.realpath(allowed_base_dir)
            if os.path.exists(resolved) and not os.path.isdir(resolved):
                raise ValueError(f"allowed_base_dir must be a directory: {allowed_base_dir}")
            self._allowed_base_dir: str | None = resolved
        else:
            self._allowed_base_dir = None
        self._tool = read_file

    def name(self) -> str:
        return self._tool.name()

    def version(self) -> str:
        return self._tool.version()

    def parameters(self):
        return self._tool.parameters()

    def info(self):
        info = self._tool.info()
        info.is_local = True
        return info

    def validate_params(self, **kwargs: Any) -> list[str]:
        return self._tool.validate_params(**kwargs)

    def safe_run(self, **kwargs: Any):
        from quartermaster_tools.types import ToolResult
        errors = self.validate_params(**kwargs)
        if errors:
            return ToolResult(success=False, error="; ".join(errors))
        return self.run(**kwargs)

    def run(self, **kwargs: Any):
        from quartermaster_tools.types import ToolResult
        path = kwargs.get("path", "")
        encoding = kwargs.get("encoding", "utf-8")
        result = _read_file_impl(
            path,
            encoding=encoding,
            max_file_size=self._max_file_size,
            allowed_base_dir=self._allowed_base_dir,
        )
        if "error" in result:
            return ToolResult(success=False, error=result["error"])
        return ToolResult(success=True, data=result)
