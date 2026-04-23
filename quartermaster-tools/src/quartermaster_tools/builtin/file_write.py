"""
WriteFileTool: Write content to a file with size limits.

Enforces a maximum content size to prevent writing excessively large files.
Optionally restricts writes to an allowed base directory.
"""

from __future__ import annotations

import os

from quartermaster_tools.decorator import tool

# Default maximum content size: 10 MB
DEFAULT_MAX_CONTENT_SIZE = 10 * 1024 * 1024

# Allowed encodings
_ALLOWED_ENCODINGS = frozenset(
    {
        "utf-8",
        "utf-16",
        "utf-32",
        "ascii",
        "latin-1",
        "iso-8859-1",
    }
)

# Paths that are never allowed to be written to
_BLOCKED_PREFIXES = (
    "/etc/",
    "/private/etc/",
    "/proc/",
    "/sys/",
    "/dev/",
    "/boot/",
    "/sbin/",
    "/usr/bin/",
    "/usr/sbin/",
    "/var/run/secrets/",
)


def _validate_write_path(path: str, allowed_base_dir: str | None = None) -> tuple[str | None, str]:
    """Validate the file path.

    Returns:
        Tuple of (error_message_or_None, resolved_real_path).
    """
    real_path = os.path.realpath(path)

    for prefix in _BLOCKED_PREFIXES:
        if real_path.startswith(prefix):
            return f"Access denied: writing to '{prefix}' is not allowed", real_path

    if allowed_base_dir is not None:
        if not real_path.startswith(allowed_base_dir + os.sep) and real_path != allowed_base_dir:
            return (
                f"Access denied: path must be under '{allowed_base_dir}'",
                real_path,
            )

    return None, real_path


def _write_file_impl(
    path: str,
    content: str,
    encoding: str = "utf-8",
    append: bool = False,
    max_content_size: int = DEFAULT_MAX_CONTENT_SIZE,
    allowed_base_dir: str | None = None,
    create_dirs: bool = False,
) -> dict:
    """Core implementation for writing a file."""
    if not path:
        return {"error": "Parameter 'path' is required"}

    # Validate encoding
    if encoding.lower().replace("-", "") not in {
        e.lower().replace("-", "") for e in _ALLOWED_ENCODINGS
    }:
        return {
            "error": f"Unsupported encoding: {encoding!r}. Allowed: {sorted(_ALLOWED_ENCODINGS)}"
        }

    # Check content size
    content_bytes = len(content.encode(encoding, errors="replace"))
    if content_bytes > max_content_size:
        return {
            "error": f"Content too large: {content_bytes} bytes (limit: {max_content_size} bytes)"
        }

    # Validate path security
    error, real_path = _validate_write_path(path, allowed_base_dir)
    if error:
        return {"error": error}

    # In append mode, check cumulative file size
    if append and os.path.exists(real_path):
        try:
            existing_size = os.path.getsize(real_path)
            if existing_size + content_bytes > max_content_size:
                return {
                    "error": (
                        f"Cumulative file size would exceed limit: "
                        f"{existing_size} existing + {content_bytes} new = "
                        f"{existing_size + content_bytes} bytes "
                        f"(limit: {max_content_size} bytes)"
                    )
                }
        except OSError:
            pass  # File may not exist yet, proceed

    # Create parent directories if needed
    parent_dir = os.path.dirname(real_path)
    if create_dirs and not os.path.exists(parent_dir):
        try:
            os.makedirs(parent_dir, exist_ok=True)
        except OSError as e:
            return {"error": f"Failed to create directories: {e}"}

    mode = "a" if append else "w"
    try:
        with open(real_path, mode, encoding=encoding) as f:
            f.write(content)
    except OSError as e:
        return {"error": f"Failed to write file: {e}"}

    return {
        "path": path,
        "bytes_written": content_bytes,
        "mode": "append" if append else "overwrite",
    }


@tool()
def write_file(path: str, content: str, encoding: str = "utf-8", append: bool = False) -> dict:
    """Write content to a file.

    Args:
        path: Path to the file to write.
        content: Text content to write to the file.
        encoding: Text encoding to use when writing the file.
        append: If true, append to the file instead of overwriting.
    """
    return _write_file_impl(path, content, encoding=encoding, append=append)
