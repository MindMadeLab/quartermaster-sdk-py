"""
Shared security utilities for filesystem tools.

Centralises path validation, blocked-prefix checking, and base-directory
sandboxing so every tool enforces the same rules.
"""

from __future__ import annotations

import os

# Paths that are never allowed to be accessed
BLOCKED_PREFIXES: tuple[str, ...] = (
    "/etc/",
    "/private/etc/",
    "/proc/",
    "/sys/",
    "/dev/",
    "/boot/",
    "/var/run/secrets/",
)


def validate_path(
    path: str,
    allowed_base_dir: str | None = None,
) -> tuple[str | None, str]:
    """Validate a filesystem path for security.

    Resolves symlinks, checks against blocked prefixes, and optionally
    enforces a base-directory sandbox.

    Args:
        path: The raw path to validate.
        allowed_base_dir: If set, the resolved path must be under this directory.

    Returns:
        Tuple of (error_message_or_None, resolved_real_path).
    """
    real_path = os.path.realpath(path)

    for prefix in BLOCKED_PREFIXES:
        if real_path.startswith(prefix):
            return f"Access denied: path under '{prefix}' is not allowed", real_path

    if allowed_base_dir is not None:
        base = os.path.realpath(allowed_base_dir)
        if not real_path.startswith(base + os.sep) and real_path != base:
            return f"Access denied: path must be under '{base}'", real_path

    return None, real_path


def resolve_base_dir(allowed_base_dir: str | None) -> str | None:
    """Resolve and validate an allowed_base_dir value at init time.

    Args:
        allowed_base_dir: Directory to restrict operations to, or None.

    Returns:
        The resolved path, or None.

    Raises:
        ValueError: If the path exists but is not a directory.
    """
    if allowed_base_dir is None:
        return None
    resolved = os.path.realpath(allowed_base_dir)
    if os.path.exists(resolved) and not os.path.isdir(resolved):
        raise ValueError(f"allowed_base_dir must be a directory: {allowed_base_dir}")
    return resolved
