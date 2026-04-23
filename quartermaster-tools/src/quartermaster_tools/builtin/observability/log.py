"""
Structured logging tool with context metadata.

Writes structured JSON log entries to an in-memory store and optionally
to a file in JSON Lines format.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quartermaster_tools.decorator import tool

_LOG_STORE: list[dict[str, Any]] = []

_VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


@tool()
def log(level: str, message: str, metadata: dict = None, log_path: str = None) -> dict:
    """Write a structured log entry.

    Creates a structured JSON log entry with timestamp, level,
    message, and optional metadata. Stores in-memory and
    optionally appends to a file in JSON Lines format.

    Args:
        level: Log level.
        message: Log message.
        metadata: Optional structured metadata to attach.
        log_path: Optional file path to append the log entry (JSON Lines).
    """
    if not level:
        raise ValueError("Parameter 'level' is required")

    level = level.upper()
    if level not in _VALID_LEVELS:
        raise ValueError(
            f"Invalid level '{level}'. Must be one of: {', '.join(sorted(_VALID_LEVELS))}"
        )

    if not message:
        raise ValueError("Parameter 'message' is required")

    metadata = metadata or {}

    timestamp = datetime.now(timezone.utc).isoformat()

    entry: dict[str, Any] = {
        "timestamp": timestamp,
        "level": level,
        "message": message,
        "metadata": metadata,
    }
    _LOG_STORE.append(entry)

    if log_path:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    return {"logged": True, "level": level, "timestamp": timestamp}


def get_logs() -> list[dict[str, Any]]:
    """Return all accumulated log entries."""
    return list(_LOG_STORE)


def clear() -> None:
    """Clear all accumulated log entries."""
    _LOG_STORE.clear()


# Attach class-method-like helpers to the FunctionTool instance
log.get_logs = get_logs  # type: ignore[attr-defined]
log.clear = clear  # type: ignore[attr-defined]
