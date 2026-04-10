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

from quartermaster_tools.base import AbstractTool
from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolResult

_LOG_STORE: list[dict[str, Any]] = []

_VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class LogTool(AbstractTool):
    """Structured logging with context metadata."""

    def name(self) -> str:
        return "log"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="level",
                description="Log level.",
                type="string",
                required=True,
                options=[],
            ),
            ToolParameter(
                name="message",
                description="Log message.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="metadata",
                description="Optional structured metadata to attach.",
                type="object",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="log_path",
                description="Optional file path to append the log entry (JSON Lines).",
                type="string",
                required=False,
                default=None,
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Write a structured log entry.",
            long_description=(
                "Creates a structured JSON log entry with timestamp, level, "
                "message, and optional metadata. Stores in-memory and "
                "optionally appends to a file in JSON Lines format."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        level: str = kwargs.get("level", "")
        if not level:
            return ToolResult(success=False, error="Parameter 'level' is required")

        level = level.upper()
        if level not in _VALID_LEVELS:
            return ToolResult(
                success=False,
                error=f"Invalid level '{level}'. Must be one of: {', '.join(sorted(_VALID_LEVELS))}",
            )

        message: str = kwargs.get("message", "")
        if not message:
            return ToolResult(success=False, error="Parameter 'message' is required")

        metadata: dict[str, Any] = kwargs.get("metadata") or {}
        log_path: str | None = kwargs.get("log_path")

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

        return ToolResult(
            success=True,
            data={"logged": True, "level": level, "timestamp": timestamp},
        )

    @classmethod
    def get_logs(cls) -> list[dict[str, Any]]:
        """Return all accumulated log entries."""
        return list(_LOG_STORE)

    @classmethod
    def clear(cls) -> None:
        """Clear all accumulated log entries."""
        _LOG_STORE.clear()
