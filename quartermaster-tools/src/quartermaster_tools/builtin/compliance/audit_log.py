"""
AuditLogTool and ReadAuditLogTool: Tamper-evident audit trail using JSON Lines.

Each entry includes a SHA-256 hash of the previous entry to form a chain,
making the audit log tamper-evident.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any

from quartermaster_tools.base import AbstractTool
from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolResult


def _hash_entry(entry_json: str) -> str:
    """Compute SHA-256 hash of an entry's JSON string."""
    return hashlib.sha256(entry_json.encode("utf-8")).hexdigest()


class AuditLogTool(AbstractTool):
    """Append entries to a tamper-evident audit trail."""

    def name(self) -> str:
        return "audit_log"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                description="Action being logged.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="actor",
                description="Who performed the action.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="system_id",
                description="Identifier of the AI system.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="details",
                description="Optional additional details.",
                type="object",
                required=False,
            ),
            ToolParameter(
                name="log_path",
                description="Path to the audit log file.",
                type="string",
                required=False,
                default="audit_log.jsonl",
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Append to tamper-evident audit trail.",
            long_description=(
                "Appends a new entry to a JSON Lines audit log file. "
                "Each entry contains a SHA-256 hash of the previous entry "
                "to form a tamper-evident chain."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        action: str = kwargs.get("action", "")
        actor: str = kwargs.get("actor", "")
        system_id: str = kwargs.get("system_id", "")

        if not action:
            return ToolResult(success=False, error="Parameter 'action' is required")
        if not actor:
            return ToolResult(success=False, error="Parameter 'actor' is required")
        if not system_id:
            return ToolResult(success=False, error="Parameter 'system_id' is required")

        details: dict[str, Any] = kwargs.get("details") or {}
        log_path: str = kwargs.get("log_path", "audit_log.jsonl")

        # Read last entry to get previous hash
        previous_hash = "0" * 64  # Genesis hash
        entry_id = 0

        if os.path.exists(log_path):
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                if lines:
                    last_line = lines[-1]
                    previous_hash = _hash_entry(last_line)
                    entry_id = len(lines)
            except OSError as e:
                return ToolResult(success=False, error=f"Failed to read log: {e}")

        timestamp = datetime.now(timezone.utc).isoformat()
        entry = {
            "entry_id": entry_id,
            "timestamp": timestamp,
            "action": action,
            "actor": actor,
            "system_id": system_id,
            "details": details,
            "previous_hash": previous_hash,
        }

        entry_json = json.dumps(entry, separators=(",", ":"))

        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(entry_json + "\n")
        except OSError as e:
            return ToolResult(success=False, error=f"Failed to write log: {e}")

        return ToolResult(
            success=True,
            data={
                "logged": True,
                "entry_id": entry_id,
                "timestamp": timestamp,
            },
        )


class ReadAuditLogTool(AbstractTool):
    """Query and verify the audit trail."""

    def name(self) -> str:
        return "read_audit_log"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="system_id",
                description="Filter entries by system ID.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="log_path",
                description="Path to the audit log file.",
                type="string",
                required=False,
                default="audit_log.jsonl",
            ),
            ToolParameter(
                name="date_from",
                description="Filter entries from this ISO 8601 date.",
                type="string",
                required=False,
            ),
            ToolParameter(
                name="date_to",
                description="Filter entries up to this ISO 8601 date.",
                type="string",
                required=False,
            ),
            ToolParameter(
                name="action_filter",
                description="Filter entries by action type.",
                type="string",
                required=False,
            ),
            ToolParameter(
                name="verify_integrity",
                description="Verify the hash chain integrity.",
                type="boolean",
                required=False,
                default=False,
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Query and verify audit trail.",
            long_description=(
                "Reads entries from a JSON Lines audit log, with optional "
                "filtering by system ID, date range, and action type. "
                "Can verify the hash chain integrity to detect tampering."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        system_id: str = kwargs.get("system_id", "")
        if not system_id:
            return ToolResult(
                success=False, error="Parameter 'system_id' is required"
            )

        log_path: str = kwargs.get("log_path", "audit_log.jsonl")
        date_from: str | None = kwargs.get("date_from")
        date_to: str | None = kwargs.get("date_to")
        action_filter: str | None = kwargs.get("action_filter")
        verify_integrity: bool = kwargs.get("verify_integrity", False)

        if not os.path.exists(log_path):
            return ToolResult(
                success=True,
                data={"entries": [], "count": 0, "integrity_valid": True},
            )

        try:
            with open(log_path, "r", encoding="utf-8") as f:
                raw_lines = [line.strip() for line in f if line.strip()]
        except OSError as e:
            return ToolResult(success=False, error=f"Failed to read log: {e}")

        # Parse all entries
        all_entries: list[dict[str, Any]] = []
        for line in raw_lines:
            try:
                all_entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        # Verify integrity if requested
        integrity_valid = True
        integrity_breaks: list[int] = []
        if verify_integrity:
            expected_hash = "0" * 64
            for i, entry in enumerate(all_entries):
                if entry.get("previous_hash") != expected_hash:
                    integrity_valid = False
                    integrity_breaks.append(i)
                # Hash the raw line for next comparison
                if i < len(raw_lines):
                    expected_hash = _hash_entry(raw_lines[i])

        # Filter by system_id
        entries = [e for e in all_entries if e.get("system_id") == system_id]

        # Filter by date range
        if date_from:
            entries = [e for e in entries if e.get("timestamp", "") >= date_from]
        if date_to:
            entries = [e for e in entries if e.get("timestamp", "") <= date_to]

        # Filter by action
        if action_filter:
            entries = [e for e in entries if e.get("action") == action_filter]

        result_data: dict[str, Any] = {
            "entries": entries,
            "count": len(entries),
            "integrity_valid": integrity_valid,
        }
        if integrity_breaks:
            result_data["integrity_breaks_at"] = integrity_breaks

        return ToolResult(success=True, data=result_data)
