"""
audit_log and read_audit_log: Tamper-evident audit trail using JSON Lines.

Each entry includes a SHA-256 hash of the previous entry to form a chain,
making the audit log tamper-evident.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any

from quartermaster_tools.decorator import tool


def _hash_entry(entry_json: str) -> str:
    """Compute SHA-256 hash of an entry's JSON string."""
    return hashlib.sha256(entry_json.encode("utf-8")).hexdigest()


@tool()
def audit_log(
    action: str,
    actor: str,
    system_id: str,
    details: dict = None,
    log_path: str = "audit_log.jsonl",
) -> dict:
    """Append to tamper-evident audit trail.

    Appends a new entry to a JSON Lines audit log file.
    Each entry contains a SHA-256 hash of the previous entry
    to form a tamper-evident chain.

    Args:
        action: Action being logged.
        actor: Who performed the action.
        system_id: Identifier of the AI system.
        details: Optional additional details.
        log_path: Path to the audit log file.
    """
    if not action:
        raise ValueError("Parameter 'action' is required")
    if not actor:
        raise ValueError("Parameter 'actor' is required")
    if not system_id:
        raise ValueError("Parameter 'system_id' is required")

    details = details or {}

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
            raise OSError(f"Failed to read log: {e}")

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
        raise OSError(f"Failed to write log: {e}")

    return {
        "logged": True,
        "entry_id": entry_id,
        "timestamp": timestamp,
    }


@tool()
def read_audit_log(
    system_id: str,
    log_path: str = "audit_log.jsonl",
    date_from: str = "",
    date_to: str = "",
    action_filter: str = "",
    verify_integrity: bool = False,
) -> dict:
    """Query and verify audit trail.

    Reads entries from a JSON Lines audit log, with optional
    filtering by system ID, date range, and action type.
    Can verify the hash chain integrity to detect tampering.

    Args:
        system_id: Filter entries by system ID.
        log_path: Path to the audit log file.
        date_from: Filter entries from this ISO 8601 date.
        date_to: Filter entries up to this ISO 8601 date.
        action_filter: Filter entries by action type.
        verify_integrity: Verify the hash chain integrity.
    """
    if not system_id:
        raise ValueError("Parameter 'system_id' is required")

    if not os.path.exists(log_path):
        return {"entries": [], "count": 0, "integrity_valid": True}

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            raw_lines = [line.strip() for line in f if line.strip()]
    except OSError as e:
        raise OSError(f"Failed to read log: {e}")

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

    return result_data
