"""
Trace and performance profiling tools for execution observability.

trace creates trace spans for debugging and distributed tracing.
performance_profile records execution timing and success/failure metrics.
"""

from __future__ import annotations

import statistics
import uuid
from datetime import datetime, timezone
from typing import Any

from quartermaster_tools.decorator import tool

_TRACE_STORE: list[dict[str, Any]] = []
_PROFILE_STORE: list[dict[str, Any]] = []


# ── trace tool ──────────────────────────────────────────────────────────


@tool()
def trace(name: str, attributes: dict = None, parent_span_id: str = None) -> dict:
    """Create a trace span for observability.

    Creates a trace span with a unique ID, name, timestamp,
    and optional attributes and parent span reference.
    Spans are accumulated in-memory for later retrieval.

    Args:
        name: Name of the trace span.
        attributes: Optional key-value attributes for the span.
        parent_span_id: Optional parent span ID to create a hierarchy.
    """
    if not name:
        raise ValueError("Parameter 'name' is required")

    attributes = attributes or {}

    span_id = str(uuid.uuid4())
    start_time = datetime.now(timezone.utc).isoformat()

    span: dict[str, Any] = {
        "span_id": span_id,
        "name": name,
        "start_time": start_time,
        "attributes": attributes,
        "parent_span_id": parent_span_id,
    }
    _TRACE_STORE.append(span)

    return {
        "span_id": span_id,
        "name": name,
        "start_time": start_time,
        "parent_span_id": parent_span_id,
    }


def get_spans() -> list[dict[str, Any]]:
    """Return all accumulated trace spans."""
    return list(_TRACE_STORE)


def clear_traces() -> None:
    """Clear all accumulated trace spans."""
    _TRACE_STORE.clear()


# Attach class-method-like helpers to the FunctionTool instance
trace.get_spans = get_spans  # type: ignore[attr-defined]
trace.clear = clear_traces  # type: ignore[attr-defined]


# ── performance_profile tool ──────────────────────────────────────────


@tool()
def performance_profile(
    tool_name: str,
    duration_ms: float = None,
    success: bool = None,
    metadata: dict = None,
) -> dict:
    """Record tool execution performance profile.

    Records execution duration and success/failure for a tool,
    enabling performance analysis with summary statistics
    including avg, min, max, p95, and error rates.

    Args:
        tool_name: Name of the tool being profiled.
        duration_ms: Execution duration in milliseconds.
        success: Whether the tool execution succeeded.
        metadata: Optional metadata about the execution.
    """
    if not tool_name:
        raise ValueError("Parameter 'tool_name' is required")

    if duration_ms is None:
        raise ValueError("Parameter 'duration_ms' is required")

    if success is None:
        raise ValueError("Parameter 'success' is required")

    metadata = metadata or {}
    timestamp = datetime.now(timezone.utc).isoformat()

    profile: dict[str, Any] = {
        "tool_name": tool_name,
        "duration_ms": float(duration_ms),
        "success": bool(success),
        "metadata": metadata,
        "timestamp": timestamp,
    }
    _PROFILE_STORE.append(profile)

    return {
        "tool_name": tool_name,
        "duration_ms": float(duration_ms),
        "recorded": True,
    }


def get_profiles() -> list[dict[str, Any]]:
    """Return all recorded performance profiles."""
    return list(_PROFILE_STORE)


def get_profile_summary(tool_name: str) -> dict[str, Any]:
    """Return summary statistics for a specific tool.

    Returns avg, min, max, p95, count, and error_rate.
    """
    entries = [p for p in _PROFILE_STORE if p["tool_name"] == tool_name]
    if not entries:
        return {}

    durations = [e["duration_ms"] for e in entries]
    count = len(entries)
    error_count = sum(1 for e in entries if not e["success"])

    sorted_durations = sorted(durations)
    # p95: index at 95th percentile
    p95_idx = int(0.95 * (count - 1))
    p95 = sorted_durations[p95_idx]

    return {
        "tool_name": tool_name,
        "count": count,
        "avg": statistics.mean(durations),
        "min": min(durations),
        "max": max(durations),
        "p95": p95,
        "error_rate": error_count / count,
    }


def clear_profiles() -> None:
    """Clear all recorded profiles."""
    _PROFILE_STORE.clear()


# Attach class-method-like helpers to the FunctionTool instance
performance_profile.get_profiles = get_profiles  # type: ignore[attr-defined]
performance_profile.get_summary = get_profile_summary  # type: ignore[attr-defined]
performance_profile.clear = clear_profiles  # type: ignore[attr-defined]
