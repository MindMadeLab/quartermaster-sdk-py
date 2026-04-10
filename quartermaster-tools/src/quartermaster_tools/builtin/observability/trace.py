"""
Trace and performance profiling tools for execution observability.

TraceTool creates trace spans for debugging and distributed tracing.
PerformanceProfileTool records execution timing and success/failure metrics.
"""

from __future__ import annotations

import statistics
import uuid
from datetime import datetime, timezone
from typing import Any

from quartermaster_tools.base import AbstractTool
from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolResult

_TRACE_STORE: list[dict[str, Any]] = []
_PROFILE_STORE: list[dict[str, Any]] = []


class TraceTool(AbstractTool):
    """Add trace spans to execution for debugging and observability."""

    def name(self) -> str:
        return "trace"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="name",
                description="Name of the trace span.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="attributes",
                description="Optional key-value attributes for the span.",
                type="object",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="parent_span_id",
                description="Optional parent span ID to create a hierarchy.",
                type="string",
                required=False,
                default=None,
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Create a trace span for observability.",
            long_description=(
                "Creates a trace span with a unique ID, name, timestamp, "
                "and optional attributes and parent span reference. "
                "Spans are accumulated in-memory for later retrieval."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        span_name: str = kwargs.get("name", "")
        if not span_name:
            return ToolResult(success=False, error="Parameter 'name' is required")

        attributes: dict[str, Any] = kwargs.get("attributes") or {}
        parent_span_id: str | None = kwargs.get("parent_span_id")

        span_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc).isoformat()

        span: dict[str, Any] = {
            "span_id": span_id,
            "name": span_name,
            "start_time": start_time,
            "attributes": attributes,
            "parent_span_id": parent_span_id,
        }
        _TRACE_STORE.append(span)

        data: dict[str, Any] = {
            "span_id": span_id,
            "name": span_name,
            "start_time": start_time,
            "parent_span_id": parent_span_id,
        }
        return ToolResult(success=True, data=data)

    @classmethod
    def get_spans(cls) -> list[dict[str, Any]]:
        """Return all accumulated trace spans."""
        return list(_TRACE_STORE)

    @classmethod
    def clear(cls) -> None:
        """Clear all accumulated trace spans."""
        _TRACE_STORE.clear()


class PerformanceProfileTool(AbstractTool):
    """Profile and benchmark tool execution time."""

    def name(self) -> str:
        return "performance_profile"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="tool_name",
                description="Name of the tool being profiled.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="duration_ms",
                description="Execution duration in milliseconds.",
                type="number",
                required=True,
            ),
            ToolParameter(
                name="success",
                description="Whether the tool execution succeeded.",
                type="boolean",
                required=True,
            ),
            ToolParameter(
                name="metadata",
                description="Optional metadata about the execution.",
                type="object",
                required=False,
                default=None,
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Record tool execution performance profile.",
            long_description=(
                "Records execution duration and success/failure for a tool, "
                "enabling performance analysis with summary statistics "
                "including avg, min, max, p95, and error rates."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        tool_name: str = kwargs.get("tool_name", "")
        if not tool_name:
            return ToolResult(success=False, error="Parameter 'tool_name' is required")

        duration_ms = kwargs.get("duration_ms")
        if duration_ms is None:
            return ToolResult(success=False, error="Parameter 'duration_ms' is required")

        success = kwargs.get("success")
        if success is None:
            return ToolResult(success=False, error="Parameter 'success' is required")

        metadata: dict[str, Any] = kwargs.get("metadata") or {}
        timestamp = datetime.now(timezone.utc).isoformat()

        profile: dict[str, Any] = {
            "tool_name": tool_name,
            "duration_ms": float(duration_ms),
            "success": bool(success),
            "metadata": metadata,
            "timestamp": timestamp,
        }
        _PROFILE_STORE.append(profile)

        return ToolResult(
            success=True,
            data={
                "tool_name": tool_name,
                "duration_ms": float(duration_ms),
                "recorded": True,
            },
        )

    @classmethod
    def get_profiles(cls) -> list[dict[str, Any]]:
        """Return all recorded performance profiles."""
        return list(_PROFILE_STORE)

    @classmethod
    def get_summary(cls, tool_name: str) -> dict[str, Any]:
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

    @classmethod
    def clear(cls) -> None:
        """Clear all recorded profiles."""
        _PROFILE_STORE.clear()
