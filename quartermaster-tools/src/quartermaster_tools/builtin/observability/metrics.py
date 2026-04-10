"""
Custom metrics recording tool supporting counters, gauges, and histograms.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timezone
from typing import Any

from quartermaster_tools.base import AbstractTool
from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolResult

_METRICS_STORE: list[dict[str, Any]] = []

_VALID_METRIC_TYPES = {"counter", "gauge", "histogram"}


class MetricTool(AbstractTool):
    """Record custom metrics (counters, gauges, histograms)."""

    def name(self) -> str:
        return "metric"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="name",
                description="Metric name.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="value",
                description="Metric value.",
                type="number",
                required=True,
            ),
            ToolParameter(
                name="unit",
                description="Optional unit (e.g. 'ms', 'bytes').",
                type="string",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="tags",
                description="Optional key-value tags for the metric.",
                type="object",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="metric_type",
                description="Type of metric: counter, gauge, or histogram.",
                type="string",
                required=False,
                default="gauge",
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Record a custom metric.",
            long_description=(
                "Records a metric value with support for counter (accumulate), "
                "gauge (overwrite last), and histogram (store all values) types. "
                "Metrics are stored in-memory for retrieval and aggregation."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        metric_name: str = kwargs.get("name", "")
        if not metric_name:
            return ToolResult(success=False, error="Parameter 'name' is required")

        value = kwargs.get("value")
        if value is None:
            return ToolResult(success=False, error="Parameter 'value' is required")

        value = float(value)
        unit: str | None = kwargs.get("unit")
        tags: dict[str, Any] = kwargs.get("tags") or {}
        metric_type: str = kwargs.get("metric_type", "gauge")

        if metric_type not in _VALID_METRIC_TYPES:
            return ToolResult(
                success=False,
                error=f"Invalid metric_type '{metric_type}'. Must be one of: {', '.join(sorted(_VALID_METRIC_TYPES))}",
            )

        timestamp = datetime.now(timezone.utc).isoformat()

        if metric_type == "counter":
            # Accumulate: find existing counter entry and add
            existing = None
            for entry in _METRICS_STORE:
                if entry["name"] == metric_name and entry["type"] == "counter":
                    existing = entry
                    break
            if existing is not None:
                existing["value"] += value
                existing["timestamp"] = timestamp
            else:
                _METRICS_STORE.append({
                    "name": metric_name,
                    "value": value,
                    "unit": unit,
                    "tags": tags,
                    "type": "counter",
                    "timestamp": timestamp,
                })
        elif metric_type == "gauge":
            # Overwrite: find existing gauge and replace value
            existing = None
            for entry in _METRICS_STORE:
                if entry["name"] == metric_name and entry["type"] == "gauge":
                    existing = entry
                    break
            if existing is not None:
                existing["value"] = value
                existing["timestamp"] = timestamp
            else:
                _METRICS_STORE.append({
                    "name": metric_name,
                    "value": value,
                    "unit": unit,
                    "tags": tags,
                    "type": "gauge",
                    "timestamp": timestamp,
                })
        else:
            # Histogram: always append
            _METRICS_STORE.append({
                "name": metric_name,
                "value": value,
                "unit": unit,
                "tags": tags,
                "type": "histogram",
                "timestamp": timestamp,
            })

        return ToolResult(
            success=True,
            data={
                "recorded": True,
                "name": metric_name,
                "value": value,
                "type": metric_type,
            },
        )

    @classmethod
    def get_metrics(cls) -> list[dict[str, Any]]:
        """Return all stored metrics."""
        return list(_METRICS_STORE)

    @classmethod
    def get_summary(cls, name: str) -> dict[str, Any]:
        """Return min/max/avg/count for histogram metrics with the given name."""
        values = [
            e["value"]
            for e in _METRICS_STORE
            if e["name"] == name and e["type"] == "histogram"
        ]
        if not values:
            return {}
        return {
            "name": name,
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": statistics.mean(values),
        }

    @classmethod
    def clear(cls) -> None:
        """Clear all stored metrics."""
        _METRICS_STORE.clear()
