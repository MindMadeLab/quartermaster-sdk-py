"""
Custom metrics recording tool supporting counters, gauges, and histograms.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timezone
from typing import Any

from quartermaster_tools.decorator import tool

_METRICS_STORE: list[dict[str, Any]] = []

_VALID_METRIC_TYPES = {"counter", "gauge", "histogram"}


@tool()
def metric(
    name: str,
    value: float = None,
    unit: str = None,
    tags: dict = None,
    metric_type: str = "gauge",
) -> dict:
    """Record a custom metric.

    Records a metric value with support for counter (accumulate),
    gauge (overwrite last), and histogram (store all values) types.
    Metrics are stored in-memory for retrieval and aggregation.

    Args:
        name: Metric name.
        value: Metric value.
        unit: Optional unit (e.g. 'ms', 'bytes').
        tags: Optional key-value tags for the metric.
        metric_type: Type of metric: counter, gauge, or histogram.
    """
    if not name:
        raise ValueError("Parameter 'name' is required")

    if value is None:
        raise ValueError("Parameter 'value' is required")

    value = float(value)
    tags = tags or {}

    if metric_type not in _VALID_METRIC_TYPES:
        raise ValueError(
            f"Invalid metric_type '{metric_type}'. Must be one of: {', '.join(sorted(_VALID_METRIC_TYPES))}"
        )

    timestamp = datetime.now(timezone.utc).isoformat()

    if metric_type == "counter":
        # Accumulate: find existing counter entry and add
        existing = None
        for entry in _METRICS_STORE:
            if entry["name"] == name and entry["type"] == "counter":
                existing = entry
                break
        if existing is not None:
            existing["value"] += value
            existing["timestamp"] = timestamp
        else:
            _METRICS_STORE.append(
                {
                    "name": name,
                    "value": value,
                    "unit": unit,
                    "tags": tags,
                    "type": "counter",
                    "timestamp": timestamp,
                }
            )
    elif metric_type == "gauge":
        # Overwrite: find existing gauge and replace value
        existing = None
        for entry in _METRICS_STORE:
            if entry["name"] == name and entry["type"] == "gauge":
                existing = entry
                break
        if existing is not None:
            existing["value"] = value
            existing["timestamp"] = timestamp
        else:
            _METRICS_STORE.append(
                {
                    "name": name,
                    "value": value,
                    "unit": unit,
                    "tags": tags,
                    "type": "gauge",
                    "timestamp": timestamp,
                }
            )
    else:
        # Histogram: always append
        _METRICS_STORE.append(
            {
                "name": name,
                "value": value,
                "unit": unit,
                "tags": tags,
                "type": "histogram",
                "timestamp": timestamp,
            }
        )

    return {
        "recorded": True,
        "name": name,
        "value": value,
        "type": metric_type,
    }


def get_metrics() -> list[dict[str, Any]]:
    """Return all stored metrics."""
    return list(_METRICS_STORE)


def get_metric_summary(name: str) -> dict[str, Any]:
    """Return min/max/avg/count for histogram metrics with the given name."""
    values = [e["value"] for e in _METRICS_STORE if e["name"] == name and e["type"] == "histogram"]
    if not values:
        return {}
    return {
        "name": name,
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "avg": statistics.mean(values),
    }


def clear_metrics() -> None:
    """Clear all stored metrics."""
    _METRICS_STORE.clear()


# Attach class-method-like helpers to the FunctionTool instance
metric.get_metrics = get_metrics  # type: ignore[attr-defined]
metric.get_summary = get_metric_summary  # type: ignore[attr-defined]
metric.clear = clear_metrics  # type: ignore[attr-defined]

# Backward-compatible alias
MetricTool = metric
