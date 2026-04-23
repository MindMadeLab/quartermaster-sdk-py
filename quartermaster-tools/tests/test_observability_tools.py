"""Tests for the observability and tracing tools."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quartermaster_tools.builtin.observability import (
    cost_tracker,
    log,
    metric,
    performance_profile,
    trace,
)


@pytest.fixture(autouse=True)
def _clear_stores():
    """Clear all module-level stores before and after each test."""
    trace.clear()
    log.clear()
    metric.clear()
    cost_tracker.clear()
    performance_profile.clear()
    yield
    trace.clear()
    log.clear()
    metric.clear()
    cost_tracker.clear()
    performance_profile.clear()


# ── trace ──────────────────────────────────────────────────────────


class TestTraceTool:
    def test_create_span(self):
        tool = trace
        result = tool.run(name="my-span")
        assert result.success
        assert result.data["name"] == "my-span"
        assert "span_id" in result.data
        assert "start_time" in result.data

    def test_span_with_attributes(self):
        tool = trace
        result = tool.run(name="op", attributes={"key": "val"})
        assert result.success
        spans = trace.get_spans()
        assert len(spans) == 1
        assert spans[0]["attributes"] == {"key": "val"}

    def test_parent_span(self):
        tool = trace
        parent = tool.run(name="parent")
        child = tool.run(name="child", parent_span_id=parent.data["span_id"])
        assert child.success
        assert child.data["parent_span_id"] == parent.data["span_id"]

    def test_get_spans_returns_all(self):
        tool = trace
        tool.run(name="a")
        tool.run(name="b")
        tool.run(name="c")
        assert len(trace.get_spans()) == 3

    def test_clear(self):
        tool = trace
        tool.run(name="x")
        trace.clear()
        assert len(trace.get_spans()) == 0

    def test_missing_name(self):
        tool = trace
        result = tool.run()
        assert not result.success
        assert "name" in result.error.lower()

    def test_info_descriptor(self):
        tool = trace
        info = tool.info()
        assert info.name == "trace"
        assert info.version == "1.0.0"


# ── log ────────────────────────────────────────────────────────────


class TestLogTool:
    def test_log_info(self):
        tool = log
        result = tool.run(level="INFO", message="hello")
        assert result.success
        assert result.data["logged"] is True
        assert result.data["level"] == "INFO"

    def test_all_levels(self):
        tool = log
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            result = tool.run(level=level, message=f"test {level}")
            assert result.success

    def test_case_insensitive_level(self):
        tool = log
        result = tool.run(level="info", message="lower")
        assert result.success
        assert result.data["level"] == "INFO"

    def test_invalid_level(self):
        tool = log
        result = tool.run(level="TRACE", message="nope")
        assert not result.success

    def test_metadata(self):
        tool = log
        tool.run(level="INFO", message="m", metadata={"k": 1})
        logs = log.get_logs()
        assert logs[0]["metadata"] == {"k": 1}

    def test_file_output(self, tmp_path: Path):
        log_file = tmp_path / "app.log"
        tool = log
        tool.run(level="ERROR", message="fail", log_path=str(log_file))
        tool.run(level="INFO", message="ok", log_path=str(log_file))

        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 2
        entry = json.loads(lines[0])
        assert entry["level"] == "ERROR"
        assert entry["message"] == "fail"

    def test_get_logs_and_clear(self):
        tool = log
        tool.run(level="INFO", message="a")
        tool.run(level="DEBUG", message="b")
        assert len(log.get_logs()) == 2
        log.clear()
        assert len(log.get_logs()) == 0

    def test_missing_message(self):
        tool = log
        result = tool.run(level="INFO")
        assert not result.success


# ── metric ─────────────────────────────────────────────────────────


class TestMetricTool:
    def test_gauge_basic(self):
        tool = metric
        result = tool.run(name="cpu", value=42.5)
        assert result.success
        assert result.data["type"] == "gauge"

    def test_gauge_overwrites(self):
        tool = metric
        tool.run(name="cpu", value=10)
        tool.run(name="cpu", value=90)
        metrics = metric.get_metrics()
        gauges = [m for m in metrics if m["name"] == "cpu"]
        assert len(gauges) == 1
        assert gauges[0]["value"] == 90.0

    def test_counter_accumulates(self):
        tool = metric
        tool.run(name="requests", value=1, metric_type="counter")
        tool.run(name="requests", value=3, metric_type="counter")
        metrics = metric.get_metrics()
        counters = [m for m in metrics if m["name"] == "requests"]
        assert len(counters) == 1
        assert counters[0]["value"] == 4.0

    def test_histogram_stores_all(self):
        tool = metric
        for v in [10, 20, 30]:
            tool.run(name="latency", value=v, metric_type="histogram")
        metrics = metric.get_metrics()
        histograms = [m for m in metrics if m["name"] == "latency"]
        assert len(histograms) == 3

    def test_histogram_summary(self):
        tool = metric
        for v in [10, 20, 30, 40, 50]:
            tool.run(name="latency", value=v, metric_type="histogram")
        summary = metric.get_summary("latency")
        assert summary["count"] == 5
        assert summary["min"] == 10.0
        assert summary["max"] == 50.0
        assert summary["avg"] == 30.0

    def test_summary_empty(self):
        assert metric.get_summary("nonexistent") == {}

    def test_invalid_type(self):
        tool = metric
        result = tool.run(name="x", value=1, metric_type="invalid")
        assert not result.success

    def test_with_unit_and_tags(self):
        tool = metric
        tool.run(name="mem", value=512, unit="bytes", tags={"host": "a"})
        m = metric.get_metrics()[0]
        assert m["unit"] == "bytes"
        assert m["tags"] == {"host": "a"}

    def test_clear(self):
        tool = metric
        tool.run(name="x", value=1)
        metric.clear()
        assert len(metric.get_metrics()) == 0


# ── cost_tracker ───────────────────────────────────────────────────


class TestCostTrackerTool:
    def test_known_model(self):
        tool = cost_tracker
        result = tool.run(model="gpt-4o", input_tokens=1000, output_tokens=500)
        assert result.success
        assert result.data["input_cost"] == pytest.approx(1000 / 1_000_000 * 2.50)
        assert result.data["output_cost"] == pytest.approx(500 / 1_000_000 * 10.00)

    def test_unknown_model_warning(self):
        tool = cost_tracker
        result = tool.run(model="unknown-v1", input_tokens=100, output_tokens=100)
        assert result.success
        assert result.data["total_cost"] == 0.0
        assert "warning" in result.data

    def test_cumulative_cost(self):
        tool = cost_tracker
        tool.run(model="gpt-4o-mini", input_tokens=1_000_000, output_tokens=0)
        tool.run(model="gpt-4o-mini", input_tokens=1_000_000, output_tokens=0)
        result = tool.run(model="gpt-4o-mini", input_tokens=0, output_tokens=0)
        assert result.data["cumulative_cost"] == pytest.approx(0.30)

    def test_get_total_cost(self):
        tool = cost_tracker
        tool.run(model="claude-3-haiku", input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost_tracker.get_total_cost() == pytest.approx(0.25 + 1.25)

    def test_cost_by_model(self):
        tool = cost_tracker
        tool.run(model="gpt-4o", input_tokens=1_000_000, output_tokens=0)
        tool.run(model="claude-3-opus", input_tokens=1_000_000, output_tokens=0)
        by_model = cost_tracker.get_cost_by_model()
        assert "gpt-4o" in by_model
        assert "claude-3-opus" in by_model
        assert by_model["gpt-4o"] == pytest.approx(2.50)
        assert by_model["claude-3-opus"] == pytest.approx(15.00)

    def test_cost_breakdown(self):
        tool = cost_tracker
        tool.run(model="gpt-4o", input_tokens=100, output_tokens=200)
        breakdown = cost_tracker.get_cost_breakdown()
        assert len(breakdown) == 1
        assert breakdown[0]["model"] == "gpt-4o"

    def test_clear(self):
        tool = cost_tracker
        tool.run(model="gpt-4o", input_tokens=100, output_tokens=100)
        cost_tracker.clear()
        assert cost_tracker.get_total_cost() == 0.0
        assert len(cost_tracker.get_cost_breakdown()) == 0

    def test_provider_field(self):
        tool = cost_tracker
        tool.run(model="gpt-4o", input_tokens=10, output_tokens=10, provider="openai")
        entry = cost_tracker.get_cost_breakdown()[0]
        assert entry["provider"] == "openai"


# ── performance_profile ─────────────────────────────────────────────


class TestPerformanceProfileTool:
    def test_record_profile(self):
        tool = performance_profile
        result = tool.run(tool_name="my_tool", duration_ms=150.5, success=True)
        assert result.success
        assert result.data["recorded"] is True
        assert result.data["duration_ms"] == 150.5

    def test_get_profiles(self):
        tool = performance_profile
        tool.run(tool_name="a", duration_ms=10, success=True)
        tool.run(tool_name="b", duration_ms=20, success=False)
        profiles = performance_profile.get_profiles()
        assert len(profiles) == 2

    def test_summary_stats(self):
        tool = performance_profile
        for d in [100, 200, 300, 400, 500]:
            tool.run(tool_name="op", duration_ms=d, success=True)
        summary = performance_profile.get_summary("op")
        assert summary["count"] == 5
        assert summary["min"] == 100.0
        assert summary["max"] == 500.0
        assert summary["avg"] == 300.0

    def test_summary_p95(self):
        tool = performance_profile
        # 20 values: 1..20
        for d in range(1, 21):
            tool.run(tool_name="op", duration_ms=float(d), success=True)
        summary = performance_profile.get_summary("op")
        # p95 index = int(0.95 * 19) = 18 -> value 19
        assert summary["p95"] == 19.0

    def test_error_rate(self):
        tool = performance_profile
        tool.run(tool_name="flaky", duration_ms=10, success=True)
        tool.run(tool_name="flaky", duration_ms=20, success=False)
        tool.run(tool_name="flaky", duration_ms=30, success=True)
        tool.run(tool_name="flaky", duration_ms=40, success=False)
        summary = performance_profile.get_summary("flaky")
        assert summary["error_rate"] == pytest.approx(0.5)

    def test_summary_empty(self):
        assert performance_profile.get_summary("nope") == {}

    def test_clear(self):
        tool = performance_profile
        tool.run(tool_name="x", duration_ms=1, success=True)
        performance_profile.clear()
        assert len(performance_profile.get_profiles()) == 0

    def test_metadata(self):
        tool = performance_profile
        tool.run(tool_name="x", duration_ms=5, success=True, metadata={"env": "test"})
        profiles = performance_profile.get_profiles()
        assert profiles[0]["metadata"] == {"env": "test"}

    def test_missing_required_params(self):
        tool = performance_profile
        assert not tool.run(tool_name="x").success
        assert not tool.run(duration_ms=10).success
