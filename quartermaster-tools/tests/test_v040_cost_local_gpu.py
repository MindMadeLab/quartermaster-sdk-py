"""
Tests for v0.4.0 (Sorex round-2 P3.5) — local-GPU cost pricing
in the Cost tracker tool.

Covers:

* Basic hour-based pricing (1hr * $1/hr == $1).
* Partial-hour pricing.
* Precedence: when a model is in the cloud table **and** local inputs
  are provided, local-GPU pricing wins.
* Persistence of registered local pricing across calls.
* Default unchanged behaviour: an unregistered model with no duration
  still returns cost 0 and cost_basis == "unknown".
* ``cost_basis`` field is always set to one of the documented values.
"""

from __future__ import annotations

import pytest

from quartermaster_tools.builtin.observability.cost import (
    CostTrackerTool,
    clear_local_pricing,
    register_local_pricing,
)


@pytest.fixture(autouse=True)
def _clear_all():
    """Isolate cost and local-pricing state between tests."""
    CostTrackerTool.clear()
    clear_local_pricing()
    yield
    CostTrackerTool.clear()
    clear_local_pricing()


class TestLocalGPUCost:
    def test_local_gpu_cost_basic(self):
        """1hr at $1/hr == $1."""
        result = CostTrackerTool.run(
            model="gemma4:26b",
            duration_seconds=3600,
            local_gpu_cost_per_hour=1.0,
        )
        assert result.success
        assert result.data["total_cost"] == pytest.approx(1.0)
        assert result.data["cost_basis"] == "local_gpu_time"

    def test_local_gpu_cost_partial_hour(self):
        """0.5/hr * 1800s == 0.25."""
        result = CostTrackerTool.run(
            model="mistral:7b",
            duration_seconds=1800,
            local_gpu_cost_per_hour=0.5,
        )
        assert result.success
        assert result.data["total_cost"] == pytest.approx(0.25)
        assert result.data["cost_basis"] == "local_gpu_time"

    def test_local_gpu_takes_precedence_over_cloud_when_both_set(self):
        """When a cloud-priced model name is used AND local GPU pricing
        is supplied, local pricing wins — the explicit duration +
        $/hr signal represents the caller's actual run."""
        result = CostTrackerTool.run(
            model="gpt-4o",  # in the cloud pricing table
            input_tokens=1000,
            output_tokens=500,
            duration_seconds=12.4,
            local_gpu_cost_per_hour=0.85,
        )
        assert result.success
        # Cloud price would be ~$0.0075; local should be ~12.4/3600 * 0.85.
        expected = 12.4 / 3600.0 * 0.85
        assert result.data["total_cost"] == pytest.approx(expected)
        assert result.data["cost_basis"] == "local_gpu_time"
        # Per-token input/output cost fields should be 0 under local pricing.
        assert result.data["input_cost"] == 0.0
        assert result.data["output_cost"] == 0.0

    def test_register_local_pricing_persists_across_calls(self):
        """Once registered, the price is used automatically when
        ``duration_seconds`` is supplied — no explicit kwarg needed."""
        register_local_pricing("gemma4:26b", 0.85)

        result_1 = CostTrackerTool.run(model="gemma4:26b", duration_seconds=10)
        assert result_1.success
        assert result_1.data["cost_basis"] == "local_gpu_time"
        assert result_1.data["total_cost"] == pytest.approx(10 / 3600.0 * 0.85)
        assert result_1.data["local_gpu_cost_per_hour"] == pytest.approx(0.85)

        # Second call still works — registration persists.
        result_2 = CostTrackerTool.run(model="gemma4:26b", duration_seconds=20)
        assert result_2.data["cost_basis"] == "local_gpu_time"
        assert result_2.data["total_cost"] == pytest.approx(20 / 3600.0 * 0.85)

    def test_register_local_pricing_explicit_kwarg_overrides(self):
        """Explicit ``local_gpu_cost_per_hour`` wins over registered value."""
        register_local_pricing("gemma4:26b", 0.85)
        result = CostTrackerTool.run(
            model="gemma4:26b",
            duration_seconds=3600,
            local_gpu_cost_per_hour=2.0,  # overrides the 0.85 registered price
        )
        assert result.success
        assert result.data["total_cost"] == pytest.approx(2.0)
        assert result.data["local_gpu_cost_per_hour"] == pytest.approx(2.0)

    def test_unregistered_model_no_duration_returns_zero(self):
        """Current default behaviour preserved: unknown model + no local
        pricing => cost 0, cost_basis 'unknown', warning emitted."""
        result = CostTrackerTool.run(
            model="some-unknown-local-model",
            input_tokens=100,
            output_tokens=100,
        )
        assert result.success
        assert result.data["total_cost"] == 0.0
        assert result.data["cost_basis"] == "unknown"
        assert "warning" in result.data

    def test_cost_basis_field_in_result(self):
        """Every result must include ``cost_basis`` in the documented set."""
        allowed = {"local_gpu_time", "cloud_per_token", "unknown"}

        # local
        r_local = CostTrackerTool.run(
            model="gemma4:26b",
            duration_seconds=60,
            local_gpu_cost_per_hour=1.0,
        )
        # cloud
        r_cloud = CostTrackerTool.run(
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
        )
        # unknown
        r_unknown = CostTrackerTool.run(
            model="no-such-model",
            input_tokens=10,
            output_tokens=10,
        )

        for r in (r_local, r_cloud, r_unknown):
            assert r.success
            assert "cost_basis" in r.data
            assert r.data["cost_basis"] in allowed

        assert r_local.data["cost_basis"] == "local_gpu_time"
        assert r_cloud.data["cost_basis"] == "cloud_per_token"
        assert r_unknown.data["cost_basis"] == "unknown"

    def test_duration_only_without_rate_falls_through_to_cloud(self):
        """Passing ``duration_seconds`` without any rate (and no registered
        price) should not trigger local pricing — cloud or unknown
        resolution still applies."""
        # gpt-4o is in the cloud table → cloud pricing used
        r_cloud = CostTrackerTool.run(
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
            duration_seconds=5.0,
        )
        assert r_cloud.data["cost_basis"] == "cloud_per_token"
        assert r_cloud.data["total_cost"] == pytest.approx(
            1000 / 1_000_000 * 2.50 + 500 / 1_000_000 * 10.00
        )
