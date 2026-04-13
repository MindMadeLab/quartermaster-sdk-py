"""
LLM API cost tracking tool.

Tracks token usage and calculates costs based on built-in pricing tables
for common models. Accumulates costs across calls for budget monitoring.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from quartermaster_tools.decorator import tool

_COST_STORE: list[dict[str, Any]] = []

# Pricing per 1M tokens: (input_cost, output_cost)
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-haiku": (0.25, 1.25),
    "claude-3-opus": (15.00, 75.00),
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
}


@tool()
def cost_tracker(
    model: str,
    input_tokens: int = None,
    output_tokens: int = None,
    provider: str = None,
) -> dict:
    """Track LLM API call costs.

    Calculates and tracks costs for LLM API calls based on
    model, token counts, and built-in pricing tables.
    Accumulates costs for budget monitoring and reporting.

    Args:
        model: Model name (e.g. 'gpt-4o', 'claude-3-5-sonnet').
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        provider: Optional provider name (e.g. 'openai', 'anthropic').
    """
    if not model:
        raise ValueError("Parameter 'model' is required")

    if input_tokens is None:
        raise ValueError("Parameter 'input_tokens' is required")

    if output_tokens is None:
        raise ValueError("Parameter 'output_tokens' is required")

    input_tokens = int(input_tokens)
    output_tokens = int(output_tokens)
    timestamp = datetime.now(timezone.utc).isoformat()

    warning: str | None = None
    pricing = _PRICING.get(model)
    if pricing is not None:
        input_cost_per_m, output_cost_per_m = pricing
        input_cost = (input_tokens / 1_000_000) * input_cost_per_m
        output_cost = (output_tokens / 1_000_000) * output_cost_per_m
    else:
        input_cost = 0.0
        output_cost = 0.0
        warning = f"Unknown model '{model}': cost set to 0"

    total_cost = input_cost + output_cost

    entry: dict[str, Any] = {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "provider": provider,
        "timestamp": timestamp,
    }
    _COST_STORE.append(entry)

    cumulative_cost = sum(e["total_cost"] for e in _COST_STORE)

    data: dict[str, Any] = {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "cumulative_cost": cumulative_cost,
    }
    if warning:
        data["warning"] = warning

    return data


def get_total_cost() -> float:
    """Return the total cumulative cost across all tracked calls."""
    return sum(e["total_cost"] for e in _COST_STORE)


def get_cost_by_model() -> dict[str, float]:
    """Return total cost grouped by model."""
    by_model: dict[str, float] = {}
    for entry in _COST_STORE:
        model = entry["model"]
        by_model[model] = by_model.get(model, 0.0) + entry["total_cost"]
    return by_model


def get_cost_breakdown() -> list[dict[str, Any]]:
    """Return all cost entries."""
    return list(_COST_STORE)


def clear_costs() -> None:
    """Clear all cost tracking data."""
    _COST_STORE.clear()


# Attach class-method-like helpers to the FunctionTool instance
cost_tracker.get_total_cost = get_total_cost  # type: ignore[attr-defined]
cost_tracker.get_cost_by_model = get_cost_by_model  # type: ignore[attr-defined]
cost_tracker.get_cost_breakdown = get_cost_breakdown  # type: ignore[attr-defined]
cost_tracker.clear = clear_costs  # type: ignore[attr-defined]

# Backward-compatible alias
CostTrackerTool = cost_tracker
