"""
LLM API cost tracking tool.

Tracks token usage and calculates costs based on built-in pricing tables
for common models. Accumulates costs across calls for budget monitoring.

v0.4.0 extends the tool with **local-GPU pricing**:
callers running an on-prem Ollama/vLLM/etc. model can pass
``duration_seconds`` and ``local_gpu_cost_per_hour`` to approximate real
spend (electricity + amortisation) when no per-token cloud price exists.

Precedence when both are available:
    If ``local_gpu_cost_per_hour`` (explicit or registered via
    :func:`register_local_pricing`) and ``duration_seconds`` are both
    provided, local-GPU pricing wins over the cloud-per-token table —
    the caller signalled a local run by supplying wall-clock time, so
    that intent takes precedence. The result's ``cost_basis`` field is
    set to ``"local_gpu_time"`` so downstream consumers can tell what
    was used.
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

# Module-level registry for per-model local-GPU hourly cost.
# Populated via :func:`register_local_pricing` at boot by callers who
# want the tool to resolve a model name → $/hr without passing it on
# every call.
_LOCAL_GPU_PRICING: dict[str, float] = {}


def register_local_pricing(model: str, cost_per_hour: float) -> None:
    """Register a local-GPU hourly cost for *model*.

    Subsequent :func:`cost_tracker` calls for this model that supply
    ``duration_seconds`` will use this price automatically unless an
    explicit ``local_gpu_cost_per_hour`` kwarg overrides it.

    Args:
        model: Model name to associate the price with (e.g. ``"gemma4:26b"``).
        cost_per_hour: GPU cost per hour in dollars (electricity +
            amortisation, at the caller's discretion).
    """
    if not model:
        raise ValueError("Parameter 'model' is required")
    _LOCAL_GPU_PRICING[model] = float(cost_per_hour)


def clear_local_pricing() -> None:
    """Clear the registered local-GPU pricing table."""
    _LOCAL_GPU_PRICING.clear()


@tool()
def cost_tracker(
    model: str,
    input_tokens: int = None,
    output_tokens: int = None,
    provider: str = None,
    duration_seconds: float = None,
    local_gpu_cost_per_hour: float = None,
) -> dict:
    """Track LLM API call costs.

    Calculates and tracks costs for LLM API calls based on
    model, token counts, and built-in pricing tables.
    Accumulates costs for budget monitoring and reporting.

    Pricing resolution (v0.4.0):

    * If ``local_gpu_cost_per_hour`` (or a registered price for *model*
      via :func:`register_local_pricing`) AND ``duration_seconds`` are
      both set, cost is computed as
      ``duration_seconds / 3600 * local_gpu_cost_per_hour`` and
      ``cost_basis`` is ``"local_gpu_time"``.
    * Else if *model* is in the cloud pricing table, per-token cloud
      pricing is used and ``cost_basis`` is ``"cloud_per_token"``.
    * Else cost is ``0.0`` and ``cost_basis`` is ``"unknown"``.

    When both local and cloud pricing could apply (cloud model name +
    explicit ``duration_seconds``/``local_gpu_cost_per_hour``), local
    pricing wins — the caller signalled intent by supplying wall-clock
    time. Token counts remain optional so callers can record a local
    run without token accounting.

    Args:
        model: Model name (e.g. 'gpt-4o', 'claude-3-5-sonnet', 'gemma4:26b').
        input_tokens: Number of input tokens (optional for local-GPU pricing).
        output_tokens: Number of output tokens (optional for local-GPU pricing).
        provider: Optional provider name (e.g. 'openai', 'anthropic', 'ollama').
        duration_seconds: Wall-clock runtime in seconds. Required for
            local-GPU pricing.
        local_gpu_cost_per_hour: GPU cost per hour in dollars (overrides
            any value registered via :func:`register_local_pricing`).
    """
    if not model:
        raise ValueError("Parameter 'model' is required")

    # Resolve effective local-GPU $/hr: explicit kwarg wins, else registry.
    effective_gpu_rate: float | None = None
    if local_gpu_cost_per_hour is not None:
        effective_gpu_rate = float(local_gpu_cost_per_hour)
    elif model in _LOCAL_GPU_PRICING:
        effective_gpu_rate = _LOCAL_GPU_PRICING[model]

    use_local_pricing = effective_gpu_rate is not None and duration_seconds is not None

    # Tokens are only required when we fall back to per-token cloud pricing
    # and the model is NOT a local one. Otherwise default to 0 so the caller
    # can record a local run without passing them.
    if use_local_pricing:
        input_tokens = int(input_tokens) if input_tokens is not None else 0
        output_tokens = int(output_tokens) if output_tokens is not None else 0
    else:
        if input_tokens is None:
            raise ValueError("Parameter 'input_tokens' is required")
        if output_tokens is None:
            raise ValueError("Parameter 'output_tokens' is required")
        input_tokens = int(input_tokens)
        output_tokens = int(output_tokens)

    timestamp = datetime.now(timezone.utc).isoformat()

    warning: str | None = None
    input_cost = 0.0
    output_cost = 0.0
    cost_basis: str

    if use_local_pricing:
        # Local-GPU pricing takes precedence when both inputs are present.
        duration_seconds_f = float(duration_seconds)
        total_cost = duration_seconds_f / 3600.0 * float(effective_gpu_rate)
        cost_basis = "local_gpu_time"
    else:
        pricing = _PRICING.get(model)
        if pricing is not None:
            input_cost_per_m, output_cost_per_m = pricing
            input_cost = (input_tokens / 1_000_000) * input_cost_per_m
            output_cost = (output_tokens / 1_000_000) * output_cost_per_m
            total_cost = input_cost + output_cost
            cost_basis = "cloud_per_token"
        else:
            total_cost = 0.0
            cost_basis = "unknown"
            warning = f"Unknown model '{model}': cost set to 0"

    entry: dict[str, Any] = {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "provider": provider,
        "timestamp": timestamp,
        "cost_basis": cost_basis,
        "duration_seconds": float(duration_seconds) if duration_seconds is not None else None,
        "local_gpu_cost_per_hour": effective_gpu_rate,
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
        "cost_basis": cost_basis,
    }
    if duration_seconds is not None:
        data["duration_seconds"] = float(duration_seconds)
    if effective_gpu_rate is not None:
        data["local_gpu_cost_per_hour"] = effective_gpu_rate
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
cost_tracker.register_local_pricing = register_local_pricing  # type: ignore[attr-defined]
cost_tracker.clear_local_pricing = clear_local_pricing  # type: ignore[attr-defined]
