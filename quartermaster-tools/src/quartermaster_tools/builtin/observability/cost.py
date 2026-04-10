"""
LLM API cost tracking tool.

Tracks token usage and calculates costs based on built-in pricing tables
for common models. Accumulates costs across calls for budget monitoring.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from quartermaster_tools.base import AbstractTool
from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolResult

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


class CostTrackerTool(AbstractTool):
    """Track LLM API call costs."""

    def name(self) -> str:
        return "cost_tracker"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="model",
                description="Model name (e.g. 'gpt-4o', 'claude-3-5-sonnet').",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="input_tokens",
                description="Number of input tokens.",
                type="number",
                required=True,
            ),
            ToolParameter(
                name="output_tokens",
                description="Number of output tokens.",
                type="number",
                required=True,
            ),
            ToolParameter(
                name="provider",
                description="Optional provider name (e.g. 'openai', 'anthropic').",
                type="string",
                required=False,
                default=None,
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Track LLM API call costs.",
            long_description=(
                "Calculates and tracks costs for LLM API calls based on "
                "model, token counts, and built-in pricing tables. "
                "Accumulates costs for budget monitoring and reporting."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        model: str = kwargs.get("model", "")
        if not model:
            return ToolResult(success=False, error="Parameter 'model' is required")

        input_tokens = kwargs.get("input_tokens")
        if input_tokens is None:
            return ToolResult(success=False, error="Parameter 'input_tokens' is required")

        output_tokens = kwargs.get("output_tokens")
        if output_tokens is None:
            return ToolResult(success=False, error="Parameter 'output_tokens' is required")

        input_tokens = int(input_tokens)
        output_tokens = int(output_tokens)
        provider: str | None = kwargs.get("provider")
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

        return ToolResult(success=True, data=data)

    @classmethod
    def get_total_cost(cls) -> float:
        """Return the total cumulative cost across all tracked calls."""
        return sum(e["total_cost"] for e in _COST_STORE)

    @classmethod
    def get_cost_by_model(cls) -> dict[str, float]:
        """Return total cost grouped by model."""
        by_model: dict[str, float] = {}
        for entry in _COST_STORE:
            model = entry["model"]
            by_model[model] = by_model.get(model, 0.0) + entry["total_cost"]
        return by_model

    @classmethod
    def get_cost_breakdown(cls) -> list[dict[str, Any]]:
        """Return all cost entries."""
        return list(_COST_STORE)

    @classmethod
    def clear(cls) -> None:
        """Clear all cost tracking data."""
        _COST_STORE.clear()
