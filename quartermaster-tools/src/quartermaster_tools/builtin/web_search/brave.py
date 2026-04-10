"""
BraveSearchTool: Web search via Brave Search API.

Requires environment variable BRAVE_API_KEY.
Uses httpx for HTTP requests.
"""

from __future__ import annotations

import os
from typing import Any

from quartermaster_tools.base import AbstractTool
from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolParameterOption, ToolResult

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

_BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
_DEFAULT_COUNT = 5
_MAX_COUNT = 20


class BraveSearchTool(AbstractTool):
    """Search the web using the Brave Search API.

    Requires BRAVE_API_KEY environment variable to be set.
    """

    def name(self) -> str:
        return "brave_search"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                description="The search query string.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="count",
                description="Number of results to return (1-20, default 5).",
                type="number",
                required=False,
                default=_DEFAULT_COUNT,
            ),
            ToolParameter(
                name="country",
                description="Country code to filter results (e.g. 'US', 'GB').",
                type="string",
                required=False,
            ),
            ToolParameter(
                name="freshness",
                description="Freshness filter: 'day', 'week', or 'month'.",
                type="string",
                required=False,
                options=[
                    ToolParameterOption(label="Past day", value="day"),
                    ToolParameterOption(label="Past week", value="week"),
                    ToolParameterOption(label="Past month", value="month"),
                ],
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Search the web using Brave Search API.",
            long_description=(
                "Performs a web search via the Brave Search API and returns "
                "structured results with title, URL, and snippet. "
                "Requires BRAVE_API_KEY environment variable."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=False,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        query: str = kwargs.get("query", "").strip()
        count: int = min(
            max(1, int(kwargs.get("count", _DEFAULT_COUNT))),
            _MAX_COUNT,
        )
        country: str | None = kwargs.get("country")
        freshness: str | None = kwargs.get("freshness")

        if not query:
            return ToolResult(success=False, error="Parameter 'query' is required.")

        api_key = os.environ.get("BRAVE_API_KEY", "")
        if not api_key:
            return ToolResult(
                success=False,
                error=(
                    "Brave Search requires the BRAVE_API_KEY environment variable. "
                    "Get one at https://api.search.brave.com/"
                ),
            )

        if httpx is None:
            return ToolResult(
                success=False,
                error=(
                    "httpx is required for BraveSearchTool. "
                    "Install it with: pip install quartermaster-tools[web]"
                ),
            )

        params: dict[str, Any] = {
            "q": query,
            "count": count,
        }
        if country:
            params["country"] = country
        if freshness and freshness in ("day", "week", "month"):
            params["freshness"] = freshness

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key,
        }

        try:
            with httpx.Client(timeout=30) as client:
                response = client.get(_BRAVE_SEARCH_URL, params=params, headers=headers)
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException:
            return ToolResult(success=False, error="Brave search request timed out.")
        except httpx.HTTPStatusError as e:
            return ToolResult(
                success=False,
                error=f"Brave API HTTP error {e.response.status_code}: {e.response.text}",
            )
        except httpx.HTTPError as e:
            return ToolResult(success=False, error=f"HTTP error during search: {e}")

        results = []
        web_results = data.get("web", {}).get("results", [])
        for item in web_results:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("description", ""),
            })

        return ToolResult(
            success=True,
            data={
                "query": query,
                "results": results,
                "result_count": len(results),
            },
        )
