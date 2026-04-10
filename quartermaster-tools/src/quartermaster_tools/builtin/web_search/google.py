"""
GoogleSearchTool: Web search via Google Custom Search JSON API.

Requires environment variables GOOGLE_API_KEY and GOOGLE_CSE_ID.
Uses httpx for HTTP requests.
"""

from __future__ import annotations

import os
from typing import Any

from quartermaster_tools.base import AbstractTool
from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolResult

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

_GOOGLE_SEARCH_URL = "https://www.googleapis.com/customsearch/v1"
_DEFAULT_NUM_RESULTS = 5
_MAX_NUM_RESULTS = 10


class GoogleSearchTool(AbstractTool):
    """Search the web using Google Custom Search JSON API.

    Requires GOOGLE_API_KEY and GOOGLE_CSE_ID environment variables to be set.
    """

    def name(self) -> str:
        return "google_search"

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
                name="num_results",
                description="Number of results to return (1-10, default 5).",
                type="number",
                required=False,
                default=_DEFAULT_NUM_RESULTS,
            ),
            ToolParameter(
                name="language",
                description="Language code for results (e.g. 'en', 'de').",
                type="string",
                required=False,
            ),
            ToolParameter(
                name="region",
                description="Region/country code for results (e.g. 'us', 'uk').",
                type="string",
                required=False,
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Search the web using Google Custom Search API.",
            long_description=(
                "Performs a web search via the Google Custom Search JSON API and "
                "returns structured results with title, URL, and snippet. "
                "Requires GOOGLE_API_KEY and GOOGLE_CSE_ID environment variables."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=False,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        query: str = kwargs.get("query", "").strip()
        num_results: int = min(
            max(1, int(kwargs.get("num_results", _DEFAULT_NUM_RESULTS))),
            _MAX_NUM_RESULTS,
        )
        language: str | None = kwargs.get("language")
        region: str | None = kwargs.get("region")

        if not query:
            return ToolResult(success=False, error="Parameter 'query' is required.")

        api_key = os.environ.get("GOOGLE_API_KEY", "")
        cse_id = os.environ.get("GOOGLE_CSE_ID", "")

        if not api_key or not cse_id:
            return ToolResult(
                success=False,
                error=(
                    "Google Search requires GOOGLE_API_KEY and GOOGLE_CSE_ID "
                    "environment variables. Get them at "
                    "https://developers.google.com/custom-search/v1/introduction"
                ),
            )

        if httpx is None:
            return ToolResult(
                success=False,
                error=(
                    "httpx is required for GoogleSearchTool. "
                    "Install it with: pip install quartermaster-tools[web]"
                ),
            )

        params: dict[str, Any] = {
            "key": api_key,
            "cx": cse_id,
            "q": query,
            "num": num_results,
        }
        if language:
            params["lr"] = f"lang_{language}"
        if region:
            params["gl"] = region

        try:
            with httpx.Client(timeout=30) as client:
                response = client.get(_GOOGLE_SEARCH_URL, params=params)
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException:
            return ToolResult(success=False, error="Google search request timed out.")
        except httpx.HTTPStatusError as e:
            return ToolResult(
                success=False,
                error=f"Google API HTTP error {e.response.status_code}: {e.response.text}",
            )
        except httpx.HTTPError as e:
            return ToolResult(success=False, error=f"HTTP error during search: {e}")

        results = []
        for item in data.get("items", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            })

        return ToolResult(
            success=True,
            data={
                "query": query,
                "results": results,
                "result_count": len(results),
            },
        )
