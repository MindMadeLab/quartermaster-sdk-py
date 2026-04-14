"""
Brave Search API web search.

Requires environment variable BRAVE_API_KEY.
Uses httpx for HTTP requests.
"""

from __future__ import annotations

import os
from typing import Any

from quartermaster_tools.decorator import tool

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

_BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
_DEFAULT_COUNT = 5
_MAX_COUNT = 20


@tool()
def brave_search(
    query: str,
    count: int = _DEFAULT_COUNT,
    country: str = None,
    freshness: str = None,
) -> dict:
    """Search the web using Brave Search API.

    Performs a web search via the Brave Search API and returns
    structured results with title, URL, and snippet.
    Requires BRAVE_API_KEY environment variable.

    Args:
        query: The search query string.
        count: Number of results to return (1-20, default 5).
        country: Country code to filter results (e.g. 'US', 'GB').
        freshness: Freshness filter: 'day', 'week', or 'month'.
    """
    query = query.strip() if query else ""
    count = min(max(1, int(count)), _MAX_COUNT)

    if not query:
        raise ValueError("Parameter 'query' is required.")

    api_key = os.environ.get("BRAVE_API_KEY", "")
    if not api_key:
        raise ValueError(
            "Brave Search requires the BRAVE_API_KEY environment variable. "
            "Get one at https://api.search.brave.com/"
        )

    if httpx is None:
        raise ImportError(
            "httpx is required for BraveSearchTool. "
            "Install it with: pip install quartermaster-tools[web]"
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
        raise TimeoutError("Brave search request timed out.")
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Brave API HTTP error {e.response.status_code}: {e.response.text}")
    except httpx.HTTPError as e:
        raise RuntimeError(f"HTTP error during search: {e}")

    results = []
    web_results = data.get("web", {}).get("results", [])
    for item in web_results:
        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("description", ""),
            }
        )

    return {
        "query": query,
        "results": results,
        "result_count": len(results),
    }


# Backward-compatible alias
BraveSearchTool = brave_search
