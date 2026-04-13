"""
Google Custom Search JSON API web search.

Requires environment variables GOOGLE_API_KEY and GOOGLE_CSE_ID.
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

_GOOGLE_SEARCH_URL = "https://www.googleapis.com/customsearch/v1"
_DEFAULT_NUM_RESULTS = 5
_MAX_NUM_RESULTS = 10


@tool()
def google_search(
    query: str,
    num_results: int = _DEFAULT_NUM_RESULTS,
    language: str = None,
    region: str = None,
) -> dict:
    """Search the web using Google Custom Search API.

    Performs a web search via the Google Custom Search JSON API and
    returns structured results with title, URL, and snippet.
    Requires GOOGLE_API_KEY and GOOGLE_CSE_ID environment variables.

    Args:
        query: The search query string.
        num_results: Number of results to return (1-10, default 5).
        language: Language code for results (e.g. 'en', 'de').
        region: Region/country code for results (e.g. 'us', 'uk').
    """
    query = query.strip() if query else ""
    num_results = min(max(1, int(num_results)), _MAX_NUM_RESULTS)

    if not query:
        raise ValueError("Parameter 'query' is required.")

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    cse_id = os.environ.get("GOOGLE_CSE_ID", "")

    if not api_key or not cse_id:
        raise ValueError(
            "Google Search requires GOOGLE_API_KEY and GOOGLE_CSE_ID "
            "environment variables. Get them at "
            "https://developers.google.com/custom-search/v1/introduction"
        )

    if httpx is None:
        raise ImportError(
            "httpx is required for GoogleSearchTool. "
            "Install it with: pip install quartermaster-tools[web]"
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
        raise TimeoutError("Google search request timed out.")
    except httpx.HTTPStatusError as e:
        raise RuntimeError(
            f"Google API HTTP error {e.response.status_code}: {e.response.text}"
        )
    except httpx.HTTPError as e:
        raise RuntimeError(f"HTTP error during search: {e}")

    results = []
    for item in data.get("items", []):
        results.append({
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "snippet": item.get("snippet", ""),
        })

    return {
        "query": query,
        "results": results,
        "result_count": len(results),
    }


# Backward-compatible alias
GoogleSearchTool = google_search
