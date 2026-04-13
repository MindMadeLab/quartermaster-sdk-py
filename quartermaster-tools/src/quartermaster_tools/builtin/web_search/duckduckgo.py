"""
DuckDuckGo web search: zero-config via the HTML interface.

Parses search results from the DuckDuckGo HTML endpoint, requiring no API key.
Uses httpx for HTTP requests and regex/string parsing for result extraction.
"""

from __future__ import annotations

import html
import re
from urllib.parse import unquote

from quartermaster_tools.decorator import tool

# DuckDuckGo HTML search endpoint
_DDG_URL = "https://html.duckduckgo.com/html/"

# Default and max results
_DEFAULT_MAX_RESULTS = 5
_HARD_MAX_RESULTS = 20

# Regex patterns for parsing DDG HTML results
_RESULT_BLOCK_RE = re.compile(
    r'<div class="result results_links results_links_deep[^"]*">(.*?)</div>\s*</div>',
    re.DOTALL,
)
_TITLE_RE = re.compile(r'<a[^>]*class="result__a"[^>]*>(.*?)</a>', re.DOTALL)
_URL_RE = re.compile(r'<a[^>]*class="result__a"[^>]*href="([^"]*)"', re.DOTALL)
_SNIPPET_RE = re.compile(
    r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL
)


def _strip_html_tags(text: str) -> str:
    """Remove HTML tags and decode entities from a string."""
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    return text.strip()


def _extract_url(raw_href: str) -> str:
    """Extract the actual URL from a DuckDuckGo redirect href."""
    # DDG wraps URLs like //duckduckgo.com/l/?uddg=<encoded_url>&...
    match = re.search(r"uddg=([^&]+)", raw_href)
    if match:
        return unquote(match.group(1))
    # Fallback: return as-is (may already be a direct URL)
    if raw_href.startswith("//"):
        return "https:" + raw_href
    return raw_href


def _parse_results(html_body: str, max_results: int) -> list[dict[str, str]]:
    """Parse search results from DuckDuckGo HTML response.

    Args:
        html_body: The raw HTML response body.
        max_results: Maximum number of results to extract.

    Returns:
        List of dicts with title, url, and snippet keys.
    """
    results: list[dict[str, str]] = []

    blocks = _RESULT_BLOCK_RE.findall(html_body)
    for block in blocks[:max_results]:
        title_match = _TITLE_RE.search(block)
        url_match = _URL_RE.search(block)
        snippet_match = _SNIPPET_RE.search(block)

        if not title_match or not url_match:
            continue

        title = _strip_html_tags(title_match.group(1))
        url = _extract_url(url_match.group(1))
        snippet = _strip_html_tags(snippet_match.group(1)) if snippet_match else ""

        results.append({
            "title": title,
            "url": url,
            "snippet": snippet,
        })

    return results


@tool()
def duckduckgo_search(query: str, max_results: int = _DEFAULT_MAX_RESULTS) -> dict:
    """Search the web using DuckDuckGo (no API key needed).

    Performs a web search via DuckDuckGo's HTML endpoint and returns
    structured results with title, URL, and snippet for each match.
    Requires no API key or configuration.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default 5, max 20).
    """
    query = query.strip() if query else ""
    max_results = min(int(max_results), _HARD_MAX_RESULTS)

    if not query:
        raise ValueError("Parameter 'query' is required")

    try:
        import httpx
    except ImportError:
        raise ImportError(
            "httpx is required for DuckDuckGoSearchTool. "
            "Install it with: pip install quartermaster-tools[web]"
        )

    try:
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            response = client.post(
                _DDG_URL,
                data={"q": query},
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; QuartermasterBot/1.0)",
                },
            )
            response.raise_for_status()
            html_body = response.text
    except httpx.TimeoutException:
        raise TimeoutError("Search request timed out")
    except httpx.HTTPError as e:
        raise RuntimeError(f"HTTP error during search: {e}")

    results = _parse_results(html_body, max_results)

    return {
        "query": query,
        "results": results,
        "result_count": len(results),
    }


# Backward-compatible alias
DuckDuckGoSearchTool = duckduckgo_search
