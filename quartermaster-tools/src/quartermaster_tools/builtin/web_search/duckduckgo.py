"""
DuckDuckGoSearchTool: Zero-config web search via DuckDuckGo HTML interface.

Parses search results from the DuckDuckGo HTML endpoint, requiring no API key.
Uses httpx for HTTP requests and regex/string parsing for result extraction.
"""

from __future__ import annotations

import html
import re
from typing import Any
from urllib.parse import quote_plus, unquote

from quartermaster_tools.base import AbstractTool
from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolResult

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


class DuckDuckGoSearchTool(AbstractTool):
    """Search the web using DuckDuckGo HTML search. No API key required.

    This is the zero-config search option for AI agents. It parses results
    from the DuckDuckGo HTML endpoint and returns structured title/url/snippet
    data for each result.
    """

    def name(self) -> str:
        """Return the tool name."""
        return "duckduckgo_search"

    def version(self) -> str:
        """Return the tool version."""
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        """Return parameter definitions for the tool."""
        return [
            ToolParameter(
                name="query",
                description="The search query string.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="max_results",
                description=f"Maximum number of results to return (default {_DEFAULT_MAX_RESULTS}, max {_HARD_MAX_RESULTS}).",
                type="number",
                required=False,
                default=_DEFAULT_MAX_RESULTS,
            ),
        ]

    def info(self) -> ToolDescriptor:
        """Return metadata describing this tool."""
        return ToolDescriptor(
            name=self.name(),
            short_description="Search the web using DuckDuckGo (no API key needed).",
            long_description=(
                "Performs a web search via DuckDuckGo's HTML endpoint and returns "
                "structured results with title, URL, and snippet for each match. "
                "Requires no API key or configuration."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=False,
        )

    def search(self, query: str, max_results: int = _DEFAULT_MAX_RESULTS) -> ToolResult:
        """Execute a DuckDuckGo search and return parsed results.

        Args:
            query: The search query string.
            max_results: Maximum number of results to return.

        Returns:
            ToolResult with a list of search results (title, url, snippet).
        """
        return self.run(query=query, max_results=max_results)

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the search and return results.

        Args:
            query: The search query string.
            max_results: Maximum number of results (default 5, max 20).

        Returns:
            ToolResult with list of dicts containing title, url, snippet.
        """
        query: str = kwargs.get("query", "").strip()
        max_results: int = min(
            int(kwargs.get("max_results", _DEFAULT_MAX_RESULTS)),
            _HARD_MAX_RESULTS,
        )

        if not query:
            return ToolResult(success=False, error="Parameter 'query' is required")

        try:
            import httpx
        except ImportError:
            return ToolResult(
                success=False,
                error=(
                    "httpx is required for DuckDuckGoSearchTool. "
                    "Install it with: pip install quartermaster-tools[web]"
                ),
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
            return ToolResult(success=False, error="Search request timed out")
        except httpx.HTTPError as e:
            return ToolResult(success=False, error=f"HTTP error during search: {e}")

        results = self._parse_results(html_body, max_results)

        return ToolResult(
            success=True,
            data={
                "query": query,
                "results": results,
                "result_count": len(results),
            },
        )

    def _parse_results(self, html_body: str, max_results: int) -> list[dict[str, str]]:
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
