"""
Web search and scraping tools for quartermaster-tools.

Provides:
- duckduckgo_search: Zero-config web search via DuckDuckGo HTML
- google_search: Web search via Google Custom Search JSON API
- brave_search: Web search via Brave Search API
- web_scraper: Fetch and convert web pages to text/markdown/html
- json_api: JSON API client with optional JMESPath filtering
"""

from quartermaster_tools.builtin.web_search.brave import BraveSearchTool, brave_search
from quartermaster_tools.builtin.web_search.duckduckgo import (
    DuckDuckGoSearchTool,
    duckduckgo_search,
)
from quartermaster_tools.builtin.web_search.google import GoogleSearchTool, google_search
from quartermaster_tools.builtin.web_search.json_api import JsonApiTool, json_api
from quartermaster_tools.builtin.web_search.scraper import WebScraperTool, web_scraper

__all__ = [
    "brave_search",
    "BraveSearchTool",
    "duckduckgo_search",
    "DuckDuckGoSearchTool",
    "google_search",
    "GoogleSearchTool",
    "json_api",
    "JsonApiTool",
    "web_scraper",
    "WebScraperTool",
]
