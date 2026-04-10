"""
Web search and scraping tools for quartermaster-tools.

Provides:
- DuckDuckGoSearchTool: Zero-config web search via DuckDuckGo HTML
- GoogleSearchTool: Web search via Google Custom Search JSON API
- BraveSearchTool: Web search via Brave Search API
- WebScraperTool: Fetch and convert web pages to text/markdown/html
- JsonApiTool: JSON API client with optional JMESPath filtering
"""

from quartermaster_tools.builtin.web_search.brave import BraveSearchTool
from quartermaster_tools.builtin.web_search.duckduckgo import DuckDuckGoSearchTool
from quartermaster_tools.builtin.web_search.google import GoogleSearchTool
from quartermaster_tools.builtin.web_search.json_api import JsonApiTool
from quartermaster_tools.builtin.web_search.scraper import WebScraperTool

__all__ = [
    "BraveSearchTool",
    "DuckDuckGoSearchTool",
    "GoogleSearchTool",
    "JsonApiTool",
    "WebScraperTool",
]
