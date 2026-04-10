"""
Browser automation tools wrapping Playwright.

Provides tools for navigating, interacting with, and extracting data from
web pages. Requires Playwright as an optional dependency — all tools fail
gracefully with a helpful error message when Playwright is not installed.

Install with: pip install playwright && playwright install chromium
"""

from quartermaster_tools.builtin.browser.extract import (
    BrowserExtractTool,
    BrowserScreenshotTool,
)
from quartermaster_tools.builtin.browser.interact import (
    BrowserClickTool,
    BrowserEvalTool,
    BrowserTypeTool,
)
from quartermaster_tools.builtin.browser.navigate import (
    BrowserNavigateTool,
    BrowserWaitTool,
)
from quartermaster_tools.builtin.browser.session import BrowserSessionManager

__all__ = [
    "BrowserClickTool",
    "BrowserEvalTool",
    "BrowserExtractTool",
    "BrowserNavigateTool",
    "BrowserScreenshotTool",
    "BrowserSessionManager",
    "BrowserTypeTool",
    "BrowserWaitTool",
]
