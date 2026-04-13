"""
Browser automation tools wrapping Playwright.

Provides tools for navigating, interacting with, and extracting data from
web pages. Requires Playwright as an optional dependency -- all tools fail
gracefully with a helpful error message when Playwright is not installed.

Install with: pip install playwright && playwright install chromium
"""

from quartermaster_tools.builtin.browser.extract import (
    BrowserExtractTool,
    BrowserScreenshotTool,
    browser_extract,
    browser_screenshot,
)
from quartermaster_tools.builtin.browser.interact import (
    BrowserClickTool,
    BrowserEvalTool,
    BrowserTypeTool,
    browser_click,
    browser_eval,
    browser_type,
)
from quartermaster_tools.builtin.browser.navigate import (
    BrowserNavigateTool,
    BrowserWaitTool,
    browser_navigate,
    browser_wait,
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
    "browser_click",
    "browser_eval",
    "browser_extract",
    "browser_navigate",
    "browser_screenshot",
    "browser_type",
    "browser_wait",
]
