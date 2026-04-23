"""
Browser content extraction tools.

browser_extract: extract text or HTML content from the page or an element.
browser_screenshot: capture a screenshot of the page or an element.
"""

from __future__ import annotations

import os

from quartermaster_tools.builtin.browser.session import BrowserSessionManager
from quartermaster_tools.decorator import tool

_PLAYWRIGHT_MISSING = (
    "Playwright is not installed. "
    "Install with: pip install playwright && playwright install chromium"
)


@tool()
def browser_extract(selector: str = "", format: str = "text") -> dict:
    """Extract content from the page.

    Extracts text or HTML content from the current page or
    a specific element identified by a CSS selector.

    Args:
        selector: CSS selector to extract from. Omit for whole page.
        format: Output format: text or html.
    """
    if not BrowserSessionManager.is_available():
        raise RuntimeError(_PLAYWRIGHT_MISSING)

    page = BrowserSessionManager.get_page()
    fmt = format

    if selector:
        element = page.query_selector(selector)
        if element is None:
            raise ValueError(f"No element found for selector: {selector}")
        if fmt == "html":
            content = element.inner_html()
        else:
            content = element.inner_text()
    else:
        if fmt == "html":
            content = page.content()
        else:
            content = page.inner_text("body")

    return {
        "content": content,
        "format": fmt,
        "length": len(content),
    }


@tool()
def browser_screenshot(output_path: str, selector: str = "", full_page: bool = False) -> dict:
    """Take a browser screenshot.

    Captures a PNG screenshot of the current page or a specific
    element and saves it to the given file path.

    Args:
        output_path: File path where the PNG screenshot will be saved.
        selector: CSS selector of the element to screenshot. Omit for full page.
        full_page: Capture the full scrollable page (ignored when selector is set).
    """
    if not BrowserSessionManager.is_available():
        raise RuntimeError(_PLAYWRIGHT_MISSING)

    if not output_path:
        raise ValueError("Parameter 'output_path' is required")

    page = BrowserSessionManager.get_page()

    if selector:
        element = page.query_selector(selector)
        if element is None:
            raise ValueError(f"No element found for selector: {selector}")
        element.screenshot(path=output_path)
    else:
        page.screenshot(path=output_path, full_page=full_page)

    size_bytes = os.path.getsize(output_path)
    return {"saved_to": output_path, "size_bytes": size_bytes}
