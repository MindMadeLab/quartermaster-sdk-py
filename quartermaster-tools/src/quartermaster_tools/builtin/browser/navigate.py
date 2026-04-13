"""
Browser navigation and wait tools.

browser_navigate: navigate to a URL.
browser_wait: wait for an element to reach a desired state.
"""

from __future__ import annotations

from quartermaster_tools.builtin.browser.session import BrowserSessionManager
from quartermaster_tools.decorator import tool

_PLAYWRIGHT_MISSING = (
    "Playwright is not installed. "
    "Install with: pip install playwright && playwright install chromium"
)


@tool()
def browser_navigate(url: str, wait_for: str = "load", timeout: int = 30000) -> dict:
    """Navigate browser to a URL.

    Opens a URL in the browser, waits for the page to reach
    the specified load state, and returns the final URL, title,
    and HTTP status code.

    Args:
        url: The URL to navigate to.
        wait_for: When to consider navigation complete.
        timeout: Navigation timeout in milliseconds.
    """
    if not BrowserSessionManager.is_available():
        raise RuntimeError(_PLAYWRIGHT_MISSING)

    if not url:
        raise ValueError("Parameter 'url' is required")

    page = BrowserSessionManager.get_page()
    response = page.goto(url, wait_until=wait_for, timeout=int(timeout))
    status = response.status if response else 0
    return {
        "url": page.url,
        "title": page.title(),
        "status": status,
    }


@tool()
def browser_wait(selector: str, timeout: int = 5000, state: str = "visible") -> dict:
    """Wait for an element on the page.

    Waits until the specified CSS selector matches an element
    in the desired state (visible, hidden, attached, or detached).

    Args:
        selector: CSS selector of the element to wait for.
        timeout: Maximum wait time in milliseconds.
        state: Desired element state.
    """
    if not BrowserSessionManager.is_available():
        raise RuntimeError(_PLAYWRIGHT_MISSING)

    if not selector:
        raise ValueError("Parameter 'selector' is required")

    page = BrowserSessionManager.get_page()
    page.wait_for_selector(selector, timeout=int(timeout), state=state)
    return {"found": True, "selector": selector, "state": state}


# Backward-compatible aliases
BrowserNavigateTool = browser_navigate
BrowserWaitTool = browser_wait
