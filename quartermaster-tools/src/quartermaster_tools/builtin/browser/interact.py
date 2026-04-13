"""
Browser interaction tools: click, type, and evaluate JavaScript.

browser_click: click an element by CSS selector.
browser_type: type text into an input field.
browser_eval: execute arbitrary JavaScript in the page context.
"""

from __future__ import annotations

from quartermaster_tools.builtin.browser.session import BrowserSessionManager
from quartermaster_tools.decorator import tool

_PLAYWRIGHT_MISSING = (
    "Playwright is not installed. "
    "Install with: pip install playwright && playwright install chromium"
)


@tool()
def browser_click(selector: str, timeout: int = 5000) -> dict:
    """Click an element on the page.

    Clicks the first element matching the given CSS selector.
    Waits for the element to be actionable before clicking.

    Args:
        selector: CSS selector of the element to click.
        timeout: Maximum wait time in milliseconds for the element.
    """
    if not BrowserSessionManager.is_available():
        raise RuntimeError(_PLAYWRIGHT_MISSING)

    if not selector:
        raise ValueError("Parameter 'selector' is required")

    page = BrowserSessionManager.get_page()
    page.click(selector, timeout=int(timeout))
    return {"clicked": True, "selector": selector}


@tool()
def browser_type(selector: str, text: str, clear_first: bool = True) -> dict:
    """Type text into an input field.

    Types text into the input element matching the given CSS
    selector. Optionally clears existing content first by
    triple-clicking to select all before typing.

    Args:
        selector: CSS selector of the input field.
        text: Text to type into the field.
        clear_first: Whether to clear the field before typing.
    """
    if not BrowserSessionManager.is_available():
        raise RuntimeError(_PLAYWRIGHT_MISSING)

    if not selector:
        raise ValueError("Parameter 'selector' is required")

    page = BrowserSessionManager.get_page()
    if clear_first:
        page.click(selector, click_count=3)
    page.type(selector, text)
    return {
        "typed": True,
        "selector": selector,
        "text_length": len(text),
    }


@tool()
def browser_eval(script: str) -> dict:
    """Execute JavaScript in the browser page.

    Evaluates the given JavaScript expression or code block
    in the context of the current page and returns the result.

    Args:
        script: JavaScript code to evaluate in the page context.
    """
    if not BrowserSessionManager.is_available():
        raise RuntimeError(_PLAYWRIGHT_MISSING)

    if not script:
        raise ValueError("Parameter 'script' is required")

    page = BrowserSessionManager.get_page()
    result = page.evaluate(script)
    return {"result": result}


# Backward-compatible aliases
BrowserClickTool = browser_click
BrowserTypeTool = browser_type
BrowserEvalTool = browser_eval
