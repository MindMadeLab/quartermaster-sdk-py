"""
Browser navigation and wait tools.

BrowserNavigateTool: navigate to a URL.
BrowserWaitTool: wait for an element to reach a desired state.
"""

from __future__ import annotations

from typing import Any

from quartermaster_tools.base import AbstractTool
from quartermaster_tools.builtin.browser.session import BrowserSessionManager
from quartermaster_tools.types import (
    ToolDescriptor,
    ToolParameter,
    ToolParameterOption,
    ToolResult,
)

_PLAYWRIGHT_MISSING = (
    "Playwright is not installed. "
    "Install with: pip install playwright && playwright install chromium"
)


class BrowserNavigateTool(AbstractTool):
    """Navigate the browser to a URL."""

    def name(self) -> str:
        return "browser_navigate"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="url",
                description="The URL to navigate to.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="wait_for",
                description="When to consider navigation complete.",
                type="string",
                required=False,
                default="load",
                options=[
                    ToolParameterOption(label="load", value="load"),
                    ToolParameterOption(label="networkidle", value="networkidle"),
                    ToolParameterOption(label="domcontentloaded", value="domcontentloaded"),
                ],
            ),
            ToolParameter(
                name="timeout",
                description="Navigation timeout in milliseconds.",
                type="number",
                required=False,
                default=30000,
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Navigate browser to a URL.",
            long_description=(
                "Opens a URL in the browser, waits for the page to reach "
                "the specified load state, and returns the final URL, title, "
                "and HTTP status code."
            ),
            version=self.version(),
            parameters=self.parameters(),
        )

    def run(self, **kwargs: Any) -> ToolResult:
        if not BrowserSessionManager.is_available():
            return ToolResult(success=False, error=_PLAYWRIGHT_MISSING)

        url: str = kwargs.get("url", "")
        if not url:
            return ToolResult(success=False, error="Parameter 'url' is required")

        wait_for: str = kwargs.get("wait_for", "load")
        timeout: int = int(kwargs.get("timeout", 30000))

        try:
            page = BrowserSessionManager.get_page()
            response = page.goto(url, wait_until=wait_for, timeout=timeout)
            status = response.status if response else 0
            return ToolResult(
                success=True,
                data={
                    "url": page.url,
                    "title": page.title(),
                    "status": status,
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Navigation failed: {e}")


class BrowserWaitTool(AbstractTool):
    """Wait for an element to reach a desired state."""

    def name(self) -> str:
        return "browser_wait"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="selector",
                description="CSS selector of the element to wait for.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="timeout",
                description="Maximum wait time in milliseconds.",
                type="number",
                required=False,
                default=5000,
            ),
            ToolParameter(
                name="state",
                description="Desired element state.",
                type="string",
                required=False,
                default="visible",
                options=[
                    ToolParameterOption(label="visible", value="visible"),
                    ToolParameterOption(label="hidden", value="hidden"),
                    ToolParameterOption(label="attached", value="attached"),
                    ToolParameterOption(label="detached", value="detached"),
                ],
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Wait for an element on the page.",
            long_description=(
                "Waits until the specified CSS selector matches an element "
                "in the desired state (visible, hidden, attached, or detached)."
            ),
            version=self.version(),
            parameters=self.parameters(),
        )

    def run(self, **kwargs: Any) -> ToolResult:
        if not BrowserSessionManager.is_available():
            return ToolResult(success=False, error=_PLAYWRIGHT_MISSING)

        selector: str = kwargs.get("selector", "")
        if not selector:
            return ToolResult(success=False, error="Parameter 'selector' is required")

        timeout: int = int(kwargs.get("timeout", 5000))
        state: str = kwargs.get("state", "visible")

        try:
            page = BrowserSessionManager.get_page()
            page.wait_for_selector(selector, timeout=timeout, state=state)
            return ToolResult(
                success=True,
                data={"found": True, "selector": selector, "state": state},
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Wait failed: {e}")
