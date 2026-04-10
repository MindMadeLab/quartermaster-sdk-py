"""
Browser interaction tools: click, type, and evaluate JavaScript.

BrowserClickTool: click an element by CSS selector.
BrowserTypeTool: type text into an input field.
BrowserEvalTool: execute arbitrary JavaScript in the page context.
"""

from __future__ import annotations

from typing import Any

from quartermaster_tools.base import AbstractTool
from quartermaster_tools.builtin.browser.session import BrowserSessionManager
from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolResult

_PLAYWRIGHT_MISSING = (
    "Playwright is not installed. "
    "Install with: pip install playwright && playwright install chromium"
)


class BrowserClickTool(AbstractTool):
    """Click an element on the page."""

    def name(self) -> str:
        return "browser_click"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="selector",
                description="CSS selector of the element to click.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="timeout",
                description="Maximum wait time in milliseconds for the element.",
                type="number",
                required=False,
                default=5000,
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Click an element on the page.",
            long_description=(
                "Clicks the first element matching the given CSS selector. "
                "Waits for the element to be actionable before clicking."
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

        try:
            page = BrowserSessionManager.get_page()
            page.click(selector, timeout=timeout)
            return ToolResult(
                success=True,
                data={"clicked": True, "selector": selector},
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Click failed: {e}")


class BrowserTypeTool(AbstractTool):
    """Type text into an input field."""

    def name(self) -> str:
        return "browser_type"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="selector",
                description="CSS selector of the input field.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="text",
                description="Text to type into the field.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="clear_first",
                description="Whether to clear the field before typing.",
                type="boolean",
                required=False,
                default=True,
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Type text into an input field.",
            long_description=(
                "Types text into the input element matching the given CSS "
                "selector. Optionally clears existing content first by "
                "triple-clicking to select all before typing."
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

        text: str = kwargs.get("text", "")
        if not text and text != "":
            return ToolResult(success=False, error="Parameter 'text' is required")

        clear_first: bool = kwargs.get("clear_first", True)

        try:
            page = BrowserSessionManager.get_page()
            if clear_first:
                page.click(selector, click_count=3)
            page.type(selector, text)
            return ToolResult(
                success=True,
                data={
                    "typed": True,
                    "selector": selector,
                    "text_length": len(text),
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Type failed: {e}")


class BrowserEvalTool(AbstractTool):
    """Execute JavaScript in the page context."""

    def name(self) -> str:
        return "browser_eval"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="script",
                description="JavaScript code to evaluate in the page context.",
                type="string",
                required=True,
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Execute JavaScript in the browser page.",
            long_description=(
                "Evaluates the given JavaScript expression or code block "
                "in the context of the current page and returns the result."
            ),
            version=self.version(),
            parameters=self.parameters(),
        )

    def run(self, **kwargs: Any) -> ToolResult:
        if not BrowserSessionManager.is_available():
            return ToolResult(success=False, error=_PLAYWRIGHT_MISSING)

        script: str = kwargs.get("script", "")
        if not script:
            return ToolResult(success=False, error="Parameter 'script' is required")

        try:
            page = BrowserSessionManager.get_page()
            result = page.evaluate(script)
            return ToolResult(
                success=True,
                data={"result": result},
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Eval failed: {e}")
