"""
Browser content extraction tools.

BrowserExtractTool: extract text or HTML content from the page or an element.
BrowserScreenshotTool: capture a screenshot of the page or an element.
"""

from __future__ import annotations

import os
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


class BrowserExtractTool(AbstractTool):
    """Extract page content as text or HTML."""

    def name(self) -> str:
        return "browser_extract"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="selector",
                description="CSS selector to extract from. Omit for whole page.",
                type="string",
                required=False,
            ),
            ToolParameter(
                name="format",
                description="Output format: text or html.",
                type="string",
                required=False,
                default="text",
                options=[
                    ToolParameterOption(label="text", value="text"),
                    ToolParameterOption(label="html", value="html"),
                ],
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Extract content from the page.",
            long_description=(
                "Extracts text or HTML content from the current page or "
                "a specific element identified by a CSS selector."
            ),
            version=self.version(),
            parameters=self.parameters(),
        )

    def run(self, **kwargs: Any) -> ToolResult:
        if not BrowserSessionManager.is_available():
            return ToolResult(success=False, error=_PLAYWRIGHT_MISSING)

        selector: str | None = kwargs.get("selector")
        fmt: str = kwargs.get("format", "text")

        try:
            page = BrowserSessionManager.get_page()

            if selector:
                element = page.query_selector(selector)
                if element is None:
                    return ToolResult(
                        success=False,
                        error=f"No element found for selector: {selector}",
                    )
                if fmt == "html":
                    content = element.inner_html()
                else:
                    content = element.inner_text()
            else:
                if fmt == "html":
                    content = page.content()
                else:
                    content = page.inner_text("body")

            return ToolResult(
                success=True,
                data={
                    "content": content,
                    "format": fmt,
                    "length": len(content),
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Extract failed: {e}")


class BrowserScreenshotTool(AbstractTool):
    """Take a screenshot of the page or an element."""

    def name(self) -> str:
        return "browser_screenshot"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="selector",
                description="CSS selector of the element to screenshot. Omit for full page.",
                type="string",
                required=False,
            ),
            ToolParameter(
                name="full_page",
                description="Capture the full scrollable page (ignored when selector is set).",
                type="boolean",
                required=False,
                default=False,
            ),
            ToolParameter(
                name="output_path",
                description="File path where the PNG screenshot will be saved.",
                type="string",
                required=True,
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Take a browser screenshot.",
            long_description=(
                "Captures a PNG screenshot of the current page or a specific "
                "element and saves it to the given file path."
            ),
            version=self.version(),
            parameters=self.parameters(),
        )

    def run(self, **kwargs: Any) -> ToolResult:
        if not BrowserSessionManager.is_available():
            return ToolResult(success=False, error=_PLAYWRIGHT_MISSING)

        output_path: str = kwargs.get("output_path", "")
        if not output_path:
            return ToolResult(success=False, error="Parameter 'output_path' is required")

        selector: str | None = kwargs.get("selector")
        full_page: bool = kwargs.get("full_page", False)

        try:
            page = BrowserSessionManager.get_page()

            if selector:
                element = page.query_selector(selector)
                if element is None:
                    return ToolResult(
                        success=False,
                        error=f"No element found for selector: {selector}",
                    )
                element.screenshot(path=output_path)
            else:
                page.screenshot(path=output_path, full_page=full_page)

            size_bytes = os.path.getsize(output_path)
            return ToolResult(
                success=True,
                data={"saved_to": output_path, "size_bytes": size_bytes},
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Screenshot failed: {e}")
