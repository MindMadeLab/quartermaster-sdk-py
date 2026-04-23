"""
Tests for browser automation tools.

All tests mock Playwright entirely so they run without Playwright installed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from quartermaster_tools.builtin.browser.extract import (
    browser_extract,
    browser_screenshot,
)
from quartermaster_tools.builtin.browser.interact import (
    browser_click,
    browser_eval,
    browser_type,
)
from quartermaster_tools.builtin.browser.navigate import (
    browser_navigate,
    browser_wait,
)
from quartermaster_tools.builtin.browser.session import BrowserSessionManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_session():
    """Reset session manager state before each test."""
    BrowserSessionManager.reset()
    yield
    BrowserSessionManager.reset()


@pytest.fixture
def mock_page():
    """Return a MagicMock configured as a Playwright Page."""
    page = MagicMock()
    page.url = "https://example.com/final"
    page.title.return_value = "Example"
    page.content.return_value = "<html><body>Hello</body></html>"
    page.inner_text.return_value = "Hello"
    return page


@pytest.fixture
def browser_available(mock_page):
    """Patch session manager: available=True, get_page returns mock_page."""
    with (
        patch.object(BrowserSessionManager, "is_available", return_value=True),
        patch.object(BrowserSessionManager, "get_page", return_value=mock_page),
    ):
        yield mock_page


@pytest.fixture
def browser_unavailable():
    """Patch session manager: available=False."""
    with patch.object(BrowserSessionManager, "is_available", return_value=False):
        yield


# ---------------------------------------------------------------------------
# browser_navigate
# ---------------------------------------------------------------------------


class TestBrowserNavigateTool:
    def test_navigate_success(self, browser_available):
        mock_page = browser_available
        response = MagicMock()
        response.status = 200
        mock_page.goto.return_value = response

        tool = browser_navigate
        result = tool.run(url="https://example.com")

        assert result.success is True
        assert result.data["url"] == "https://example.com/final"
        assert result.data["title"] == "Example"
        assert result.data["status"] == 200
        mock_page.goto.assert_called_once_with(
            "https://example.com", wait_until="load", timeout=30000
        )

    def test_navigate_timeout_error(self, browser_available):
        mock_page = browser_available
        mock_page.goto.side_effect = Exception("Timeout 30000ms exceeded")

        tool = browser_navigate
        result = tool.run(url="https://slow.example.com", timeout=30000)

        assert result.success is False
        assert "Timeout" in result.error

    def test_navigate_missing_url(self, browser_available):
        tool = browser_navigate
        result = tool.run()

        assert result.success is False
        assert "url" in result.error.lower()

    def test_navigate_custom_wait_for(self, browser_available):
        mock_page = browser_available
        response = MagicMock()
        response.status = 200
        mock_page.goto.return_value = response

        tool = browser_navigate
        tool.run(url="https://example.com", wait_for="networkidle", timeout=5000)

        mock_page.goto.assert_called_once_with(
            "https://example.com", wait_until="networkidle", timeout=5000
        )

    def test_navigate_playwright_not_installed(self, browser_unavailable):
        tool = browser_navigate
        result = tool.run(url="https://example.com")

        assert result.success is False
        assert "Playwright is not installed" in result.error

    def test_navigate_info(self):
        tool = browser_navigate
        info = tool.info()
        assert info.name == "browser_navigate"
        assert info.version == "1.0.0"

    def test_navigate_null_response(self, browser_available):
        """page.goto can return None for some navigations."""
        mock_page = browser_available
        mock_page.goto.return_value = None

        tool = browser_navigate
        result = tool.run(url="about:blank")

        assert result.success is True
        assert result.data["status"] == 0


# ---------------------------------------------------------------------------
# browser_click
# ---------------------------------------------------------------------------


class TestBrowserClickTool:
    def test_click_success(self, browser_available):
        tool = browser_click
        result = tool.run(selector="#submit-btn")

        assert result.success is True
        assert result.data["clicked"] is True
        assert result.data["selector"] == "#submit-btn"
        browser_available.click.assert_called_once_with("#submit-btn", timeout=5000)

    def test_click_element_not_found(self, browser_available):
        browser_available.click.side_effect = Exception("Element not found")

        tool = browser_click
        result = tool.run(selector="#nonexistent")

        assert result.success is False
        assert "Element not found" in result.error

    def test_click_missing_selector(self, browser_available):
        tool = browser_click
        result = tool.run()

        assert result.success is False
        assert "selector" in result.error.lower()

    def test_click_playwright_not_installed(self, browser_unavailable):
        tool = browser_click
        result = tool.run(selector="#btn")

        assert result.success is False
        assert "Playwright is not installed" in result.error


# ---------------------------------------------------------------------------
# browser_type
# ---------------------------------------------------------------------------


class TestBrowserTypeTool:
    def test_type_with_clear(self, browser_available):
        tool = browser_type
        result = tool.run(selector="#input", text="hello world", clear_first=True)

        assert result.success is True
        assert result.data["typed"] is True
        assert result.data["text_length"] == 11
        browser_available.click.assert_called_once_with("#input", click_count=3)
        browser_available.type.assert_called_once_with("#input", "hello world")

    def test_type_without_clear(self, browser_available):
        tool = browser_type
        result = tool.run(selector="#input", text="appended", clear_first=False)

        assert result.success is True
        browser_available.click.assert_not_called()
        browser_available.type.assert_called_once_with("#input", "appended")

    def test_type_missing_selector(self, browser_available):
        tool = browser_type
        result = tool.run(text="hello")

        assert result.success is False
        assert "selector" in result.error.lower()

    def test_type_playwright_not_installed(self, browser_unavailable):
        tool = browser_type
        result = tool.run(selector="#input", text="hello")

        assert result.success is False
        assert "Playwright is not installed" in result.error


# ---------------------------------------------------------------------------
# browser_extract
# ---------------------------------------------------------------------------


class TestBrowserExtractTool:
    def test_extract_text_whole_page(self, browser_available):
        browser_available.inner_text.return_value = "Page text content"

        tool = browser_extract
        result = tool.run()

        assert result.success is True
        assert result.data["content"] == "Page text content"
        assert result.data["format"] == "text"
        assert result.data["length"] == len("Page text content")

    def test_extract_html_whole_page(self, browser_available):
        browser_available.content.return_value = "<html><body>Hi</body></html>"

        tool = browser_extract
        result = tool.run(format="html")

        assert result.success is True
        assert result.data["content"] == "<html><body>Hi</body></html>"
        assert result.data["format"] == "html"

    def test_extract_text_with_selector(self, browser_available):
        element = MagicMock()
        element.inner_text.return_value = "Element text"
        browser_available.query_selector.return_value = element

        tool = browser_extract
        result = tool.run(selector="#content")

        assert result.success is True
        assert result.data["content"] == "Element text"

    def test_extract_html_with_selector(self, browser_available):
        element = MagicMock()
        element.inner_html.return_value = "<b>Bold</b>"
        browser_available.query_selector.return_value = element

        tool = browser_extract
        result = tool.run(selector="#content", format="html")

        assert result.success is True
        assert result.data["content"] == "<b>Bold</b>"

    def test_extract_selector_not_found(self, browser_available):
        browser_available.query_selector.return_value = None

        tool = browser_extract
        result = tool.run(selector="#missing")

        assert result.success is False
        assert "No element found" in result.error

    def test_extract_playwright_not_installed(self, browser_unavailable):
        tool = browser_extract
        result = tool.run()

        assert result.success is False
        assert "Playwright is not installed" in result.error


# ---------------------------------------------------------------------------
# browser_screenshot
# ---------------------------------------------------------------------------


class TestBrowserScreenshotTool:
    def test_screenshot_full_page(self, browser_available, tmp_path):
        # Create a fake screenshot file so os.path.getsize works
        out = tmp_path / "shot.png"
        out.write_bytes(b"\x89PNG" + b"\x00" * 100)

        tool = browser_screenshot
        result = tool.run(output_path=str(out), full_page=True)

        assert result.success is True
        assert result.data["saved_to"] == str(out)
        assert result.data["size_bytes"] == 104
        browser_available.screenshot.assert_called_once_with(path=str(out), full_page=True)

    def test_screenshot_element(self, browser_available, tmp_path):
        out = tmp_path / "elem.png"
        out.write_bytes(b"\x89PNG" + b"\x00" * 50)

        element = MagicMock()
        browser_available.query_selector.return_value = element

        tool = browser_screenshot
        result = tool.run(selector="#hero", output_path=str(out))

        assert result.success is True
        element.screenshot.assert_called_once_with(path=str(out))

    def test_screenshot_element_not_found(self, browser_available, tmp_path):
        browser_available.query_selector.return_value = None

        tool = browser_screenshot
        result = tool.run(selector="#missing", output_path=str(tmp_path / "x.png"))

        assert result.success is False
        assert "No element found" in result.error

    def test_screenshot_missing_output_path(self, browser_available):
        tool = browser_screenshot
        result = tool.run()

        assert result.success is False
        assert "output_path" in result.error.lower()

    def test_screenshot_playwright_not_installed(self, browser_unavailable):
        tool = browser_screenshot
        result = tool.run(output_path="/tmp/x.png")

        assert result.success is False
        assert "Playwright is not installed" in result.error


# ---------------------------------------------------------------------------
# browser_wait
# ---------------------------------------------------------------------------


class TestBrowserWaitTool:
    def test_wait_element_found(self, browser_available):
        tool = browser_wait
        result = tool.run(selector=".loaded")

        assert result.success is True
        assert result.data["found"] is True
        assert result.data["selector"] == ".loaded"
        assert result.data["state"] == "visible"
        browser_available.wait_for_selector.assert_called_once_with(
            ".loaded", timeout=5000, state="visible"
        )

    def test_wait_timeout(self, browser_available):
        browser_available.wait_for_selector.side_effect = Exception("Timeout 5000ms exceeded")

        tool = browser_wait
        result = tool.run(selector=".spinner", timeout=5000)

        assert result.success is False
        assert "Timeout" in result.error

    def test_wait_missing_selector(self, browser_available):
        tool = browser_wait
        result = tool.run()

        assert result.success is False
        assert "selector" in result.error.lower()

    def test_wait_playwright_not_installed(self, browser_unavailable):
        tool = browser_wait
        result = tool.run(selector=".x")

        assert result.success is False
        assert "Playwright is not installed" in result.error


# ---------------------------------------------------------------------------
# browser_eval
# ---------------------------------------------------------------------------


class TestBrowserEvalTool:
    def test_eval_returns_value(self, browser_available):
        browser_available.evaluate.return_value = 42

        tool = browser_eval
        result = tool.run(script="1 + 41")

        assert result.success is True
        assert result.data["result"] == 42

    def test_eval_returns_string(self, browser_available):
        browser_available.evaluate.return_value = "hello"

        tool = browser_eval
        result = tool.run(script="'hello'")

        assert result.success is True
        assert result.data["result"] == "hello"

    def test_eval_error(self, browser_available):
        browser_available.evaluate.side_effect = Exception("ReferenceError: foo is not defined")

        tool = browser_eval
        result = tool.run(script="foo.bar")

        assert result.success is False
        assert "ReferenceError" in result.error

    def test_eval_missing_script(self, browser_available):
        tool = browser_eval
        result = tool.run()

        assert result.success is False
        assert "script" in result.error.lower()

    def test_eval_playwright_not_installed(self, browser_unavailable):
        tool = browser_eval
        result = tool.run(script="1+1")

        assert result.success is False
        assert "Playwright is not installed" in result.error


# ---------------------------------------------------------------------------
# Session Manager
# ---------------------------------------------------------------------------


class TestBrowserSessionManager:
    def test_is_available_false_without_playwright(self):
        """Without mocking, Playwright may or may not be installed."""
        # Just verify the method runs without error
        result = BrowserSessionManager.is_available()
        assert isinstance(result, bool)

    def test_reset(self):
        BrowserSessionManager._page = "fake"
        BrowserSessionManager._browser = "fake"
        BrowserSessionManager.reset()
        assert BrowserSessionManager._page is None
        assert BrowserSessionManager._browser is None
        assert BrowserSessionManager._playwright is None
