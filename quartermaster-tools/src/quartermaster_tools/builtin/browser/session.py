"""
Browser session manager for Playwright-based tools.

Provides a singleton that manages the browser lifecycle (launch, page access,
close). Tools reference this manager to obtain the current Playwright Page.
"""

from __future__ import annotations

from typing import Any


class BrowserSessionManager:
    """Singleton manager for a Playwright browser session.

    Lazily launches a headless Chromium browser on first access and reuses
    the same page across tool invocations.
    """

    _instance: BrowserSessionManager | None = None
    _browser: Any = None
    _page: Any = None
    _playwright: Any = None

    @classmethod
    def is_available(cls) -> bool:
        """Check whether Playwright is installed."""
        try:
            import playwright.sync_api  # noqa: F401
            return True
        except ImportError:
            return False

    @classmethod
    def get_page(cls) -> Any:
        """Return the current Playwright Page, launching the browser if needed.

        Returns:
            A ``playwright.sync_api.Page`` instance.

        Raises:
            ImportError: If Playwright is not installed.
            RuntimeError: If the browser cannot be launched.
        """
        if cls._page is not None:
            return cls._page

        from playwright.sync_api import sync_playwright

        cls._playwright = sync_playwright().start()
        cls._browser = cls._playwright.chromium.launch(headless=True)
        cls._page = cls._browser.new_page()
        return cls._page

    @classmethod
    def close(cls) -> None:
        """Close the browser and clean up resources."""
        if cls._browser is not None:
            cls._browser.close()
            cls._browser = None
        if cls._playwright is not None:
            cls._playwright.stop()
            cls._playwright = None
        cls._page = None

    @classmethod
    def reset(cls) -> None:
        """Reset internal state without closing (useful for testing)."""
        cls._browser = None
        cls._page = None
        cls._playwright = None
