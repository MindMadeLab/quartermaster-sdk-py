"""Tests for web search tools: DuckDuckGoSearchTool, WebScraperTool, JsonApiTool."""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from quartermaster_tools.builtin.web_search.duckduckgo import DuckDuckGoSearchTool
from quartermaster_tools.builtin.web_search.json_api import JsonApiTool
from quartermaster_tools.builtin.web_search.scraper import WebScraperTool
from quartermaster_tools.types import ToolResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_httpx() -> MagicMock:
    """Create a mock httpx module with required exception classes."""
    mock_httpx = MagicMock()
    mock_httpx.TimeoutException = type("TimeoutException", (Exception,), {})
    mock_httpx.ConnectError = type("ConnectError", (Exception,), {})
    mock_httpx.HTTPError = type("HTTPError", (Exception,), {})
    mock_httpx.HTTPStatusError = type(
        "HTTPStatusError",
        (Exception,),
        {"__init__": lambda self, msg, *, response: (setattr(self, "response", response) or None)},
    )
    return mock_httpx


def _make_mock_response(
    text: str = "",
    status_code: int = 200,
    headers: dict | None = None,
    content: bytes | None = None,
    url: str = "https://example.com",
    json_data: object = None,
) -> MagicMock:
    """Create a mock httpx response."""
    resp = MagicMock()
    resp.text = text
    resp.status_code = status_code
    resp.headers = headers or {}
    resp.content = content if content is not None else text.encode("utf-8")
    resp.url = url
    if json_data is not None:
        resp.json.return_value = json_data
    else:
        resp.json.side_effect = json.JSONDecodeError("No JSON", "", 0)
    resp.raise_for_status = MagicMock()
    return resp


def _mock_client(mock_httpx: MagicMock, response: MagicMock) -> MagicMock:
    """Wire up a mock httpx.Client context manager returning the given response."""
    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    client.get.return_value = response
    client.post.return_value = response
    client.put.return_value = response
    client.delete.return_value = response
    client.patch.return_value = response
    client.request.return_value = response
    mock_httpx.Client.return_value = client
    return client


# ---------------------------------------------------------------------------
# DuckDuckGoSearchTool
# ---------------------------------------------------------------------------


class TestDuckDuckGoSearchTool:
    """Tests for DuckDuckGoSearchTool."""

    def test_name_and_version(self) -> None:
        tool = DuckDuckGoSearchTool()
        assert tool.name() == "duckduckgo_search"
        assert tool.version() == "1.0.0"

    def test_info_returns_descriptor(self) -> None:
        tool = DuckDuckGoSearchTool()
        info = tool.info()
        assert info.name == "duckduckgo_search"
        assert info.is_local is False

    def test_missing_query(self) -> None:
        tool = DuckDuckGoSearchTool()
        result = tool.run()
        assert result.success is False
        assert "required" in result.error.lower()

    def test_empty_query(self) -> None:
        tool = DuckDuckGoSearchTool()
        result = tool.run(query="   ")
        assert result.success is False
        assert "required" in result.error.lower()

    def test_successful_search(self) -> None:
        html_body = """
        <div class="result results_links results_links_deep web-result">
            <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage1">
                Example <b>Page</b> One
            </a>
            <a class="result__snippet" href="#">This is the first result snippet.</a>
        </div>
        </div>
        <div class="result results_links results_links_deep web-result">
            <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage2">
                Page Two Title
            </a>
            <a class="result__snippet" href="#">Second result snippet here.</a>
        </div>
        </div>
        """
        mock_httpx = _make_mock_httpx()
        mock_response = _make_mock_response(text=html_body)
        _mock_client(mock_httpx, mock_response)

        sys.modules["httpx"] = mock_httpx
        try:
            tool = DuckDuckGoSearchTool()
            result = tool.run(query="test query")

            assert result.success is True
            assert result.data["query"] == "test query"
            assert result.data["result_count"] == 2
            results = result.data["results"]
            assert results[0]["title"] == "Example Page One"
            assert results[0]["url"] == "https://example.com/page1"
            assert "first result" in results[0]["snippet"]
        finally:
            sys.modules.pop("httpx", None)

    def test_max_results_limit(self) -> None:
        # Generate 5 results but request only 2
        blocks = ""
        for i in range(5):
            blocks += f"""
            <div class="result results_links results_links_deep web-result">
                <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2F{i}">
                    Result {i}
                </a>
                <a class="result__snippet" href="#">Snippet {i}.</a>
            </div>
            </div>
            """
        mock_httpx = _make_mock_httpx()
        mock_response = _make_mock_response(text=blocks)
        _mock_client(mock_httpx, mock_response)

        sys.modules["httpx"] = mock_httpx
        try:
            tool = DuckDuckGoSearchTool()
            result = tool.run(query="test", max_results=2)

            assert result.success is True
            assert result.data["result_count"] == 2
        finally:
            sys.modules.pop("httpx", None)

    def test_search_timeout(self) -> None:
        mock_httpx = _make_mock_httpx()
        client = MagicMock()
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__ = MagicMock(return_value=False)
        client.post.side_effect = mock_httpx.TimeoutException("timeout")
        mock_httpx.Client.return_value = client

        sys.modules["httpx"] = mock_httpx
        try:
            tool = DuckDuckGoSearchTool()
            result = tool.run(query="slow query")

            assert result.success is False
            assert "timed out" in result.error.lower()
        finally:
            sys.modules.pop("httpx", None)

    def test_search_convenience_method(self) -> None:
        html_body = """
        <div class="result results_links results_links_deep web-result">
            <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com">
                Title
            </a>
            <a class="result__snippet" href="#">Snippet.</a>
        </div>
        </div>
        """
        mock_httpx = _make_mock_httpx()
        mock_response = _make_mock_response(text=html_body)
        _mock_client(mock_httpx, mock_response)

        sys.modules["httpx"] = mock_httpx
        try:
            tool = DuckDuckGoSearchTool()
            result = tool.search("test", max_results=3)
            assert result.success is True
        finally:
            sys.modules.pop("httpx", None)

    def test_no_results_found(self) -> None:
        mock_httpx = _make_mock_httpx()
        mock_response = _make_mock_response(text="<html><body>No results</body></html>")
        _mock_client(mock_httpx, mock_response)

        sys.modules["httpx"] = mock_httpx
        try:
            tool = DuckDuckGoSearchTool()
            result = tool.run(query="xyznonexistent123")

            assert result.success is True
            assert result.data["result_count"] == 0
            assert result.data["results"] == []
        finally:
            sys.modules.pop("httpx", None)

    def test_httpx_not_installed(self) -> None:
        httpx_mod = sys.modules.get("httpx")
        sys.modules["httpx"] = None  # type: ignore[assignment]
        try:
            tool = DuckDuckGoSearchTool()
            result = tool.run(query="test")
            assert result.success is False
            assert "httpx is required" in result.error
        finally:
            if httpx_mod is not None:
                sys.modules["httpx"] = httpx_mod
            else:
                sys.modules.pop("httpx", None)


# ---------------------------------------------------------------------------
# WebScraperTool
# ---------------------------------------------------------------------------


class TestWebScraperTool:
    """Tests for WebScraperTool."""

    def test_name_and_version(self) -> None:
        tool = WebScraperTool()
        assert tool.name() == "web_scraper"
        assert tool.version() == "1.0.0"

    def test_info_returns_descriptor(self) -> None:
        tool = WebScraperTool()
        info = tool.info()
        assert info.name == "web_scraper"
        assert info.is_local is False

    def test_missing_url(self) -> None:
        tool = WebScraperTool()
        result = tool.run()
        assert result.success is False
        assert "required" in result.error.lower()

    def test_invalid_output_format(self) -> None:
        tool = WebScraperTool()
        result = tool.run(url="https://example.com", output_format="pdf")
        assert result.success is False
        assert "invalid output_format" in result.error.lower()

    def test_scrape_text_format(self) -> None:
        html = "<html><body><h1>Hello</h1><p>World &amp; friends</p><script>evil()</script></body></html>"
        mock_httpx = _make_mock_httpx()
        mock_response = _make_mock_response(text=html)
        _mock_client(mock_httpx, mock_response)

        sys.modules["httpx"] = mock_httpx
        try:
            tool = WebScraperTool()
            result = tool.run(url="https://example.com", output_format="text")

            assert result.success is True
            content = result.data["content"]
            assert "Hello" in content
            assert "World & friends" in content
            assert "evil()" not in content
        finally:
            sys.modules.pop("httpx", None)

    def test_scrape_markdown_format(self) -> None:
        html = '<html><body><h2>Title</h2><p>A <b>bold</b> word and <a href="https://link.com">a link</a>.</p></body></html>'
        mock_httpx = _make_mock_httpx()
        mock_response = _make_mock_response(text=html)
        _mock_client(mock_httpx, mock_response)

        sys.modules["httpx"] = mock_httpx
        try:
            tool = WebScraperTool()
            result = tool.run(url="https://example.com", output_format="markdown")

            assert result.success is True
            content = result.data["content"]
            assert "## Title" in content
            assert "**bold**" in content
            assert "[a link](https://link.com)" in content
        finally:
            sys.modules.pop("httpx", None)

    def test_scrape_html_format(self) -> None:
        html = "<html><body><p>Raw HTML</p></body></html>"
        mock_httpx = _make_mock_httpx()
        mock_response = _make_mock_response(text=html)
        _mock_client(mock_httpx, mock_response)

        sys.modules["httpx"] = mock_httpx
        try:
            tool = WebScraperTool()
            result = tool.run(url="https://example.com", output_format="html")

            assert result.success is True
            assert result.data["content"] == html
        finally:
            sys.modules.pop("httpx", None)

    def test_scrape_timeout(self) -> None:
        mock_httpx = _make_mock_httpx()
        client = MagicMock()
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__ = MagicMock(return_value=False)
        client.get.side_effect = mock_httpx.TimeoutException("timeout")
        mock_httpx.Client.return_value = client

        sys.modules["httpx"] = mock_httpx
        try:
            tool = WebScraperTool()
            result = tool.run(url="https://example.com/slow")

            assert result.success is False
            assert "timed out" in result.error.lower()
        finally:
            sys.modules.pop("httpx", None)

    def test_scrape_convenience_method(self) -> None:
        html = "<html><body><p>Content</p></body></html>"
        mock_httpx = _make_mock_httpx()
        mock_response = _make_mock_response(text=html)
        _mock_client(mock_httpx, mock_response)

        sys.modules["httpx"] = mock_httpx
        try:
            tool = WebScraperTool()
            result = tool.scrape("https://example.com", output_format="text", timeout=10)
            assert result.success is True
            assert "Content" in result.data["content"]
        finally:
            sys.modules.pop("httpx", None)

    def test_httpx_not_installed(self) -> None:
        httpx_mod = sys.modules.get("httpx")
        sys.modules["httpx"] = None  # type: ignore[assignment]
        try:
            tool = WebScraperTool()
            result = tool.run(url="https://example.com")
            assert result.success is False
            assert "httpx is required" in result.error
        finally:
            if httpx_mod is not None:
                sys.modules["httpx"] = httpx_mod
            else:
                sys.modules.pop("httpx", None)


# ---------------------------------------------------------------------------
# JsonApiTool
# ---------------------------------------------------------------------------


class TestJsonApiTool:
    """Tests for JsonApiTool."""

    def test_name_and_version(self) -> None:
        tool = JsonApiTool()
        assert tool.name() == "json_api"
        assert tool.version() == "1.0.0"

    def test_info_returns_descriptor(self) -> None:
        tool = JsonApiTool()
        info = tool.info()
        assert info.name == "json_api"
        assert info.is_local is False

    def test_missing_url(self) -> None:
        tool = JsonApiTool()
        result = tool.run()
        assert result.success is False
        assert "required" in result.error.lower()

    def test_unsupported_method(self) -> None:
        tool = JsonApiTool()
        result = tool.run(url="https://api.example.com", method="OPTIONS")
        assert result.success is False
        assert "unsupported" in result.error.lower()

    def test_get_json_success(self) -> None:
        json_data = {"users": [{"id": 1, "name": "Alice"}]}
        mock_httpx = _make_mock_httpx()
        mock_response = _make_mock_response(
            text=json.dumps(json_data),
            status_code=200,
            headers={"content-type": "application/json"},
            json_data=json_data,
        )
        client = _mock_client(mock_httpx, mock_response)

        sys.modules["httpx"] = mock_httpx
        try:
            tool = JsonApiTool()
            result = tool.run(url="https://api.example.com/users")

            assert result.success is True
            assert result.data["json"] == json_data
            assert result.data["status_code"] == 200
            client.request.assert_called_once()
        finally:
            sys.modules.pop("httpx", None)

    def test_post_with_json_body(self) -> None:
        req_body = {"name": "Bob", "email": "bob@example.com"}
        resp_data = {"id": 2, "name": "Bob"}
        mock_httpx = _make_mock_httpx()
        mock_response = _make_mock_response(
            text=json.dumps(resp_data),
            status_code=201,
            json_data=resp_data,
        )
        client = _mock_client(mock_httpx, mock_response)

        sys.modules["httpx"] = mock_httpx
        try:
            tool = JsonApiTool()
            result = tool.run(
                url="https://api.example.com/users",
                method="POST",
                body=req_body,
            )

            assert result.success is True
            assert result.data["json"]["id"] == 2
            # Verify JSON serialization of body
            call_kwargs = client.request.call_args
            assert call_kwargs.kwargs["content"] == json.dumps(req_body)
        finally:
            sys.modules.pop("httpx", None)

    def test_invalid_json_response(self) -> None:
        mock_httpx = _make_mock_httpx()
        mock_response = _make_mock_response(text="Not JSON at all")
        _mock_client(mock_httpx, mock_response)

        sys.modules["httpx"] = mock_httpx
        try:
            tool = JsonApiTool()
            result = tool.run(url="https://api.example.com/broken")

            assert result.success is False
            assert "json" in result.error.lower()
        finally:
            sys.modules.pop("httpx", None)

    def test_api_timeout(self) -> None:
        mock_httpx = _make_mock_httpx()
        client = MagicMock()
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__ = MagicMock(return_value=False)
        client.request.side_effect = mock_httpx.TimeoutException("timeout")
        mock_httpx.Client.return_value = client

        sys.modules["httpx"] = mock_httpx
        try:
            tool = JsonApiTool()
            result = tool.run(url="https://api.example.com/slow")

            assert result.success is False
            assert "timed out" in result.error.lower()
        finally:
            sys.modules.pop("httpx", None)

    def test_jmespath_filter_missing_library(self) -> None:
        json_data = {"items": [1, 2, 3]}
        mock_httpx = _make_mock_httpx()
        mock_response = _make_mock_response(
            text=json.dumps(json_data),
            json_data=json_data,
        )
        _mock_client(mock_httpx, mock_response)

        sys.modules["httpx"] = mock_httpx
        # Temporarily hide jmespath
        jmespath_mod = sys.modules.get("jmespath")
        sys.modules["jmespath"] = None  # type: ignore[assignment]
        try:
            tool = JsonApiTool()
            result = tool.run(
                url="https://api.example.com/data",
                jmespath_filter="items[0]",
            )

            assert result.success is False
            assert "jmespath" in result.error.lower()
        finally:
            sys.modules.pop("httpx", None)
            if jmespath_mod is not None:
                sys.modules["jmespath"] = jmespath_mod
            else:
                sys.modules.pop("jmespath", None)

    def test_call_api_convenience_method(self) -> None:
        json_data = {"status": "ok"}
        mock_httpx = _make_mock_httpx()
        mock_response = _make_mock_response(
            text=json.dumps(json_data),
            json_data=json_data,
        )
        _mock_client(mock_httpx, mock_response)

        sys.modules["httpx"] = mock_httpx
        try:
            tool = JsonApiTool()
            result = tool.call_api(
                url="https://api.example.com/health",
                method="GET",
                headers={"Authorization": "Bearer token123"},
            )

            assert result.success is True
            assert result.data["json"]["status"] == "ok"
        finally:
            sys.modules.pop("httpx", None)

    def test_httpx_not_installed(self) -> None:
        httpx_mod = sys.modules.get("httpx")
        sys.modules["httpx"] = None  # type: ignore[assignment]
        try:
            tool = JsonApiTool()
            result = tool.run(url="https://api.example.com")
            assert result.success is False
            assert "httpx is required" in result.error
        finally:
            if httpx_mod is not None:
                sys.modules["httpx"] = httpx_mod
            else:
                sys.modules.pop("httpx", None)

    def test_custom_headers_merged(self) -> None:
        json_data = {"ok": True}
        mock_httpx = _make_mock_httpx()
        mock_response = _make_mock_response(
            text=json.dumps(json_data),
            json_data=json_data,
        )
        client = _mock_client(mock_httpx, mock_response)

        sys.modules["httpx"] = mock_httpx
        try:
            tool = JsonApiTool()
            result = tool.run(
                url="https://api.example.com",
                headers={"X-Custom": "value"},
            )

            assert result.success is True
            call_kwargs = client.request.call_args
            passed_headers = call_kwargs.kwargs["headers"]
            assert passed_headers["Accept"] == "application/json"
            assert passed_headers["X-Custom"] == "value"
        finally:
            sys.modules.pop("httpx", None)
