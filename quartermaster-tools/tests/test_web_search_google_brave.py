"""Tests for GoogleSearchTool and BraveSearchTool."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from quartermaster_tools.builtin.web_search.google import GoogleSearchTool
from quartermaster_tools.builtin.web_search.brave import BraveSearchTool


# --- GoogleSearchTool Tests ---


class TestGoogleSearchTool:
    def setup_method(self) -> None:
        self.tool = GoogleSearchTool()

    def test_name(self) -> None:
        assert self.tool.name() == "google_search"

    def test_version(self) -> None:
        assert self.tool.version() == "1.0.0"

    def test_parameters_list(self) -> None:
        params = self.tool.parameters()
        names = [p.name for p in params]
        assert "query" in names
        assert "num_results" in names
        assert "language" in names
        assert "region" in names

    def test_info_descriptor(self) -> None:
        info = self.tool.info()
        assert info.name == "google_search"
        assert info.version == "1.0.0"
        assert info.is_local is False

    def test_empty_query_returns_error(self) -> None:
        result = self.tool.run(query="")
        assert result.success is False
        assert "query" in result.error.lower()

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_api_key_returns_error(self) -> None:
        result = self.tool.run(query="test")
        assert result.success is False
        assert "GOOGLE_API_KEY" in result.error

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "key123", "GOOGLE_CSE_ID": "cse123"})
    @patch("quartermaster_tools.builtin.web_search.google.httpx")
    def test_successful_search(self, mock_httpx: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "items": [
                {"title": "Result 1", "link": "https://example.com/1", "snippet": "Snippet 1"},
                {"title": "Result 2", "link": "https://example.com/2", "snippet": "Snippet 2"},
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_httpx.Client.return_value = mock_client

        result = self.tool.run(query="test query", num_results=2)
        assert result.success is True
        assert result.data["result_count"] == 2
        assert result.data["results"][0]["title"] == "Result 1"
        assert result.data["results"][0]["url"] == "https://example.com/1"

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "key123", "GOOGLE_CSE_ID": "cse123"})
    @patch("quartermaster_tools.builtin.web_search.google.httpx")
    def test_timeout_error(self, mock_httpx: MagicMock) -> None:
        import httpx
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = httpx.TimeoutException("timeout")
        mock_httpx.Client.return_value = mock_client
        mock_httpx.TimeoutException = httpx.TimeoutException
        mock_httpx.HTTPStatusError = httpx.HTTPStatusError
        mock_httpx.HTTPError = httpx.HTTPError

        result = self.tool.run(query="test")
        assert result.success is False
        assert "timed out" in result.error.lower()

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "key123", "GOOGLE_CSE_ID": "cse123"})
    @patch("quartermaster_tools.builtin.web_search.google.httpx")
    def test_http_error(self, mock_httpx: MagicMock) -> None:
        import httpx
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.text = "Forbidden"
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=mock_response
        )
        mock_httpx.Client.return_value = mock_client
        mock_httpx.TimeoutException = httpx.TimeoutException
        mock_httpx.HTTPStatusError = httpx.HTTPStatusError
        mock_httpx.HTTPError = httpx.HTTPError

        result = self.tool.run(query="test")
        assert result.success is False
        assert "403" in result.error

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "key123", "GOOGLE_CSE_ID": "cse123"})
    @patch("quartermaster_tools.builtin.web_search.google.httpx")
    def test_language_and_region_params(self, mock_httpx: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {"items": []}
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_httpx.Client.return_value = mock_client

        result = self.tool.run(query="test", language="en", region="us")
        assert result.success is True
        # Verify params were passed
        call_kwargs = mock_client.get.call_args
        params = call_kwargs[1]["params"] if "params" in call_kwargs[1] else call_kwargs.kwargs["params"]
        assert params["lr"] == "lang_en"
        assert params["gl"] == "us"

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "key123", "GOOGLE_CSE_ID": "cse123"})
    @patch("quartermaster_tools.builtin.web_search.google.httpx")
    def test_num_results_clamped(self, mock_httpx: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {"items": []}
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_httpx.Client.return_value = mock_client

        self.tool.run(query="test", num_results=50)
        call_kwargs = mock_client.get.call_args
        params = call_kwargs[1]["params"] if "params" in call_kwargs[1] else call_kwargs.kwargs["params"]
        assert params["num"] == 10  # clamped to max


# --- BraveSearchTool Tests ---


class TestBraveSearchTool:
    def setup_method(self) -> None:
        self.tool = BraveSearchTool()

    def test_name(self) -> None:
        assert self.tool.name() == "brave_search"

    def test_version(self) -> None:
        assert self.tool.version() == "1.0.0"

    def test_parameters_list(self) -> None:
        params = self.tool.parameters()
        names = [p.name for p in params]
        assert "query" in names
        assert "count" in names
        assert "country" in names
        assert "freshness" in names

    def test_info_descriptor(self) -> None:
        info = self.tool.info()
        assert info.name == "brave_search"

    def test_empty_query_returns_error(self) -> None:
        result = self.tool.run(query="")
        assert result.success is False
        assert "query" in result.error.lower()

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_api_key_returns_error(self) -> None:
        result = self.tool.run(query="test")
        assert result.success is False
        assert "BRAVE_API_KEY" in result.error

    @patch.dict("os.environ", {"BRAVE_API_KEY": "brave_key"})
    @patch("quartermaster_tools.builtin.web_search.brave.httpx")
    def test_successful_search(self, mock_httpx: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {"title": "Brave Result", "url": "https://brave.com", "description": "A snippet"},
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_httpx.Client.return_value = mock_client

        result = self.tool.run(query="test")
        assert result.success is True
        assert result.data["result_count"] == 1
        assert result.data["results"][0]["snippet"] == "A snippet"

    @patch.dict("os.environ", {"BRAVE_API_KEY": "brave_key"})
    @patch("quartermaster_tools.builtin.web_search.brave.httpx")
    def test_timeout_error(self, mock_httpx: MagicMock) -> None:
        import httpx
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = httpx.TimeoutException("timeout")
        mock_httpx.Client.return_value = mock_client
        mock_httpx.TimeoutException = httpx.TimeoutException
        mock_httpx.HTTPStatusError = httpx.HTTPStatusError
        mock_httpx.HTTPError = httpx.HTTPError

        result = self.tool.run(query="test")
        assert result.success is False
        assert "timed out" in result.error.lower()

    @patch.dict("os.environ", {"BRAVE_API_KEY": "brave_key"})
    @patch("quartermaster_tools.builtin.web_search.brave.httpx")
    def test_freshness_param(self, mock_httpx: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {"web": {"results": []}}
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_httpx.Client.return_value = mock_client

        result = self.tool.run(query="test", freshness="week")
        assert result.success is True
        call_kwargs = mock_client.get.call_args
        params = call_kwargs[1]["params"] if "params" in call_kwargs[1] else call_kwargs.kwargs["params"]
        assert params["freshness"] == "week"

    @patch.dict("os.environ", {"BRAVE_API_KEY": "brave_key"})
    @patch("quartermaster_tools.builtin.web_search.brave.httpx")
    def test_count_clamped(self, mock_httpx: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {"web": {"results": []}}
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_httpx.Client.return_value = mock_client

        self.tool.run(query="test", count=100)
        call_kwargs = mock_client.get.call_args
        params = call_kwargs[1]["params"] if "params" in call_kwargs[1] else call_kwargs.kwargs["params"]
        assert params["count"] == 20  # clamped
