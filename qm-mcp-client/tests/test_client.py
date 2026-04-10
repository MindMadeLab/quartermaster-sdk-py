"""Tests for McpClient: initialization, context managers, and API methods."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from qm_mcp_client import McpClient, McpServerInfo, McpTool
from qm_mcp_client.errors import (
    McpConnectionError,
    McpServerError,
    McpToolNotFoundError,
)


# === Client initialization ===


class TestClientInit:
    def test_valid_http_url(self) -> None:
        client = McpClient("http://localhost:8000/mcp")
        assert client._url == "http://localhost:8000/mcp"
        assert client._transport_kind == "sse"

    def test_valid_https_url(self) -> None:
        client = McpClient("https://example.com/mcp")
        assert client._url == "https://example.com/mcp"

    def test_trailing_slash_stripped(self) -> None:
        client = McpClient("http://localhost:8000/mcp/")
        assert client._url == "http://localhost:8000/mcp"

    def test_invalid_url_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid URL"):
            McpClient("not-a-url")

    def test_ftp_url_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid URL"):
            McpClient("ftp://example.com/mcp")

    def test_invalid_transport_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid transport"):
            McpClient("http://localhost:8000/mcp", transport="websocket")

    def test_valid_transport_sse(self) -> None:
        client = McpClient("http://localhost:8000/mcp", transport="sse")
        assert client._transport_kind == "sse"

    def test_valid_transport_streamable(self) -> None:
        client = McpClient("http://localhost:8000/mcp", transport="streamable")
        assert client._transport_kind == "streamable"

    def test_custom_timeout(self) -> None:
        client = McpClient("http://localhost:8000/mcp", timeout=60.0)
        assert client._timeout == 60.0

    def test_auth_token(self) -> None:
        client = McpClient("http://localhost:8000/mcp", auth_token="secret")
        assert client._auth_token == "secret"

    def test_custom_headers(self) -> None:
        client = McpClient(
            "http://localhost:8000/mcp",
            headers={"X-Custom": "value"},
        )
        assert client._extra_headers == {"X-Custom": "value"}

    def test_not_connected_initially(self) -> None:
        client = McpClient("http://localhost:8000/mcp")
        assert client.is_connected is False


# === Context managers ===


class TestAsyncContextManager:
    @pytest.mark.asyncio
    async def test_enter_exit(self) -> None:
        async with McpClient("http://localhost:8000/mcp") as client:
            assert client.is_connected is True
        assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_exit_on_exception(self) -> None:
        try:
            async with McpClient("http://localhost:8000/mcp") as client:
                assert client.is_connected is True
                raise RuntimeError("test")
        except RuntimeError:
            pass
        assert client.is_connected is False


class TestSyncContextManager:
    def test_enter_exit(self) -> None:
        with McpClient("http://localhost:8000/mcp") as client:
            assert client.is_connected is True
        assert client.is_connected is False

    def test_exit_on_exception(self) -> None:
        try:
            with McpClient("http://localhost:8000/mcp") as client:
                assert client.is_connected is True
                raise RuntimeError("test")
        except RuntimeError:
            pass
        assert client.is_connected is False


# === Async API methods (with mocked transport) ===


class TestServerInfo:
    @respx.mock
    @pytest.mark.asyncio
    async def test_server_info(
        self, sample_server_info_response: dict[str, Any]
    ) -> None:
        url = "http://localhost:8000/mcp"
        respx.post(url).mock(
            return_value=httpx.Response(200, json=sample_server_info_response)
        )

        async with McpClient(url) as client:
            info = await client.server_info()

        assert isinstance(info, McpServerInfo)
        assert info.name == "test-server"
        assert info.version == "1.0.0"
        assert info.protocol_version == "2024-11-05"
        assert "tools" in info.capabilities


class TestListTools:
    @respx.mock
    @pytest.mark.asyncio
    async def test_list_tools(self, sample_tools_response: dict[str, Any]) -> None:
        url = "http://localhost:8000/mcp"
        respx.post(url).mock(
            return_value=httpx.Response(200, json=sample_tools_response)
        )

        async with McpClient(url) as client:
            tools = await client.list_tools()

        assert len(tools) == 2
        assert all(isinstance(t, McpTool) for t in tools)

        weather = tools[0]
        assert weather.name == "weather"
        assert weather.description == "Get weather for a location"
        assert len(weather.parameters) == 2

        location_param = weather.parameters[0]
        assert location_param.name == "location"
        assert location_param.type == "string"
        assert location_param.required is True

        units_param = weather.parameters[1]
        assert units_param.name == "units"
        assert units_param.enum == ["celsius", "fahrenheit"]
        assert units_param.default == "celsius"

    @respx.mock
    @pytest.mark.asyncio
    async def test_empty_tools(self) -> None:
        url = "http://localhost:8000/mcp"
        respx.post(url).mock(
            return_value=httpx.Response(
                200,
                json={"jsonrpc": "2.0", "id": 1, "result": {"tools": []}},
            )
        )

        async with McpClient(url) as client:
            tools = await client.list_tools()
        assert tools == []


class TestCallTool:
    @respx.mock
    @pytest.mark.asyncio
    async def test_call_tool_text_result(
        self, sample_tool_call_response: dict[str, Any]
    ) -> None:
        url = "http://localhost:8000/mcp"
        respx.post(url).mock(
            return_value=httpx.Response(200, json=sample_tool_call_response)
        )

        async with McpClient(url) as client:
            result = await client.call_tool("weather", {"location": "SF"})

        assert result == "Sunny, 22°C in San Francisco"

    @respx.mock
    @pytest.mark.asyncio
    async def test_call_tool_multi_content(self) -> None:
        url = "http://localhost:8000/mcp"
        response = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [
                    {"type": "text", "text": "Part 1"},
                    {"type": "text", "text": "Part 2"},
                ]
            },
        }
        respx.post(url).mock(return_value=httpx.Response(200, json=response))

        async with McpClient(url) as client:
            result = await client.call_tool("multi", {})

        assert isinstance(result, list)
        assert len(result) == 2

    @respx.mock
    @pytest.mark.asyncio
    async def test_call_tool_empty_content(self) -> None:
        url = "http://localhost:8000/mcp"
        response = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"content": []},
        }
        respx.post(url).mock(return_value=httpx.Response(200, json=response))

        async with McpClient(url) as client:
            result = await client.call_tool("empty", {})
        # Empty content returns result dict
        assert result == {"content": []}

    @respx.mock
    @pytest.mark.asyncio
    async def test_call_tool_not_found(
        self, jsonrpc_error_method_not_found: dict[str, Any]
    ) -> None:
        url = "http://localhost:8000/mcp"
        respx.post(url).mock(
            return_value=httpx.Response(200, json=jsonrpc_error_method_not_found)
        )

        async with McpClient(url, max_retries=1) as client:
            with pytest.raises(McpToolNotFoundError, match="not found"):
                await client.call_tool("nonexistent", {})

    @respx.mock
    @pytest.mark.asyncio
    async def test_call_tool_server_error(
        self, jsonrpc_error_internal: dict[str, Any]
    ) -> None:
        url = "http://localhost:8000/mcp"
        respx.post(url).mock(
            return_value=httpx.Response(200, json=jsonrpc_error_internal)
        )

        async with McpClient(url, max_retries=1) as client:
            with pytest.raises(McpServerError, match="Internal server error"):
                await client.call_tool("broken_tool", {})


class TestListResources:
    @respx.mock
    @pytest.mark.asyncio
    async def test_list_resources(
        self, sample_resources_response: dict[str, Any]
    ) -> None:
        url = "http://localhost:8000/mcp"
        respx.post(url).mock(
            return_value=httpx.Response(200, json=sample_resources_response)
        )

        async with McpClient(url) as client:
            resources = await client.list_resources()

        assert len(resources) == 2
        assert resources[0]["uri"] == "file:///data/config.json"
        assert resources[1]["name"] == "README"


class TestReadResource:
    @respx.mock
    @pytest.mark.asyncio
    async def test_read_resource(
        self, sample_resource_read_response: dict[str, Any]
    ) -> None:
        url = "http://localhost:8000/mcp"
        respx.post(url).mock(
            return_value=httpx.Response(200, json=sample_resource_read_response)
        )

        async with McpClient(url) as client:
            content = await client.read_resource("file:///data/config.json")

        assert content == '{"key": "value"}'

    @respx.mock
    @pytest.mark.asyncio
    async def test_read_resource_empty(self) -> None:
        url = "http://localhost:8000/mcp"
        response = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"contents": []},
        }
        respx.post(url).mock(return_value=httpx.Response(200, json=response))

        async with McpClient(url) as client:
            content = await client.read_resource("file:///missing")
        assert content == ""


# === Retry logic ===


class TestRetryLogic:
    @respx.mock
    @pytest.mark.asyncio
    async def test_retry_on_connection_error(self) -> None:
        url = "http://localhost:8000/mcp"
        call_count = 0

        def side_effect(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.ConnectError("Connection refused")
            return httpx.Response(
                200,
                json={"jsonrpc": "2.0", "id": 1, "result": {"ok": True}},
            )

        respx.post(url).mock(side_effect=side_effect)

        async with McpClient(url, max_retries=3) as client:
            result = await client._send_request("test")
        assert result == {"ok": True}
        assert call_count == 3

    @respx.mock
    @pytest.mark.asyncio
    async def test_retry_exhausted(self) -> None:
        url = "http://localhost:8000/mcp"

        respx.post(url).mock(side_effect=httpx.ConnectError("refused"))

        async with McpClient(url, max_retries=2) as client:
            with pytest.raises(McpConnectionError):
                await client._send_request("test")

    @respx.mock
    @pytest.mark.asyncio
    async def test_no_retry_on_protocol_error(self) -> None:
        url = "http://localhost:8000/mcp"
        call_count = 0

        def side_effect(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(
                200,
                json={"jsonrpc": "1.0", "id": 1, "result": {}},
            )

        respx.post(url).mock(side_effect=side_effect)

        async with McpClient(url, max_retries=3) as client:
            with pytest.raises(Exception):
                await client._send_request("test")
        # Protocol errors are not retried
        assert call_count == 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_no_retry_on_server_error(self) -> None:
        url = "http://localhost:8000/mcp"
        call_count = 0

        def side_effect(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "error": {"code": -32603, "message": "error"},
                },
            )

        respx.post(url).mock(side_effect=side_effect)

        async with McpClient(url, max_retries=3) as client:
            with pytest.raises(McpServerError):
                await client._send_request("test")
        assert call_count == 1


# === Not connected errors ===


class TestNotConnected:
    @pytest.mark.asyncio
    async def test_send_request_without_context_manager(self) -> None:
        client = McpClient("http://localhost:8000/mcp")
        with pytest.raises(McpConnectionError, match="not connected"):
            await client._send_request("test")


# === Sync wrappers ===


class TestSyncWrappers:
    @respx.mock
    def test_list_tools_sync(self, sample_tools_response: dict[str, Any]) -> None:
        url = "http://localhost:8000/mcp"
        respx.post(url).mock(
            return_value=httpx.Response(200, json=sample_tools_response)
        )

        with McpClient(url) as client:
            tools = client.list_tools_sync()

        assert len(tools) == 2
        assert tools[0].name == "weather"

    @respx.mock
    def test_call_tool_sync(self, sample_tool_call_response: dict[str, Any]) -> None:
        url = "http://localhost:8000/mcp"
        respx.post(url).mock(
            return_value=httpx.Response(200, json=sample_tool_call_response)
        )

        with McpClient(url) as client:
            result = client.call_tool_sync("weather", {"location": "SF"})

        assert result == "Sunny, 22°C in San Francisco"

    @respx.mock
    def test_server_info_sync(
        self, sample_server_info_response: dict[str, Any]
    ) -> None:
        url = "http://localhost:8000/mcp"
        respx.post(url).mock(
            return_value=httpx.Response(200, json=sample_server_info_response)
        )

        with McpClient(url) as client:
            info = client.server_info_sync()

        assert info.name == "test-server"

    @respx.mock
    def test_list_resources_sync(
        self, sample_resources_response: dict[str, Any]
    ) -> None:
        url = "http://localhost:8000/mcp"
        respx.post(url).mock(
            return_value=httpx.Response(200, json=sample_resources_response)
        )

        with McpClient(url) as client:
            resources = client.list_resources_sync()

        assert len(resources) == 2

    @respx.mock
    def test_read_resource_sync(
        self, sample_resource_read_response: dict[str, Any]
    ) -> None:
        url = "http://localhost:8000/mcp"
        respx.post(url).mock(
            return_value=httpx.Response(200, json=sample_resource_read_response)
        )

        with McpClient(url) as client:
            content = client.read_resource_sync("file:///data/config.json")

        assert content == '{"key": "value"}'


# === Auth header ===


class TestAuthHeaders:
    @respx.mock
    @pytest.mark.asyncio
    async def test_auth_token_sent(self) -> None:
        url = "http://localhost:8000/mcp"
        received_headers: dict[str, str] = {}

        def capture_request(request: httpx.Request) -> httpx.Response:
            received_headers.update(dict(request.headers))
            return httpx.Response(
                200,
                json={"jsonrpc": "2.0", "id": 1, "result": {"tools": []}},
            )

        respx.post(url).mock(side_effect=capture_request)

        async with McpClient(url, auth_token="my-secret-token") as client:
            await client.list_tools()

        assert received_headers.get("authorization") == "Bearer my-secret-token"

    @respx.mock
    @pytest.mark.asyncio
    async def test_custom_headers_sent(self) -> None:
        url = "http://localhost:8000/mcp"
        received_headers: dict[str, str] = {}

        def capture_request(request: httpx.Request) -> httpx.Response:
            received_headers.update(dict(request.headers))
            return httpx.Response(
                200,
                json={"jsonrpc": "2.0", "id": 1, "result": {"tools": []}},
            )

        respx.post(url).mock(side_effect=capture_request)

        async with McpClient(url, headers={"X-Custom": "value123"}) as client:
            await client.list_tools()

        assert received_headers.get("x-custom") == "value123"
