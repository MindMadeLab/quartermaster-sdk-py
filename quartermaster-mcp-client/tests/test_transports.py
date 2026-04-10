"""Unit tests for transport layer implementations."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from quartermaster_mcp_client.errors import (
    McpConnectionError,
    McpProtocolError,
    McpServerError,
    McpTimeoutError,
)
from quartermaster_mcp_client.transports import (
    SSETransport,
    StreamableTransport,
    create_transport,
    parse_sse_lines,
)


# === parse_sse_lines ===


class TestParseSSELines:
    def test_single_event(self) -> None:
        text = 'data: {"key": "value"}\n\n'
        result = parse_sse_lines(text)
        assert result == [{"key": "value"}]

    def test_multiple_events(self) -> None:
        text = 'data: {"a": 1}\n\ndata: {"b": 2}\n\n'
        result = parse_sse_lines(text)
        assert result == [{"a": 1}, {"b": 2}]

    def test_no_trailing_newline(self) -> None:
        text = 'data: {"key": "value"}'
        result = parse_sse_lines(text)
        assert result == [{"key": "value"}]

    def test_empty_lines_between_events(self) -> None:
        text = 'data: {"a": 1}\n\n\ndata: {"b": 2}\n\n'
        result = parse_sse_lines(text)
        assert result == [{"a": 1}, {"b": 2}]

    def test_no_data_raises(self) -> None:
        with pytest.raises(McpProtocolError, match="No valid SSE data"):
            parse_sse_lines("event: ping\n\n")

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(McpProtocolError, match="Invalid JSON"):
            parse_sse_lines("data: {bad json}\n\n")

    def test_data_with_extra_spaces(self) -> None:
        text = 'data:   {"x": 42}  \n\n'
        result = parse_sse_lines(text)
        assert result == [{"x": 42}]

    def test_ignores_comments_and_other_fields(self) -> None:
        text = ': comment\nevent: message\ndata: {"ok": true}\n\n'
        result = parse_sse_lines(text)
        assert result == [{"ok": True}]


# === SSETransport ===


class TestSSETransport:
    @respx.mock
    @pytest.mark.asyncio
    async def test_successful_request(self) -> None:
        url = "http://localhost:8000/mcp"
        response_data = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"tools": []},
        }
        sse_body = f"data: {json.dumps(response_data)}\n\n"

        respx.post(url).mock(
            return_value=httpx.Response(
                200,
                text=sse_body,
                headers={"content-type": "text/event-stream"},
            )
        )

        async with httpx.AsyncClient() as client:
            transport = SSETransport(client, url, timeout=10.0)
            result = await transport.send_request("tools/list")
            assert result == {"tools": []}

    @respx.mock
    @pytest.mark.asyncio
    async def test_plain_json_fallback(self) -> None:
        url = "http://localhost:8000/mcp"
        response_data = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"status": "ok"},
        }

        respx.post(url).mock(
            return_value=httpx.Response(
                200,
                json=response_data,
                headers={"content-type": "application/json"},
            )
        )

        async with httpx.AsyncClient() as client:
            transport = SSETransport(client, url, timeout=10.0)
            result = await transport.send_request("ping")
            assert result == {"status": "ok"}

    @respx.mock
    @pytest.mark.asyncio
    async def test_server_error_response(self) -> None:
        url = "http://localhost:8000/mcp"
        response_data = {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32603, "message": "Internal error"},
        }

        respx.post(url).mock(
            return_value=httpx.Response(
                200,
                json=response_data,
                headers={"content-type": "application/json"},
            )
        )

        async with httpx.AsyncClient() as client:
            transport = SSETransport(client, url, timeout=10.0)
            with pytest.raises(McpServerError, match="Internal error"):
                await transport.send_request("bad_method")

    @respx.mock
    @pytest.mark.asyncio
    async def test_connection_error(self) -> None:
        url = "http://localhost:9999/mcp"

        respx.post(url).mock(side_effect=httpx.ConnectError("Connection refused"))

        async with httpx.AsyncClient() as client:
            transport = SSETransport(client, url, timeout=10.0)
            with pytest.raises(McpConnectionError, match="Failed to connect"):
                await transport.send_request("test")

    @respx.mock
    @pytest.mark.asyncio
    async def test_timeout_error(self) -> None:
        url = "http://localhost:8000/mcp"

        respx.post(url).mock(side_effect=httpx.ReadTimeout("timed out"))

        async with httpx.AsyncClient() as client:
            transport = SSETransport(client, url, timeout=0.1)
            with pytest.raises(McpTimeoutError, match="timed out"):
                await transport.send_request("slow_method")

    @respx.mock
    @pytest.mark.asyncio
    async def test_http_status_error(self) -> None:
        url = "http://localhost:8000/mcp"

        respx.post(url).mock(return_value=httpx.Response(500, text="Server Error"))

        async with httpx.AsyncClient() as client:
            transport = SSETransport(client, url, timeout=10.0)
            with pytest.raises(McpConnectionError, match="HTTP 500"):
                await transport.send_request("test")

    @respx.mock
    @pytest.mark.asyncio
    async def test_invalid_json_response(self) -> None:
        url = "http://localhost:8000/mcp"

        respx.post(url).mock(
            return_value=httpx.Response(
                200,
                text="not json at all",
                headers={"content-type": "application/json"},
            )
        )

        async with httpx.AsyncClient() as client:
            transport = SSETransport(client, url, timeout=10.0)
            with pytest.raises(McpProtocolError, match="not valid JSON"):
                await transport.send_request("test")

    @respx.mock
    @pytest.mark.asyncio
    async def test_missing_result_field(self) -> None:
        url = "http://localhost:8000/mcp"
        response_data = {"jsonrpc": "2.0", "id": 1}  # no result or error

        respx.post(url).mock(return_value=httpx.Response(200, json=response_data))

        async with httpx.AsyncClient() as client:
            transport = SSETransport(client, url, timeout=10.0)
            with pytest.raises(McpProtocolError, match="missing both"):
                await transport.send_request("test")

    @respx.mock
    @pytest.mark.asyncio
    async def test_invalid_jsonrpc_version(self) -> None:
        url = "http://localhost:8000/mcp"
        response_data = {"jsonrpc": "1.0", "id": 1, "result": {}}

        respx.post(url).mock(return_value=httpx.Response(200, json=response_data))

        async with httpx.AsyncClient() as client:
            transport = SSETransport(client, url, timeout=10.0)
            with pytest.raises(McpProtocolError, match="Invalid JSON-RPC version"):
                await transport.send_request("test")


# === StreamableTransport ===


class TestStreamableTransport:
    @respx.mock
    @pytest.mark.asyncio
    async def test_successful_request(self) -> None:
        url = "http://localhost:8000/mcp"
        response_data = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"data": "hello"},
        }

        respx.post(url).mock(return_value=httpx.Response(200, json=response_data))

        async with httpx.AsyncClient() as client:
            transport = StreamableTransport(client, url, timeout=10.0)
            result = await transport.send_request("test_method")
            assert result == {"data": "hello"}

    @respx.mock
    @pytest.mark.asyncio
    async def test_connection_error(self) -> None:
        url = "http://localhost:9999/mcp"

        respx.post(url).mock(side_effect=httpx.ConnectError("refused"))

        async with httpx.AsyncClient() as client:
            transport = StreamableTransport(client, url, timeout=10.0)
            with pytest.raises(McpConnectionError):
                await transport.send_request("test")

    @respx.mock
    @pytest.mark.asyncio
    async def test_timeout_error(self) -> None:
        url = "http://localhost:8000/mcp"

        respx.post(url).mock(side_effect=httpx.ReadTimeout("timed out"))

        async with httpx.AsyncClient() as client:
            transport = StreamableTransport(client, url, timeout=0.1)
            with pytest.raises(McpTimeoutError):
                await transport.send_request("slow")

    @respx.mock
    @pytest.mark.asyncio
    async def test_server_error_in_json(self) -> None:
        url = "http://localhost:8000/mcp"
        response_data = {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32601, "message": "Method not found"},
        }

        respx.post(url).mock(return_value=httpx.Response(200, json=response_data))

        async with httpx.AsyncClient() as client:
            transport = StreamableTransport(client, url, timeout=10.0)
            with pytest.raises(McpServerError, match="Method not found"):
                await transport.send_request("unknown")


# === create_transport factory ===


class TestCreateTransport:
    def test_create_sse(self) -> None:
        client = httpx.AsyncClient()
        transport = create_transport("sse", client, "http://localhost:8000/mcp")
        assert isinstance(transport, SSETransport)

    def test_create_streamable(self) -> None:
        client = httpx.AsyncClient()
        transport = create_transport("streamable", client, "http://localhost:8000/mcp")
        assert isinstance(transport, StreamableTransport)

    def test_invalid_transport_raises(self) -> None:
        client = httpx.AsyncClient()
        with pytest.raises(ValueError, match="Unknown transport"):
            create_transport("websocket", client, "http://localhost:8000/mcp")
