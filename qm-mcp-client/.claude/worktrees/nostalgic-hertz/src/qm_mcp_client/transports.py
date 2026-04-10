"""Transport layer implementations for MCP protocol.

Supports Server-Sent Events (SSE) and Streamable HTTP transports
as defined by the Model Context Protocol specification.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

import httpx

from qm_mcp_client.errors import (
    McpConnectionError,
    McpProtocolError,
    McpServerError,
    McpTimeoutError,
)

logger = logging.getLogger(__name__)

# JSON-RPC error codes mapped to specific exception types
_JSONRPC_METHOD_NOT_FOUND = -32601
_JSONRPC_INTERNAL_ERROR = -32603
_JSONRPC_PARSE_ERROR = -32700
_JSONRPC_INVALID_REQUEST = -32600


def _build_jsonrpc_request(
    method: str,
    params: dict[str, Any] | None,
    request_id: int,
) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 request payload."""
    req: dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
    }
    if params is not None:
        req["params"] = params
    return req


def _validate_jsonrpc_response(data: dict[str, Any], request_id: int) -> None:
    """Validate a JSON-RPC 2.0 response structure.

    Raises:
        McpProtocolError: If the response doesn't conform to JSON-RPC 2.0.
        McpServerError: If the response contains a JSON-RPC error.
    """
    if data.get("jsonrpc") != "2.0":
        raise McpProtocolError(
            f"Invalid JSON-RPC version: {data.get('jsonrpc')!r}, expected '2.0'"
        )

    if "error" in data:
        err = data["error"]
        code = err.get("code", 0)
        message = err.get("message", "Unknown server error")
        raise McpServerError(message, code=code)

    if "result" not in data:
        raise McpProtocolError(
            "JSON-RPC response missing both 'result' and 'error' fields"
        )


def parse_sse_lines(text: str) -> list[dict[str, Any]]:
    """Parse Server-Sent Events text into JSON objects.

    SSE format uses "data: {json}" lines separated by blank lines.
    Multiple events can appear in a single response.

    Args:
        text: Raw SSE response text.

    Returns:
        List of parsed JSON objects from SSE data fields.

    Raises:
        McpProtocolError: If no valid data lines are found or JSON is invalid.
    """
    results: list[dict[str, Any]] = []
    current_data_lines: list[str] = []

    for line in text.split("\n"):
        stripped = line.strip()

        if stripped.startswith("data:"):
            payload = stripped[5:].strip()
            if payload:
                current_data_lines.append(payload)
        elif stripped == "" and current_data_lines:
            # End of SSE event — flush
            combined = "\n".join(current_data_lines)
            try:
                results.append(json.loads(combined))
            except json.JSONDecodeError as e:
                raise McpProtocolError(f"Invalid JSON in SSE data field: {e}") from e
            current_data_lines = []

    # Handle trailing data without final blank line
    if current_data_lines:
        combined = "\n".join(current_data_lines)
        try:
            results.append(json.loads(combined))
        except json.JSONDecodeError as e:
            raise McpProtocolError(f"Invalid JSON in SSE data field: {e}") from e

    if not results:
        raise McpProtocolError("No valid SSE data events found in response")

    return results


class Transport(ABC):
    """Abstract base class for MCP transport layers."""

    @abstractmethod
    async def send_request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send a JSON-RPC request and return the result.

        Args:
            method: JSON-RPC method name.
            params: Request parameters.

        Returns:
            The 'result' value from the JSON-RPC response.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the transport and release resources."""
        ...


class SSETransport(Transport):
    """Server-Sent Events transport for MCP.

    Sends JSON-RPC requests as HTTP POST and reads the response
    as an SSE stream. This is the recommended MCP transport.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        url: str,
        timeout: float = 30.0,
    ) -> None:
        self._client = client
        self._url = url
        self._timeout = timeout
        self._request_id = 0

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def send_request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send request via SSE transport.

        Posts JSON-RPC payload and parses the SSE-formatted response.
        """
        req_id = self._next_id()
        payload = _build_jsonrpc_request(method, params, req_id)

        logger.debug("SSE request [%d]: %s %s", req_id, method, params)

        try:
            response = await self._client.post(
                self._url,
                json=payload,
                headers={
                    "Accept": "text/event-stream",
                    "Content-Type": "application/json",
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
        except httpx.ConnectError as e:
            raise McpConnectionError(
                f"Failed to connect to MCP server at {self._url}: {e}"
            ) from e
        except httpx.TimeoutException as e:
            raise McpTimeoutError(
                f"Request timed out after {self._timeout}s: {e}"
            ) from e
        except httpx.HTTPStatusError as e:
            raise McpConnectionError(
                f"HTTP {e.response.status_code} from MCP server: {e}"
            ) from e

        content_type = response.headers.get("content-type", "")
        text = response.text

        if "text/event-stream" in content_type:
            events = parse_sse_lines(text)
            data = events[-1]  # Use last event as the final response
        else:
            # Server might return plain JSON
            try:
                data = json.loads(text)
            except json.JSONDecodeError as e:
                raise McpProtocolError(f"Response is not valid JSON or SSE: {e}") from e

        _validate_jsonrpc_response(data, req_id)
        result: dict[str, Any] = data["result"]
        return result

    async def close(self) -> None:
        """Close is handled by the parent client."""
        pass


class StreamableTransport(Transport):
    """Streamable HTTP transport for MCP.

    Uses standard HTTP POST with JSON responses. Supports chunked
    Transfer-Encoding for streaming. Fallback when SSE is unavailable.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        url: str,
        timeout: float = 30.0,
    ) -> None:
        self._client = client
        self._url = url
        self._timeout = timeout
        self._request_id = 0

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def send_request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send request via Streamable HTTP transport.

        Posts JSON-RPC payload and reads a plain JSON response,
        with support for chunked streaming.
        """
        req_id = self._next_id()
        payload = _build_jsonrpc_request(method, params, req_id)

        logger.debug("Streamable request [%d]: %s %s", req_id, method, params)

        try:
            response = await self._client.post(
                self._url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self._timeout,
            )
            response.raise_for_status()
        except httpx.ConnectError as e:
            raise McpConnectionError(
                f"Failed to connect to MCP server at {self._url}: {e}"
            ) from e
        except httpx.TimeoutException as e:
            raise McpTimeoutError(
                f"Request timed out after {self._timeout}s: {e}"
            ) from e
        except httpx.HTTPStatusError as e:
            raise McpConnectionError(
                f"HTTP {e.response.status_code} from MCP server: {e}"
            ) from e

        try:
            data = response.json()
        except json.JSONDecodeError as e:
            raise McpProtocolError(f"Response is not valid JSON: {e}") from e

        _validate_jsonrpc_response(data, req_id)
        result: dict[str, Any] = data["result"]
        return result

    async def close(self) -> None:
        """Close is handled by the parent client."""
        pass


def create_transport(
    kind: str,
    client: httpx.AsyncClient,
    url: str,
    timeout: float = 30.0,
) -> Transport:
    """Factory function to create a transport by name.

    Args:
        kind: Transport type — "sse" or "streamable".
        client: httpx.AsyncClient to use for requests.
        url: MCP server URL.
        timeout: Request timeout in seconds.

    Returns:
        A Transport instance.

    Raises:
        ValueError: If kind is not a recognized transport.
    """
    if kind == "sse":
        return SSETransport(client, url, timeout)
    elif kind == "streamable":
        return StreamableTransport(client, url, timeout)
    else:
        raise ValueError(f"Unknown transport {kind!r}. Must be 'sse' or 'streamable'.")
