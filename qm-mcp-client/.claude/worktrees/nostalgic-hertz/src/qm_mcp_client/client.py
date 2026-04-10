"""MCP Client implementation.

Lightweight async/sync client for Model Context Protocol servers.
Implements JSON-RPC 2.0 over SSE or Streamable HTTP transports.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from typing import Any

import httpx

from qm_mcp_client.errors import (
    McpAuthenticationError,
    McpConnectionError,
    McpProtocolError,
    McpServerError,
    McpTimeoutError,
    McpToolNotFoundError,
)
from qm_mcp_client.transports import Transport, create_transport
from qm_mcp_client.types import (
    McpServerInfo,
    McpTool,
    ToolParameter,
    ToolParameterOption,
)

logger = logging.getLogger(__name__)

_URL_PATTERN = re.compile(r"^https?://", re.IGNORECASE)


def parse_sse_response(text: str) -> dict[str, Any]:
    """Parse a single Server-Sent Event response text.

    SSE format: "data: {json}\\n\\n"

    Args:
        text: Raw SSE response text.

    Returns:
        Parsed JSON object.

    Raises:
        McpProtocolError: If text cannot be parsed as valid JSON.
    """
    if not text.strip():
        raise McpProtocolError("Empty SSE response")

    # Collect data lines from SSE format
    data_parts: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("data:"):
            data_parts.append(stripped[5:].strip())

    # If we found data: lines, join them; otherwise try raw text
    payload = "\n".join(data_parts) if data_parts else text.strip()

    try:
        result: dict[str, Any] = json.loads(payload)
        return result
    except json.JSONDecodeError as e:
        raise McpProtocolError(f"Invalid JSON in SSE response: {e}") from e


def parse_json_schema_type(schema: dict[str, Any]) -> str:
    """Extract type string from JSON Schema object.

    Handles simple types, type arrays, anyOf/oneOf compositions.

    Args:
        schema: JSON Schema definition.

    Returns:
        Type string suitable for display and parameter handling.
    """
    if not isinstance(schema, dict):
        return "object"

    schema_type = schema.get("type")

    if isinstance(schema_type, list):
        for t in schema_type:
            if t != "null":
                return str(t)
        return "null"

    if isinstance(schema_type, str):
        return schema_type

    # Handle anyOf/oneOf — pick first non-null type
    for key in ("anyOf", "oneOf"):
        variants = schema.get(key)
        if isinstance(variants, list):
            for variant in variants:
                if isinstance(variant, dict):
                    vtype = variant.get("type")
                    if vtype and vtype != "null":
                        return str(vtype)

    # Handle allOf — take first type found
    all_of = schema.get("allOf")
    if isinstance(all_of, list):
        for item in all_of:
            if isinstance(item, dict) and "type" in item:
                return str(item["type"])

    return "object"


def parse_tool_parameters(input_schema: dict[str, Any]) -> list[ToolParameter]:
    """Parse tool parameters from JSON Schema input definition.

    Args:
        input_schema: JSON Schema object defining tool inputs.

    Returns:
        List of ToolParameter objects.

    Raises:
        McpProtocolError: If schema is malformed.
    """
    if not isinstance(input_schema, dict):
        raise McpProtocolError("input_schema must be a dict")

    properties = input_schema.get("properties", {})
    required_fields = set(input_schema.get("required", []))
    parameters: list[ToolParameter] = []

    for prop_name, prop_schema in properties.items():
        if not isinstance(prop_schema, dict):
            continue

        param_type = parse_json_schema_type(prop_schema)
        description = prop_schema.get("description", "")

        enum_values: list[str] | None = None
        options: list[ToolParameterOption] = []

        if "enum" in prop_schema:
            enum_values = [str(v) for v in prop_schema["enum"]]
            options = [
                ToolParameterOption(label=str(v), value=str(v))
                for v in prop_schema["enum"]
            ]

        param = ToolParameter(
            name=prop_name,
            type=param_type,
            description=description,
            required=prop_name in required_fields,
            default=prop_schema.get("default"),
            enum=enum_values,
            options=options,
            min_value=prop_schema.get("minimum"),
            max_value=prop_schema.get("maximum"),
            min_length=prop_schema.get("minLength"),
            max_length=prop_schema.get("maxLength"),
            pattern=prop_schema.get("pattern"),
        )
        parameters.append(param)

    return parameters


def _run_sync(coro: Any) -> Any:
    """Run an async coroutine synchronously.

    Handles the case where an event loop is already running
    by creating a new thread-based loop.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # We're inside an async context — use a thread
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


class McpClient:
    """Async/sync client for Model Context Protocol servers.

    Supports dual interfaces:
    - Async: ``async with McpClient(...) as client: await client.list_tools()``
    - Sync: ``with McpClient(...) as client: client.list_tools_sync()``

    Args:
        url: Base URL of the MCP server.
        transport: Transport layer — "sse" or "streamable".
        timeout: Request timeout in seconds.
        max_retries: Maximum retries for transient failures.
        auth_token: Optional Bearer token for Authorization header.
        headers: Additional HTTP headers to send with every request.

    Raises:
        ValueError: If url or transport is invalid.
    """

    def __init__(
        self,
        url: str,
        transport: str = "sse",
        timeout: float = 30.0,
        max_retries: int = 3,
        auth_token: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        if not _URL_PATTERN.match(url):
            raise ValueError(
                f"Invalid URL {url!r}: must start with http:// or https://"
            )
        if transport not in ("sse", "streamable"):
            raise ValueError(
                f"Invalid transport {transport!r}: must be 'sse' or 'streamable'"
            )

        self._url = url.rstrip("/")
        self._transport_kind = transport
        self._timeout = timeout
        self._max_retries = max_retries
        self._auth_token = auth_token
        self._extra_headers = headers or {}

        self._async_client: httpx.AsyncClient | None = None
        self._sync_client: httpx.Client | None = None
        self._transport: Transport | None = None

    def _build_headers(self) -> dict[str, str]:
        """Build common HTTP headers."""
        h: dict[str, str] = {}
        if self._auth_token:
            h["Authorization"] = f"Bearer {self._auth_token}"
        h.update(self._extra_headers)
        return h

    # --- Async context manager ---

    async def __aenter__(self) -> McpClient:
        self._async_client = httpx.AsyncClient(headers=self._build_headers())
        self._transport = create_transport(
            self._transport_kind,
            self._async_client,
            self._url,
            self._timeout,
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None
        self._transport = None

    # --- Sync context manager ---

    def __enter__(self) -> McpClient:
        self._async_client = httpx.AsyncClient(headers=self._build_headers())
        self._transport = create_transport(
            self._transport_kind,
            self._async_client,
            self._url,
            self._timeout,
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        if self._async_client:
            _run_sync(self._async_client.aclose())
            self._async_client = None
        self._transport = None

    # --- Core async methods ---

    async def _send_request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send JSON-RPC request with retry logic.

        Args:
            method: JSON-RPC method name.
            params: Method parameters.

        Returns:
            Response result dict.

        Raises:
            McpConnectionError: If connection fails after all retries.
            McpProtocolError: If response is malformed.
            McpServerError: If server returns an error.
            McpTimeoutError: If request times out.
        """
        if self._transport is None:
            raise McpConnectionError(
                "Client is not connected. Use 'async with' or 'with' context manager."
            )

        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                return await self._transport.send_request(method, params)
            except (McpConnectionError, McpTimeoutError) as e:
                last_error = e
                if attempt < self._max_retries:
                    # Exponential backoff with jitter
                    delay = min(2 ** (attempt - 1) + random.uniform(0, 1), 10)
                    logger.warning(
                        "Request %s failed (attempt %d/%d), retrying in %.1fs: %s",
                        method,
                        attempt,
                        self._max_retries,
                        delay,
                        e,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "Request %s failed after %d attempts: %s",
                        method,
                        self._max_retries,
                        e,
                    )
            except (McpProtocolError, McpServerError, McpAuthenticationError):
                # Non-retriable errors — raise immediately
                raise

        assert last_error is not None
        raise last_error

    async def server_info(self) -> McpServerInfo:
        """Get server information and capabilities.

        Returns:
            McpServerInfo with server metadata.

        Raises:
            McpConnectionError: If connection fails.
            McpProtocolError: If response is malformed.
        """
        result = await self._send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "qm-mcp-client", "version": "0.1.0"},
            },
        )

        server_info = result.get("serverInfo", {})
        return McpServerInfo(
            name=server_info.get("name", "unknown"),
            version=server_info.get("version", "unknown"),
            protocol_version=result.get("protocolVersion", "unknown"),
            capabilities=result.get("capabilities", {}),
        )

    async def list_tools(self) -> list[McpTool]:
        """List all tools available on the server.

        Returns:
            List of McpTool objects with parameter metadata.
        """
        result = await self._send_request("tools/list")

        tools: list[McpTool] = []
        for tool_data in result.get("tools", []):
            input_schema = tool_data.get("inputSchema", {})
            parameters = parse_tool_parameters(input_schema)
            tools.append(
                McpTool(
                    name=tool_data.get("name", ""),
                    description=tool_data.get("description", ""),
                    parameters=parameters,
                    input_schema=input_schema,
                )
            )
        return tools

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        """Invoke a tool on the server.

        Args:
            name: Tool name.
            arguments: Tool arguments as dict.

        Returns:
            Tool result content.

        Raises:
            McpToolNotFoundError: If tool doesn't exist.
            McpServerError: If tool execution fails.
        """
        try:
            result = await self._send_request(
                "tools/call",
                {
                    "name": name,
                    "arguments": arguments or {},
                },
            )
        except McpServerError as e:
            if e.code == -32601:
                raise McpToolNotFoundError(f"Tool {name!r} not found on server") from e
            raise

        # MCP tool results have a 'content' array
        content = result.get("content", [])
        if len(content) == 1:
            item = content[0]
            if item.get("type") == "text":
                return item.get("text", "")
            return item
        return content if content else result

    async def list_resources(self) -> list[dict[str, Any]]:
        """List all resources available on the server.

        Returns:
            List of resource metadata dicts.
        """
        result = await self._send_request("resources/list")
        resources: list[dict[str, Any]] = result.get("resources", [])
        return resources

    async def read_resource(self, uri: str) -> str:
        """Read content of a resource.

        Args:
            uri: Resource URI.

        Returns:
            Resource content as string.
        """
        result = await self._send_request("resources/read", {"uri": uri})
        contents = result.get("contents", [])
        if contents:
            return str(contents[0].get("text", ""))
        return ""

    # --- Sync wrappers ---

    def server_info_sync(self) -> McpServerInfo:
        """Sync wrapper for server_info()."""
        result: McpServerInfo = _run_sync(self.server_info())
        return result

    def list_tools_sync(self) -> list[McpTool]:
        """Sync wrapper for list_tools()."""
        result: list[McpTool] = _run_sync(self.list_tools())
        return result

    def call_tool_sync(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        """Sync wrapper for call_tool()."""
        return _run_sync(self.call_tool(name, arguments))

    def list_resources_sync(self) -> list[dict[str, Any]]:
        """Sync wrapper for list_resources()."""
        result: list[dict[str, Any]] = _run_sync(self.list_resources())
        return result

    def read_resource_sync(self, uri: str) -> str:
        """Sync wrapper for read_resource()."""
        result: str = _run_sync(self.read_resource(uri))
        return result

    @property
    def is_connected(self) -> bool:
        """Check if client has an active connection."""
        return self._transport is not None
