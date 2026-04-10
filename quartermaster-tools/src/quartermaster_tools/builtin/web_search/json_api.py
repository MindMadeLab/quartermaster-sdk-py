"""
JsonApiTool: JSON API caller with optional JMESPath filtering.

Makes HTTP requests expecting JSON responses, auto-parses the response,
and optionally filters the result using JMESPath expressions.
"""

from __future__ import annotations

import json
from typing import Any

from quartermaster_tools.base import AbstractTool
from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolParameterOption, ToolResult

_DEFAULT_TIMEOUT = 30
_SUPPORTED_METHODS = ("GET", "POST", "PUT", "DELETE", "PATCH")


class JsonApiTool(AbstractTool):
    """Call a JSON API endpoint and return parsed results.

    Makes HTTP requests with JSON content-type handling, auto-parses JSON
    responses, and supports optional JMESPath filtering. Gracefully handles
    the case where the jmespath library is not installed.
    """

    def name(self) -> str:
        """Return the tool name."""
        return "json_api"

    def version(self) -> str:
        """Return the tool version."""
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        """Return parameter definitions for the tool."""
        return [
            ToolParameter(
                name="url",
                description="The API endpoint URL.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="method",
                description="HTTP method (default GET).",
                type="string",
                required=False,
                default="GET",
                options=[
                    ToolParameterOption(label=m, value=m) for m in _SUPPORTED_METHODS
                ],
            ),
            ToolParameter(
                name="headers",
                description="Optional HTTP headers as a JSON object.",
                type="object",
                required=False,
            ),
            ToolParameter(
                name="body",
                description="Request body (will be serialised as JSON if dict/list).",
                type="object",
                required=False,
            ),
            ToolParameter(
                name="jmespath_filter",
                description="Optional JMESPath expression to filter the JSON response.",
                type="string",
                required=False,
            ),
        ]

    def info(self) -> ToolDescriptor:
        """Return metadata describing this tool."""
        return ToolDescriptor(
            name=self.name(),
            short_description="Call a JSON API and return parsed results with optional JMESPath filtering.",
            long_description=(
                "Makes an HTTP request to a JSON API endpoint, automatically parses "
                "the JSON response, and optionally filters it with a JMESPath "
                "expression. Supports all standard HTTP methods."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=False,
        )

    def call_api(
        self,
        url: str,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        body: Any = None,
        jmespath_filter: str | None = None,
    ) -> ToolResult:
        """Call a JSON API endpoint and return parsed results.

        Args:
            url: The API endpoint URL.
            method: HTTP method (default GET).
            headers: Optional HTTP headers.
            body: Optional request body (dict/list serialised as JSON, or string).
            jmespath_filter: Optional JMESPath expression to filter the response.

        Returns:
            ToolResult with parsed JSON data.
        """
        return self.run(
            url=url,
            method=method,
            headers=headers,
            body=body,
            jmespath_filter=jmespath_filter,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the API call and return parsed JSON.

        Args:
            url: The API endpoint URL.
            method: HTTP method (default GET).
            headers: Optional request headers dict.
            body: Optional request body.
            jmespath_filter: Optional JMESPath filter expression.

        Returns:
            ToolResult with parsed JSON data, status_code, and headers.
        """
        url: str = kwargs.get("url", "").strip()
        method: str = kwargs.get("method", "GET").upper()
        headers: dict[str, str] | None = kwargs.get("headers")
        body: Any = kwargs.get("body")
        jmespath_filter: str | None = kwargs.get("jmespath_filter")

        if not url:
            return ToolResult(success=False, error="Parameter 'url' is required")

        if method not in _SUPPORTED_METHODS:
            return ToolResult(
                success=False,
                error=f"Unsupported HTTP method: {method}. Use one of: {', '.join(_SUPPORTED_METHODS)}.",
            )

        try:
            import httpx
        except ImportError:
            return ToolResult(
                success=False,
                error=(
                    "httpx is required for JsonApiTool. "
                    "Install it with: pip install quartermaster-tools[web]"
                ),
            )

        # Build request kwargs
        request_headers = {"Accept": "application/json"}
        if headers:
            request_headers.update(headers)

        content: str | None = None
        if body is not None:
            if isinstance(body, (dict, list)):
                content = json.dumps(body)
                request_headers.setdefault("Content-Type", "application/json")
            else:
                content = str(body)

        try:
            with httpx.Client(timeout=_DEFAULT_TIMEOUT, follow_redirects=True) as client:
                response = client.request(
                    method,
                    url,
                    headers=request_headers,
                    content=content,
                )
                response.raise_for_status()

                try:
                    json_data = response.json()
                except (json.JSONDecodeError, ValueError) as e:
                    return ToolResult(
                        success=False,
                        error=f"Failed to parse JSON response: {e}",
                        data={"raw_body": response.text[:2000]},
                    )

        except httpx.TimeoutException:
            return ToolResult(success=False, error="API request timed out")
        except httpx.HTTPStatusError as e:
            # Try to parse error body as JSON
            error_body = ""
            try:
                error_body = e.response.text[:2000]
            except Exception:
                pass
            return ToolResult(
                success=False,
                error=f"HTTP {e.response.status_code}",
                data={"response_body": error_body},
            )
        except httpx.HTTPError as e:
            return ToolResult(success=False, error=f"HTTP error: {e}")

        # Apply JMESPath filter if provided
        filtered_data = json_data
        if jmespath_filter:
            filtered_data = self._apply_jmespath(json_data, jmespath_filter)
            if isinstance(filtered_data, ToolResult):
                return filtered_data  # Error result from jmespath

        return ToolResult(
            success=True,
            data={
                "json": filtered_data,
                "status_code": response.status_code,
                "headers": dict(response.headers),
            },
        )

    def _apply_jmespath(self, data: Any, expression: str) -> Any:
        """Apply a JMESPath filter to JSON data.

        Args:
            data: The parsed JSON data.
            expression: JMESPath expression string.

        Returns:
            Filtered data, or a ToolResult on error.
        """
        try:
            import jmespath
        except ImportError:
            return ToolResult(
                success=False,
                error=(
                    "jmespath library is required for JMESPath filtering. "
                    "Install it with: pip install jmespath"
                ),
            )

        try:
            return jmespath.search(expression, data)
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"JMESPath filter error: {e}",
            )
