"""
WebRequestTool: HTTP GET/POST requests using httpx.

Provides a simple HTTP client tool for AI agents. Requires the ``httpx``
package, installable via ``pip install qm-tools[web]``.
"""

from __future__ import annotations

from typing import Any

from qm_tools.base import AbstractTool
from qm_tools.types import ToolDescriptor, ToolParameter, ToolParameterOption, ToolResult

# Default timeout in seconds
DEFAULT_TIMEOUT = 30

# Default maximum response body size: 5 MB
DEFAULT_MAX_RESPONSE_SIZE = 5 * 1024 * 1024


class WebRequestTool(AbstractTool):
    """Make HTTP GET or POST requests and return the response body.

    Requires httpx to be installed (``pip install qm-tools[web]``).

    Features:
    - Supports GET and POST methods
    - Configurable timeout and max response size
    - Returns response body, status code, and headers
    """

    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT,
        max_response_size: int = DEFAULT_MAX_RESPONSE_SIZE,
    ) -> None:
        """Initialise the WebRequestTool.

        Args:
            timeout: Request timeout in seconds.
            max_response_size: Maximum response body size in bytes.
        """
        self._timeout = timeout
        self._max_response_size = max_response_size

    def name(self) -> str:
        """Return the tool name."""
        return "web_request"

    def version(self) -> str:
        """Return the tool version."""
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        """Return parameter definitions for the tool."""
        return [
            ToolParameter(
                name="url",
                description="The URL to send the request to.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="method",
                description="HTTP method: GET or POST.",
                type="string",
                required=False,
                default="GET",
                options=[
                    ToolParameterOption(label="GET", value="GET"),
                    ToolParameterOption(label="POST", value="POST"),
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
                description="Request body for POST requests (string or JSON object).",
                type="string",
                required=False,
            ),
        ]

    def info(self) -> ToolDescriptor:
        """Return metadata describing this tool."""
        return ToolDescriptor(
            name=self.name(),
            short_description="Make an HTTP GET or POST request.",
            long_description=(
                "Sends an HTTP request to the specified URL and returns "
                "the response body, status code, and headers. Supports "
                "GET and POST methods with configurable timeout."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=False,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the HTTP request and return the response.

        Args:
            url: The target URL.
            method: HTTP method (GET or POST, default GET).
            headers: Optional request headers dict.
            body: Optional request body for POST.

        Returns:
            ToolResult with response data including body, status_code, and headers.
        """
        url: str = kwargs.get("url", "")
        method: str = kwargs.get("method", "GET").upper()
        headers: dict[str, str] | None = kwargs.get("headers")
        body: str | None = kwargs.get("body")

        if not url:
            return ToolResult(success=False, error="Parameter 'url' is required")

        if method not in ("GET", "POST"):
            return ToolResult(
                success=False,
                error=f"Unsupported HTTP method: {method}. Use GET or POST.",
            )

        try:
            import httpx
        except ImportError:
            return ToolResult(
                success=False,
                error=(
                    "httpx is required for WebRequestTool. "
                    "Install it with: pip install qm-tools[web]"
                ),
            )

        try:
            with httpx.Client(timeout=self._timeout) as client:
                if method == "GET":
                    response = client.get(url, headers=headers)
                else:
                    response = client.post(url, headers=headers, content=body)

                # Check response size
                content_length = len(response.content)
                if content_length > self._max_response_size:
                    return ToolResult(
                        success=False,
                        error=(
                            f"Response too large: {content_length} bytes "
                            f"(limit: {self._max_response_size} bytes)"
                        ),
                    )

                return ToolResult(
                    success=True,
                    data={
                        "body": response.text,
                        "status_code": response.status_code,
                        "headers": dict(response.headers),
                        "url": str(response.url),
                    },
                )

        except httpx.TimeoutException:
            return ToolResult(
                success=False,
                error=f"Request timed out after {self._timeout} seconds",
            )
        except httpx.ConnectError as e:
            return ToolResult(
                success=False,
                error=f"Connection error: {e}",
            )
        except httpx.HTTPError as e:
            return ToolResult(
                success=False,
                error=f"HTTP error: {e}",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Unexpected error: {e}",
            )
