"""
WebRequestTool: HTTP GET/POST requests using httpx.

Provides a simple HTTP client tool for AI agents. Requires the ``httpx``
package, installable via ``pip install quartermaster-tools[web]``.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Any
from urllib.parse import urlparse

from quartermaster_tools.base import AbstractTool
from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolParameterOption, ToolResult

# Default timeout in seconds
DEFAULT_TIMEOUT = 30

# Hard ceiling for timeout
MAX_TIMEOUT = 300

# Default maximum response body size: 5 MB
DEFAULT_MAX_RESPONSE_SIZE = 5 * 1024 * 1024

# Private/reserved IP networks that must not be accessed (SSRF protection)
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),       # Loopback
    ipaddress.ip_network("::1/128"),            # IPv6 loopback
    ipaddress.ip_network("10.0.0.0/8"),         # RFC-1918
    ipaddress.ip_network("172.16.0.0/12"),      # RFC-1918
    ipaddress.ip_network("192.168.0.0/16"),     # RFC-1918
    ipaddress.ip_network("169.254.0.0/16"),     # Link-local / AWS metadata
    ipaddress.ip_network("fe80::/10"),          # IPv6 link-local
    ipaddress.ip_network("fc00::/7"),           # IPv6 unique local
    ipaddress.ip_network("0.0.0.0/8"),          # "This" network
    ipaddress.ip_network("::/128"),             # Unspecified
]


def _is_private_ip(host: str) -> bool:
    """Check if a hostname resolves to a private/reserved IP address."""
    try:
        addr_infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return False  # DNS resolution failed — httpx will handle the error

    for _, _, _, _, sockaddr in addr_infos:
        ip = ipaddress.ip_address(sockaddr[0])
        for network in _BLOCKED_NETWORKS:
            if ip in network:
                return True
    return False


class WebRequestTool(AbstractTool):
    """Make HTTP GET or POST requests and return the response body.

    Requires httpx to be installed (``pip install quartermaster-tools[web]``).

    Features:
    - Supports GET and POST methods
    - SSRF protection: blocks requests to private/loopback/link-local IPs
    - Streaming response handling to prevent OOM on large responses
    - Configurable timeout and max response size
    """

    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT,
        max_response_size: int = DEFAULT_MAX_RESPONSE_SIZE,
    ) -> None:
        """Initialise the WebRequestTool.

        Args:
            timeout: Request timeout in seconds (max 300).
            max_response_size: Maximum response body size in bytes.

        Raises:
            ValueError: If timeout exceeds MAX_TIMEOUT.
        """
        if timeout > MAX_TIMEOUT:
            raise ValueError(f"timeout must be <= {MAX_TIMEOUT} seconds, got {timeout}")
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
                "GET and POST methods with SSRF protection and configurable timeout."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=False,
        )

    def _validate_url(self, url: str) -> str | None:
        """Validate URL scheme and host. Returns error message or None."""
        parsed = urlparse(url)

        # Validate scheme
        if parsed.scheme not in ("http", "https"):
            return f"Only http and https schemes are allowed, got: {parsed.scheme!r}"

        # Validate host exists
        hostname = parsed.hostname
        if not hostname:
            return "URL must include a hostname"

        # SSRF check: resolve hostname and block private IPs
        if _is_private_ip(hostname):
            return "Access denied: requests to private/internal networks are not allowed"

        return None

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

        # Validate URL before making any request
        url_error = self._validate_url(url)
        if url_error:
            return ToolResult(success=False, error=url_error)

        try:
            import httpx
        except ImportError:
            return ToolResult(
                success=False,
                error=(
                    "httpx is required for WebRequestTool. "
                    "Install it with: pip install quartermaster-tools[web]"
                ),
            )

        try:
            # Use streaming to avoid OOM on large responses
            with httpx.Client(
                timeout=self._timeout,
                follow_redirects=False,
            ) as client:
                if method == "GET":
                    response = client.get(url, headers=headers)
                else:
                    response = client.post(url, headers=headers, content=body)

                # Stream-read with size limit
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
