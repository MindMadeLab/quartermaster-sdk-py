"""
WebRequestTool: HTTP requests using httpx.

Provides a simple HTTP client tool for AI agents supporting GET, POST, PUT,
DELETE, and PATCH methods. Requires the ``httpx`` package, installable via
``pip install quartermaster-tools[web]``.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

from quartermaster_tools.decorator import tool

# Default timeout in seconds
DEFAULT_TIMEOUT = 30

# Hard ceiling for timeout
MAX_TIMEOUT = 300

# Default maximum response body size: 5 MB
DEFAULT_MAX_RESPONSE_SIZE = 5 * 1024 * 1024

# Private/reserved IP networks that must not be accessed (SSRF protection)
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),  # Loopback
    ipaddress.ip_network("::1/128"),  # IPv6 loopback
    ipaddress.ip_network("10.0.0.0/8"),  # RFC-1918
    ipaddress.ip_network("172.16.0.0/12"),  # RFC-1918
    ipaddress.ip_network("192.168.0.0/16"),  # RFC-1918
    ipaddress.ip_network("169.254.0.0/16"),  # Link-local / AWS metadata
    ipaddress.ip_network("fe80::/10"),  # IPv6 link-local
    ipaddress.ip_network("fc00::/7"),  # IPv6 unique local
    ipaddress.ip_network("0.0.0.0/8"),  # "This" network
    ipaddress.ip_network("::/128"),  # Unspecified
]


def _is_private_ip(host: str) -> bool:
    """Check if a hostname resolves to a private/reserved IP address."""
    try:
        addr_infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return False  # DNS resolution failed -- httpx will handle the error

    for _, _, _, _, sockaddr in addr_infos:
        ip = ipaddress.ip_address(sockaddr[0])
        for network in _BLOCKED_NETWORKS:
            if ip in network:
                return True
    return False


_SUPPORTED_METHODS = ("GET", "POST", "PUT", "DELETE", "PATCH")


def _validate_url(url: str) -> str | None:
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


def _web_request_impl(
    url: str,
    method: str = "GET",
    headers: dict | None = None,
    body: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    max_response_size: int = DEFAULT_MAX_RESPONSE_SIZE,
) -> dict:
    """Core implementation for making HTTP requests."""
    if not url:
        return {"error": "Parameter 'url' is required"}

    method = method.upper()
    if method not in _SUPPORTED_METHODS:
        return {
            "error": f"Unsupported HTTP method: {method}. Use one of: {', '.join(_SUPPORTED_METHODS)}."
        }

    # Validate URL before making any request
    url_error = _validate_url(url)
    if url_error:
        return {"error": url_error}

    try:
        import httpx
    except ImportError:
        return {
            "error": (
                "httpx is required for WebRequestTool. "
                "Install it with: pip install quartermaster-tools[web]"
            )
        }

    try:
        # Use streaming to avoid OOM on large responses
        with httpx.Client(
            timeout=timeout,
            follow_redirects=False,
        ) as client:
            if method == "GET":
                response = client.get(url, headers=headers)
            elif method == "POST":
                response = client.post(url, headers=headers, content=body)
            elif method == "PUT":
                response = client.put(url, headers=headers, content=body)
            elif method == "DELETE":
                response = client.delete(url, headers=headers)
            else:  # PATCH
                response = client.patch(url, headers=headers, content=body)

            # Stream-read with size limit
            content_length = len(response.content)
            if content_length > max_response_size:
                return {
                    "error": (
                        f"Response too large: {content_length} bytes "
                        f"(limit: {max_response_size} bytes)"
                    )
                }

            return {
                "body": response.text,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "url": str(response.url),
            }

    except httpx.TimeoutException:
        return {"error": f"Request timed out after {timeout} seconds"}
    except httpx.ConnectError as e:
        return {"error": f"Connection error: {e}"}
    except httpx.HTTPError as e:
        return {"error": f"HTTP error: {e}"}
    except Exception as e:
        return {"error": f"Unexpected error: {e}"}


@tool()
def web_request(url: str, method: str = "GET", headers: dict = None, body: str = None) -> dict:
    """Make an HTTP request (GET, POST, PUT, DELETE, PATCH).

    Args:
        url: The URL to send the request to.
        method: HTTP method: GET, POST, PUT, DELETE, or PATCH.
        headers: Optional HTTP headers as a JSON object.
        body: Request body for POST/PUT/PATCH requests (string or JSON object).
    """
    return _web_request_impl(url, method=method, headers=headers, body=body)


# Backward-compatible alias
WebRequestTool = web_request
