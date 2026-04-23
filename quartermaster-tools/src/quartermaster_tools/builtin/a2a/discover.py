"""
a2a_discover: Discover remote agent capabilities via the A2A protocol.

Fetches the Agent Card from a remote agent's well-known endpoint to learn
about its name, skills, and capabilities.  Requires ``httpx``.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

from quartermaster_tools.decorator import tool

# Private/reserved IP networks that must not be accessed (SSRF protection)
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("fe80::/10"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("::/128"),
]


def _is_private_url(url: str) -> bool:
    """Return True if the URL's hostname resolves to a private/reserved IP."""
    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        return False

    # Check if hostname is a literal IP address first
    try:
        ip = ipaddress.ip_address(hostname)
        return any(ip in net for net in _BLOCKED_NETWORKS)
    except ValueError:
        pass

    # Resolve hostname via DNS
    try:
        addr_infos = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return False

    for _, _, _, _, sockaddr in addr_infos:
        ip = ipaddress.ip_address(sockaddr[0])
        if any(ip in net for net in _BLOCKED_NETWORKS):
            return True
    return False


def _validate_agent_url(url: str) -> str | None:
    """Validate agent URL. Returns error message or None."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"Only http and https schemes are allowed, got: {parsed.scheme!r}"
    if not parsed.hostname:
        return "URL must include a hostname"
    if _is_private_url(url):
        return "Access denied: requests to private/internal networks are not allowed"
    return None


DEFAULT_TIMEOUT = 30


@tool()
def a2a_discover(agent_url: str) -> dict:
    """Discover a remote A2A agent's capabilities.

    Fetches the Agent Card from a remote agent's
    /.well-known/agent.json endpoint and returns the agent's
    name, description, version, skills, and capabilities.

    Args:
        agent_url: Base URL of the remote A2A agent (e.g. https://agent.example.com).
    """
    if not agent_url:
        raise ValueError("Parameter 'agent_url' is required")

    url_error = _validate_agent_url(agent_url)
    if url_error:
        raise ValueError(url_error)

    try:
        import httpx
    except ImportError:
        raise ImportError(
            "httpx is required for A2ADiscoverTool. "
            "Install it with: pip install quartermaster-tools[web]"
        )

    card_url = agent_url.rstrip("/") + "/.well-known/agent.json"

    try:
        with httpx.Client(timeout=DEFAULT_TIMEOUT, follow_redirects=True) as client:
            response = client.get(card_url)

        if response.status_code != 200:
            raise RuntimeError(f"Agent card request failed with status {response.status_code}")

        card = response.json()
        return {
            "agent_name": card.get("name", ""),
            "description": card.get("description", ""),
            "skills": card.get("skills", []),
            "capabilities": card.get("capabilities", {}),
            "version": card.get("version", ""),
        }

    except httpx.TimeoutException:
        raise TimeoutError(f"Request timed out after {DEFAULT_TIMEOUT} seconds")
    except httpx.ConnectError as e:
        raise ConnectionError(f"Connection error: {e}")
    except httpx.HTTPError as e:
        raise RuntimeError(f"HTTP error: {e}")
