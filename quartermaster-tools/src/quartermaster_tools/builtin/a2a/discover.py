"""
A2ADiscoverTool: Discover remote agent capabilities via the A2A protocol.

Fetches the Agent Card from a remote agent's well-known endpoint to learn
about its name, skills, and capabilities.  Requires ``httpx``.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Any
from urllib.parse import urlparse

from quartermaster_tools.base import AbstractTool
from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolResult

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


class A2ADiscoverTool(AbstractTool):
    """Discover a remote A2A agent by fetching its Agent Card.

    Sends a GET request to ``{agent_url}/.well-known/agent.json`` and parses
    the returned Agent Card JSON to extract the agent's name, description,
    version, skills, and capabilities.

    Requires httpx (``pip install quartermaster-tools[web]``).
    """

    def __init__(self, timeout: int = DEFAULT_TIMEOUT) -> None:
        self._timeout = timeout

    def name(self) -> str:
        return "a2a_discover"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="agent_url",
                description="Base URL of the remote A2A agent (e.g. https://agent.example.com).",
                type="string",
                required=True,
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Discover a remote A2A agent's capabilities.",
            long_description=(
                "Fetches the Agent Card from a remote agent's "
                "/.well-known/agent.json endpoint and returns the agent's "
                "name, description, version, skills, and capabilities."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=False,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        agent_url: str = kwargs.get("agent_url", "")
        if not agent_url:
            return ToolResult(success=False, error="Parameter 'agent_url' is required")

        url_error = _validate_agent_url(agent_url)
        if url_error:
            return ToolResult(success=False, error=url_error)

        try:
            import httpx
        except ImportError:
            return ToolResult(
                success=False,
                error=(
                    "httpx is required for A2ADiscoverTool. "
                    "Install it with: pip install quartermaster-tools[web]"
                ),
            )

        card_url = agent_url.rstrip("/") + "/.well-known/agent.json"

        try:
            with httpx.Client(timeout=self._timeout, follow_redirects=True) as client:
                response = client.get(card_url)

            if response.status_code != 200:
                return ToolResult(
                    success=False,
                    error=f"Agent card request failed with status {response.status_code}",
                )

            card = response.json()
            return ToolResult(
                success=True,
                data={
                    "agent_name": card.get("name", ""),
                    "description": card.get("description", ""),
                    "skills": card.get("skills", []),
                    "capabilities": card.get("capabilities", {}),
                    "version": card.get("version", ""),
                },
            )

        except httpx.TimeoutException:
            return ToolResult(
                success=False,
                error=f"Request timed out after {self._timeout} seconds",
            )
        except httpx.ConnectError as e:
            return ToolResult(success=False, error=f"Connection error: {e}")
        except httpx.HTTPError as e:
            return ToolResult(success=False, error=f"HTTP error: {e}")
        except Exception as e:
            return ToolResult(success=False, error=f"Unexpected error: {e}")
