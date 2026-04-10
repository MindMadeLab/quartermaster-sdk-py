"""API key authentication middleware.

Supports two modes:
- API key via X-API-Key header (multiple keys, comma-separated)
- Bearer token via Authorization header

When no keys/tokens are configured, authentication is disabled and
all requests are allowed through.
"""

from __future__ import annotations

from typing import Optional

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Populated at startup from settings
_api_keys: list[str] = []
_auth_token: str | None = None
_auth_enabled: bool = False


def configure_auth(
    api_keys: list[str] | None = None,
    auth_token: str | None = None,
) -> None:
    """Configure authentication at startup.

    Args:
        api_keys: List of valid API keys for X-API-Key header.
        auth_token: Bearer token for Authorization header.
    """
    global _api_keys, _auth_token, _auth_enabled
    _api_keys = list(api_keys) if api_keys else []
    _auth_token = auth_token
    _auth_enabled = bool(_api_keys) or bool(_auth_token)


async def verify_auth(
    request: Request,
    api_key: Optional[str] = Security(api_key_header),
) -> Optional[str]:
    """Verify request authentication.

    Checks X-API-Key header first, then Authorization Bearer token.
    If no auth is configured, allows all requests.

    Returns:
        The authenticated key/token, or None if auth is disabled.

    Raises:
        HTTPException: If authentication fails.
    """
    if not _auth_enabled:
        return None

    # Check X-API-Key header
    if api_key and api_key in _api_keys:
        return api_key

    # Check Authorization Bearer token
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer ") and _auth_token:
        token = auth_header[7:]
        if token == _auth_token:
            return token

    # If we have an API key but it's invalid
    if api_key:
        raise HTTPException(status_code=403, detail="Invalid API Key")

    # No credentials provided
    raise HTTPException(
        status_code=401,
        detail="Not authenticated. Provide X-API-Key header or Authorization Bearer token.",
    )
