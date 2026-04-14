"""
JSON API caller with optional JMESPath filtering.

Makes HTTP requests expecting JSON responses, auto-parses the response,
and optionally filters the result using JMESPath expressions.
"""

from __future__ import annotations

import json
from typing import Any

from quartermaster_tools.builtin.web_request import _validate_url
from quartermaster_tools.decorator import tool
from quartermaster_tools.types import ToolResult

_DEFAULT_TIMEOUT = 30
_SUPPORTED_METHODS = ("GET", "POST", "PUT", "DELETE", "PATCH")


def _apply_jmespath(data: Any, expression: str) -> Any:
    """Apply a JMESPath filter to JSON data.

    Args:
        data: The parsed JSON data.
        expression: JMESPath expression string.

    Returns:
        Filtered data, or raises on error.
    """
    try:
        import jmespath
    except ImportError:
        raise ImportError(
            "jmespath library is required for JMESPath filtering. "
            "Install it with: pip install jmespath"
        )

    try:
        return jmespath.search(expression, data)
    except Exception as e:
        raise ValueError(f"JMESPath filter error: {e}")


@tool()
def json_api(
    url: str,
    method: str = "GET",
    headers: dict = None,
    body: dict = None,
    jmespath_filter: str = None,
) -> dict:
    """Call a JSON API and return parsed results with optional JMESPath filtering.

    Makes an HTTP request to a JSON API endpoint, automatically parses
    the JSON response, and optionally filters it with a JMESPath
    expression. Supports all standard HTTP methods.

    Args:
        url: The API endpoint URL.
        method: HTTP method (default GET).
        headers: Optional HTTP headers as a JSON object.
        body: Request body (will be serialised as JSON if dict/list).
        jmespath_filter: Optional JMESPath expression to filter the JSON response.
    """
    url = url.strip() if url else ""
    method = method.upper() if method else "GET"

    if not url:
        raise ValueError("Parameter 'url' is required")

    # SSRF protection: block private/internal network access
    url_error = _validate_url(url)
    if url_error:
        raise ValueError(url_error)

    if method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported HTTP method: {method}. Use one of: {', '.join(_SUPPORTED_METHODS)}."
        )

    try:
        import httpx
    except ImportError:
        raise ImportError(
            "httpx is required for JsonApiTool. "
            "Install it with: pip install quartermaster-tools[web]"
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
        raise TimeoutError("API request timed out")
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
        raise RuntimeError(f"HTTP error: {e}")

    # If we got a ToolResult from error handling above, return it directly
    # (the decorator will pass it through)
    if isinstance(json_data, ToolResult):
        return json_data

    # Apply JMESPath filter if provided
    filtered_data = json_data
    if jmespath_filter:
        filtered_data = _apply_jmespath(json_data, jmespath_filter)

    return {
        "json": filtered_data,
        "status_code": response.status_code,
        "headers": dict(response.headers),
    }


# Backward-compatible alias
JsonApiTool = json_api
