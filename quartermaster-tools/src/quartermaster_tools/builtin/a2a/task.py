"""
A2A task tools: send tasks, check status, and collect results.

Implements JSON-RPC 2.0 over HTTP for the A2A protocol's task lifecycle.
Requires ``httpx``.
"""

from __future__ import annotations

import uuid

from quartermaster_tools.builtin.a2a.discover import _validate_agent_url
from quartermaster_tools.decorator import tool

DEFAULT_TIMEOUT = 60


@tool()
def a2a_send_task(
    agent_url: str,
    task_message: str,
    task_id: str = "",
    metadata: dict = None,
) -> dict:
    """Send a task to a remote A2A agent.

    Sends a task/instruction to a remote A2A agent using
    JSON-RPC 2.0 over HTTP and returns the task ID, status,
    and any initial artifacts.

    Args:
        agent_url: Base URL of the remote A2A agent.
        task_message: The task instruction to send to the agent.
        task_id: Optional task ID (auto-generated uuid4 if omitted).
        metadata: Optional metadata dict to include with the task.
    """
    if not agent_url:
        raise ValueError("Parameter 'agent_url' is required")
    if not task_message:
        raise ValueError("Parameter 'task_message' is required")

    task_id = task_id or str(uuid.uuid4())
    metadata = metadata or {}

    url_error = _validate_agent_url(agent_url)
    if url_error:
        raise ValueError(url_error)

    try:
        import httpx
    except ImportError:
        raise ImportError(
            "httpx is required for A2ASendTaskTool. "
            "Install it with: pip install quartermaster-tools[web]"
        )

    endpoint = agent_url.rstrip("/") + "/a2a"
    request_id = str(uuid.uuid4())
    payload = {
        "jsonrpc": "2.0",
        "method": "tasks/send",
        "id": request_id,
        "params": {
            "id": task_id,
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": task_message}],
            },
            "metadata": metadata,
        },
    }

    try:
        with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
            response = client.post(endpoint, json=payload)

        body = response.json()

        if "error" in body:
            err = body["error"]
            raise RuntimeError(f"JSON-RPC error {err.get('code', '?')}: {err.get('message', '')}")

        result = body.get("result", {})
        status = result.get("status", {})
        return {
            "task_id": result.get("id", task_id),
            "status": status.get("state", "unknown"),
            "artifacts": result.get("artifacts", []),
        }

    except httpx.TimeoutException:
        raise TimeoutError(f"Request timed out after {DEFAULT_TIMEOUT} seconds")
    except httpx.ConnectError as e:
        raise ConnectionError(f"Connection error: {e}")
    except httpx.HTTPError as e:
        raise RuntimeError(f"HTTP error: {e}")


@tool()
def a2a_check_status(agent_url: str, task_id: str) -> dict:
    """Check the status of a remote A2A task.

    Retrieves the current status and artifacts of a previously
    sent A2A task using JSON-RPC 2.0.

    Args:
        agent_url: Base URL of the remote A2A agent.
        task_id: ID of the task to check.
    """
    if not agent_url:
        raise ValueError("Parameter 'agent_url' is required")
    if not task_id:
        raise ValueError("Parameter 'task_id' is required")

    url_error = _validate_agent_url(agent_url)
    if url_error:
        raise ValueError(url_error)

    try:
        import httpx
    except ImportError:
        raise ImportError(
            "httpx is required for A2ACheckStatusTool. "
            "Install it with: pip install quartermaster-tools[web]"
        )

    endpoint = agent_url.rstrip("/") + "/a2a"
    request_id = str(uuid.uuid4())
    payload = {
        "jsonrpc": "2.0",
        "method": "tasks/get",
        "id": request_id,
        "params": {"id": task_id},
    }

    try:
        with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
            response = client.post(endpoint, json=payload)

        body = response.json()

        if "error" in body:
            err = body["error"]
            raise RuntimeError(f"JSON-RPC error {err.get('code', '?')}: {err.get('message', '')}")

        result = body.get("result", {})
        status = result.get("status", {})
        return {
            "task_id": result.get("id", task_id),
            "status": status,
            "artifacts": result.get("artifacts", []),
        }

    except httpx.TimeoutException:
        raise TimeoutError(f"Request timed out after {DEFAULT_TIMEOUT} seconds")
    except httpx.ConnectError as e:
        raise ConnectionError(f"Connection error: {e}")
    except httpx.HTTPError as e:
        raise RuntimeError(f"HTTP error: {e}")


@tool()
def a2a_collect_result(agent_url: str, task_id: str) -> dict:
    """Collect results from a completed A2A task.

    Retrieves and formats the artifacts/results from a
    completed A2A task. Returns structured result data
    including completion status and extracted text content.

    Args:
        agent_url: Base URL of the remote A2A agent.
        task_id: ID of the task whose results to collect.
    """
    if not agent_url:
        raise ValueError("Parameter 'agent_url' is required")
    if not task_id:
        raise ValueError("Parameter 'task_id' is required")

    url_error = _validate_agent_url(agent_url)
    if url_error:
        raise ValueError(url_error)

    try:
        import httpx
    except ImportError:
        raise ImportError(
            "httpx is required for A2ACollectResultTool. "
            "Install it with: pip install quartermaster-tools[web]"
        )

    endpoint = agent_url.rstrip("/") + "/a2a"
    request_id = str(uuid.uuid4())
    payload = {
        "jsonrpc": "2.0",
        "method": "tasks/get",
        "id": request_id,
        "params": {"id": task_id},
    }

    try:
        with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
            response = client.post(endpoint, json=payload)

        body = response.json()

        if "error" in body:
            err = body["error"]
            raise RuntimeError(f"JSON-RPC error {err.get('code', '?')}: {err.get('message', '')}")

        result = body.get("result", {})
        status = result.get("status", {})
        state = status.get("state", "unknown")
        completed = state == "completed"
        artifacts = result.get("artifacts", [])

        # Extract text content from artifact parts
        results: list[dict[str, str]] = []
        for artifact in artifacts:
            for part in artifact.get("parts", []):
                results.append(
                    {
                        "type": part.get("type", "text"),
                        "content": part.get("text", ""),
                    }
                )

        return {
            "task_id": result.get("id", task_id),
            "completed": completed,
            "results": results,
            "status": state,
        }

    except httpx.TimeoutException:
        raise TimeoutError(f"Request timed out after {DEFAULT_TIMEOUT} seconds")
    except httpx.ConnectError as e:
        raise ConnectionError(f"Connection error: {e}")
    except httpx.HTTPError as e:
        raise RuntimeError(f"HTTP error: {e}")


# Backward-compatible aliases
A2ASendTaskTool = a2a_send_task
A2ACheckStatusTool = a2a_check_status
A2ACollectResultTool = a2a_collect_result
