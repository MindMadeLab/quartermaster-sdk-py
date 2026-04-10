"""
A2A task tools: send tasks, check status, and collect results.

Implements JSON-RPC 2.0 over HTTP for the A2A protocol's task lifecycle.
Requires ``httpx``.
"""

from __future__ import annotations

import uuid
from typing import Any

from quartermaster_tools.base import AbstractTool
from quartermaster_tools.builtin.a2a.discover import _validate_agent_url
from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolResult

DEFAULT_TIMEOUT = 60


class A2ASendTaskTool(AbstractTool):
    """Send a task to a remote A2A agent.

    Posts a JSON-RPC ``tasks/send`` request to the agent's ``/a2a`` endpoint
    and returns the task ID, status, and any artifacts.

    Requires httpx (``pip install quartermaster-tools[web]``).
    """

    def __init__(self, timeout: int = DEFAULT_TIMEOUT) -> None:
        self._timeout = timeout

    def name(self) -> str:
        return "a2a_send_task"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="agent_url",
                description="Base URL of the remote A2A agent.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="task_message",
                description="The task instruction to send to the agent.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="task_id",
                description="Optional task ID (auto-generated uuid4 if omitted).",
                type="string",
                required=False,
            ),
            ToolParameter(
                name="metadata",
                description="Optional metadata dict to include with the task.",
                type="object",
                required=False,
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Send a task to a remote A2A agent.",
            long_description=(
                "Sends a task/instruction to a remote A2A agent using "
                "JSON-RPC 2.0 over HTTP and returns the task ID, status, "
                "and any initial artifacts."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=False,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        agent_url: str = kwargs.get("agent_url", "")
        task_message: str = kwargs.get("task_message", "")
        task_id: str = kwargs.get("task_id", "") or str(uuid.uuid4())
        metadata: dict[str, Any] = kwargs.get("metadata") or {}

        if not agent_url:
            return ToolResult(success=False, error="Parameter 'agent_url' is required")
        if not task_message:
            return ToolResult(success=False, error="Parameter 'task_message' is required")

        url_error = _validate_agent_url(agent_url)
        if url_error:
            return ToolResult(success=False, error=url_error)

        try:
            import httpx
        except ImportError:
            return ToolResult(
                success=False,
                error=(
                    "httpx is required for A2ASendTaskTool. "
                    "Install it with: pip install quartermaster-tools[web]"
                ),
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
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(endpoint, json=payload)

            body = response.json()

            if "error" in body:
                err = body["error"]
                return ToolResult(
                    success=False,
                    error=f"JSON-RPC error {err.get('code', '?')}: {err.get('message', '')}",
                )

            result = body.get("result", {})
            status = result.get("status", {})
            return ToolResult(
                success=True,
                data={
                    "task_id": result.get("id", task_id),
                    "status": status.get("state", "unknown"),
                    "artifacts": result.get("artifacts", []),
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


class A2ACheckStatusTool(AbstractTool):
    """Check the status of a previously sent A2A task.

    Posts a JSON-RPC ``tasks/get`` request to retrieve the current status
    and any artifacts of the specified task.

    Requires httpx (``pip install quartermaster-tools[web]``).
    """

    def __init__(self, timeout: int = DEFAULT_TIMEOUT) -> None:
        self._timeout = timeout

    def name(self) -> str:
        return "a2a_check_status"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="agent_url",
                description="Base URL of the remote A2A agent.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="task_id",
                description="ID of the task to check.",
                type="string",
                required=True,
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Check the status of a remote A2A task.",
            long_description=(
                "Retrieves the current status and artifacts of a previously "
                "sent A2A task using JSON-RPC 2.0."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=False,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        agent_url: str = kwargs.get("agent_url", "")
        task_id: str = kwargs.get("task_id", "")

        if not agent_url:
            return ToolResult(success=False, error="Parameter 'agent_url' is required")
        if not task_id:
            return ToolResult(success=False, error="Parameter 'task_id' is required")

        url_error = _validate_agent_url(agent_url)
        if url_error:
            return ToolResult(success=False, error=url_error)

        try:
            import httpx
        except ImportError:
            return ToolResult(
                success=False,
                error=(
                    "httpx is required for A2ACheckStatusTool. "
                    "Install it with: pip install quartermaster-tools[web]"
                ),
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
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(endpoint, json=payload)

            body = response.json()

            if "error" in body:
                err = body["error"]
                return ToolResult(
                    success=False,
                    error=f"JSON-RPC error {err.get('code', '?')}: {err.get('message', '')}",
                )

            result = body.get("result", {})
            status = result.get("status", {})
            return ToolResult(
                success=True,
                data={
                    "task_id": result.get("id", task_id),
                    "status": status,
                    "artifacts": result.get("artifacts", []),
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


class A2ACollectResultTool(AbstractTool):
    """Collect completed results from a remote A2A agent task.

    Similar to CheckStatus but specifically extracts and formats the
    artifacts/results from a completed task.

    Requires httpx (``pip install quartermaster-tools[web]``).
    """

    def __init__(self, timeout: int = DEFAULT_TIMEOUT) -> None:
        self._timeout = timeout

    def name(self) -> str:
        return "a2a_collect_result"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="agent_url",
                description="Base URL of the remote A2A agent.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="task_id",
                description="ID of the task whose results to collect.",
                type="string",
                required=True,
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Collect results from a completed A2A task.",
            long_description=(
                "Retrieves and formats the artifacts/results from a "
                "completed A2A task. Returns structured result data "
                "including completion status and extracted text content."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=False,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        agent_url: str = kwargs.get("agent_url", "")
        task_id: str = kwargs.get("task_id", "")

        if not agent_url:
            return ToolResult(success=False, error="Parameter 'agent_url' is required")
        if not task_id:
            return ToolResult(success=False, error="Parameter 'task_id' is required")

        url_error = _validate_agent_url(agent_url)
        if url_error:
            return ToolResult(success=False, error=url_error)

        try:
            import httpx
        except ImportError:
            return ToolResult(
                success=False,
                error=(
                    "httpx is required for A2ACollectResultTool. "
                    "Install it with: pip install quartermaster-tools[web]"
                ),
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
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(endpoint, json=payload)

            body = response.json()

            if "error" in body:
                err = body["error"]
                return ToolResult(
                    success=False,
                    error=f"JSON-RPC error {err.get('code', '?')}: {err.get('message', '')}",
                )

            result = body.get("result", {})
            status = result.get("status", {})
            state = status.get("state", "unknown")
            completed = state == "completed"
            artifacts = result.get("artifacts", [])

            # Extract text content from artifact parts
            results: list[dict[str, str]] = []
            for artifact in artifacts:
                for part in artifact.get("parts", []):
                    results.append({
                        "type": part.get("type", "text"),
                        "content": part.get("text", ""),
                    })

            return ToolResult(
                success=True,
                data={
                    "task_id": result.get("id", task_id),
                    "completed": completed,
                    "results": results,
                    "status": state,
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
