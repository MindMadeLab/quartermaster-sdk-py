"""Quartermaster Cloud client — upload, manage, and run agent graphs.

Provides a simple interface for uploading agent graphs built with the
fluent ``GraphBuilder`` API to Quartermaster Cloud for execution and
management.

Usage::

    from quartermaster_graph import Graph
    from quartermaster_graph.cloud import QuartermasterCloud

    agent = (
        Graph("My Agent")
        .start()
        .user("Input")
        .instruction("Process", system_instruction="Handle the request")
        .end()
    )

    cloud = QuartermasterCloud()  # uses QUARTERMASTER_API_KEY env var
    result = cloud.upload(agent, version="1.0.0", publish=True)
    print(f"Uploaded: {result['id']}")
"""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from quartermaster_graph.builder import GraphBuilder
from quartermaster_graph.serialization import to_json
from quartermaster_graph.validation import validate_graph

QUARTERMASTER_API_URL = "https://api.quartermaster.ai"


class CloudError(Exception):
    """Error from Quartermaster Cloud API."""

    def __init__(self, message: str, status_code: int | None = None, body: str = ""):
        self.status_code = status_code
        self.body = body
        super().__init__(message)


class QuartermasterCloud:
    """Client for uploading and managing agent graphs on Quartermaster Cloud.

    Args:
        api_key: Quartermaster API key (``qm-...``).
            Falls back to ``QUARTERMASTER_API_KEY`` environment variable.
        base_url: Override the API endpoint.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("QUARTERMASTER_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Quartermaster API key required. "
                "Pass api_key= or set QUARTERMASTER_API_KEY environment variable."
            )
        self.base_url = (
            base_url
            or os.environ.get("QUARTERMASTER_API_URL", QUARTERMASTER_API_URL)
        ).rstrip("/")

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make an authenticated request to the Quartermaster Cloud API."""
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "quartermaster-graph/0.1.0",
        }

        data = json.dumps(body).encode("utf-8") if body else None
        req = Request(url, data=data, headers=headers, method=method)

        try:
            with urlopen(req, timeout=30) as resp:
                response_body = resp.read().decode("utf-8")
                return json.loads(response_body) if response_body else {}
        except HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            raise CloudError(
                f"API request failed: {e.code} {e.reason}",
                status_code=e.code,
                body=error_body,
            ) from e

    # ------------------------------------------------------------------
    # Agent management
    # ------------------------------------------------------------------

    def create_agent(
        self,
        name: str,
        description: str = "",
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a new agent on Quartermaster Cloud.

        Returns:
            Agent metadata including ``id``.
        """
        return self._request("POST", "/v1/agents", {
            "name": name,
            "description": description,
            "tags": tags or [],
        })

    def list_agents(self) -> list[dict[str, Any]]:
        """List all agents in the account."""
        result = self._request("GET", "/v1/agents")
        return result.get("data", [])

    def get_agent(self, agent_id: str) -> dict[str, Any]:
        """Get agent details by ID."""
        return self._request("GET", f"/v1/agents/{agent_id}")

    def delete_agent(self, agent_id: str) -> dict[str, Any]:
        """Delete an agent."""
        return self._request("DELETE", f"/v1/agents/{agent_id}")

    # ------------------------------------------------------------------
    # Version management
    # ------------------------------------------------------------------

    def upload(
        self,
        graph: GraphBuilder,
        name: str | None = None,
        description: str = "",
        version: str = "0.1.0",
        publish: bool = False,
        agent_id: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Upload a graph to Quartermaster Cloud.

        If ``agent_id`` is not provided, creates a new agent first using
        the graph's name.

        Args:
            graph: The ``GraphBuilder`` (or ``Graph``) instance.
            name: Agent name (defaults to the graph's internal name).
            description: Agent description.
            version: Semantic version string.
            publish: Whether to publish the version immediately.
            agent_id: Existing agent ID to upload a new version to.
            tags: Optional tags for the agent.

        Returns:
            Version metadata from the API.

        Raises:
            ValueError: If the graph fails local validation.
            CloudError: If the API request fails.
        """
        # Local validation first
        agent_version = graph.to_version(validate=True, version=version)
        graph_data = to_json(agent_version)

        # Create agent if needed
        if agent_id is None:
            agent_name = name or graph._name
            agent_meta = self.create_agent(
                name=agent_name,
                description=description,
                tags=tags or [],
            )
            agent_id = agent_meta["id"]

        # Upload the version
        result = self._request("POST", f"/v1/agents/{agent_id}/versions", {
            "version": version,
            "graph": graph_data,
            "features": "",
            "publish": publish,
        })

        result["agent_id"] = agent_id
        return result

    def list_versions(self, agent_id: str) -> list[dict[str, Any]]:
        """List all versions of an agent."""
        result = self._request("GET", f"/v1/agents/{agent_id}/versions")
        return result.get("data", [])

    def publish(self, agent_id: str, version_id: str) -> dict[str, Any]:
        """Publish a specific version."""
        return self._request(
            "POST", f"/v1/agents/{agent_id}/versions/{version_id}/publish"
        )

    def rollback(self, agent_id: str, version_id: str) -> dict[str, Any]:
        """Rollback to a specific version."""
        return self._request(
            "POST", f"/v1/agents/{agent_id}/versions/{version_id}/rollback"
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        agent_id: str,
        input_text: str,
        variables: dict[str, str] | None = None,
        version: str | None = None,
    ) -> dict[str, Any]:
        """Execute an agent and return the result.

        Args:
            agent_id: The agent to execute.
            input_text: User input to the agent.
            variables: Pre-set variables for the execution.
            version: Specific version to run (defaults to published).

        Returns:
            Execution result from the API.
        """
        body: dict[str, Any] = {"input": input_text}
        if variables:
            body["variables"] = variables
        if version:
            body["version"] = version

        return self._request("POST", f"/v1/agents/{agent_id}/run", body)
