"""
A2ARegisterTool: Generate an A2A Agent Card for local agent registration.

Uses only the standard library (json, pathlib) -- no external dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quartermaster_tools.base import AbstractTool
from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolResult


class A2ARegisterTool(AbstractTool):
    """Generate an A2A Agent Card JSON document.

    Builds a valid Agent Card describing the local agent's capabilities,
    skills, and protocol features.  Optionally writes the card to a file.

    No external dependencies required.
    """

    def name(self) -> str:
        return "a2a_register"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="name",
                description="Name of the agent.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="description",
                description="Short description of the agent.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="url",
                description="Public URL where the agent is reachable.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="skills",
                description=(
                    "List of skill dicts, each with 'id', 'name', and 'description' keys."
                ),
                type="array",
                required=True,
            ),
            ToolParameter(
                name="version",
                description="Agent version string.",
                type="string",
                required=False,
                default="1.0.0",
            ),
            ToolParameter(
                name="streaming",
                description="Whether the agent supports streaming responses.",
                type="boolean",
                required=False,
                default=False,
            ),
            ToolParameter(
                name="push_notifications",
                description="Whether the agent supports push notifications.",
                type="boolean",
                required=False,
                default=False,
            ),
            ToolParameter(
                name="output_path",
                description="Optional file path to save the Agent Card JSON.",
                type="string",
                required=False,
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Generate an A2A Agent Card for local agent registration.",
            long_description=(
                "Builds a valid A2A Agent Card JSON document describing the "
                "local agent's name, description, URL, skills, and protocol "
                "capabilities.  Can optionally save the card to a file."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        agent_name: str = kwargs.get("name", "")
        description: str = kwargs.get("description", "")
        url: str = kwargs.get("url", "")
        skills: list[dict[str, str]] = kwargs.get("skills") or []
        agent_version: str = kwargs.get("version", "1.0.0") or "1.0.0"
        streaming: bool = kwargs.get("streaming", False)
        push_notifications: bool = kwargs.get("push_notifications", False)
        output_path: str | None = kwargs.get("output_path")

        if not agent_name:
            return ToolResult(success=False, error="Parameter 'name' is required")
        if not description:
            return ToolResult(success=False, error="Parameter 'description' is required")
        if not url:
            return ToolResult(success=False, error="Parameter 'url' is required")
        if not skills:
            return ToolResult(success=False, error="Parameter 'skills' is required")

        agent_card: dict[str, Any] = {
            "name": agent_name,
            "description": description,
            "url": url,
            "version": agent_version,
            "skills": [
                {
                    "id": skill.get("id", ""),
                    "name": skill.get("name", ""),
                    "description": skill.get("description", ""),
                }
                for skill in skills
            ],
            "capabilities": {
                "streaming": streaming,
                "pushNotifications": push_notifications,
            },
        }

        saved_to: str | None = None
        if output_path:
            try:
                path = Path(output_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(agent_card, indent=2) + "\n")
                saved_to = str(path)
            except OSError as e:
                return ToolResult(
                    success=False,
                    error=f"Failed to write agent card to {output_path}: {e}",
                )

        return ToolResult(
            success=True,
            data={
                "agent_card": agent_card,
                "saved_to": saved_to,
            },
        )
