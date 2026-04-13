"""
a2a_register: Generate an A2A Agent Card for local agent registration.

Uses only the standard library (json, pathlib) -- no external dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quartermaster_tools.decorator import tool


@tool()
def a2a_register(
    name: str,
    description: str,
    url: str,
    skills: list,
    version: str = "1.0.0",
    streaming: bool = False,
    push_notifications: bool = False,
    output_path: str = "",
) -> dict:
    """Generate an A2A Agent Card for local agent registration.

    Builds a valid A2A Agent Card JSON document describing the
    local agent's name, description, URL, skills, and protocol
    capabilities.  Can optionally save the card to a file.

    Args:
        name: Name of the agent.
        description: Short description of the agent.
        url: Public URL where the agent is reachable.
        skills: List of skill dicts, each with 'id', 'name', and 'description' keys.
        version: Agent version string.
        streaming: Whether the agent supports streaming responses.
        push_notifications: Whether the agent supports push notifications.
        output_path: Optional file path to save the Agent Card JSON.
    """
    agent_name = name
    agent_version = version or "1.0.0"

    if not agent_name:
        raise ValueError("Parameter 'name' is required")
    if not description:
        raise ValueError("Parameter 'description' is required")
    if not url:
        raise ValueError("Parameter 'url' is required")
    if not skills:
        raise ValueError("Parameter 'skills' is required")

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
            raise OSError(f"Failed to write agent card to {output_path}: {e}")

    return {
        "agent_card": agent_card,
        "saved_to": saved_to,
    }


# Backward-compatible alias
A2ARegisterTool = a2a_register
