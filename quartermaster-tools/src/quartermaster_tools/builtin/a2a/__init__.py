"""
A2A (Agent-to-Agent) protocol tools for agent discovery and task management.

Implements Google's A2A protocol for inter-agent communication:
- a2a_discover: fetch a remote agent's Agent Card
- a2a_send_task: send a task to a remote agent
- a2a_check_status: poll task status
- a2a_collect_result: retrieve completed task results
- a2a_register: generate a local Agent Card
"""

from quartermaster_tools.builtin.a2a.discover import a2a_discover
from quartermaster_tools.builtin.a2a.register import a2a_register
from quartermaster_tools.builtin.a2a.task import (
    a2a_check_status,
    a2a_collect_result,
    a2a_send_task,
)

__all__ = [
    "a2a_check_status",
    "a2a_collect_result",
    "a2a_discover",
    "a2a_register",
    "a2a_send_task",
]
