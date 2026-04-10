"""
A2A (Agent-to-Agent) protocol tools for agent discovery and task management.

Implements Google's A2A protocol for inter-agent communication:
- A2ADiscoverTool: fetch a remote agent's Agent Card
- A2ASendTaskTool: send a task to a remote agent
- A2ACheckStatusTool: poll task status
- A2ACollectResultTool: retrieve completed task results
- A2ARegisterTool: generate a local Agent Card
"""

from quartermaster_tools.builtin.a2a.discover import A2ADiscoverTool
from quartermaster_tools.builtin.a2a.register import A2ARegisterTool
from quartermaster_tools.builtin.a2a.task import (
    A2ACheckStatusTool,
    A2ACollectResultTool,
    A2ASendTaskTool,
)

__all__ = [
    "A2ACheckStatusTool",
    "A2ACollectResultTool",
    "A2ADiscoverTool",
    "A2ARegisterTool",
    "A2ASendTaskTool",
]
