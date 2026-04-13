"""
Agent session management tools for parallel agent execution.

Provides tools and core classes for creating, running, monitoring,
and collecting results from concurrent agent sessions.
"""

from quartermaster_tools.builtin.agents.session import (
    AgentMessage,
    AgentSession,
    SessionManager,
    SessionStatus,
    get_default_manager,
)
from quartermaster_tools.builtin.agents.tools import (
    AddFinishHookTool,
    CancelSessionTool,
    CollectResultsTool,
    CreateSessionTool,
    GetSessionStatusTool,
    InjectMessageTool,
    ListSessionsTool,
    SpawnAgentTool,
    StartSessionTool,
    WaitSessionTool,
)

__all__ = [
    "AddFinishHookTool",
    "AgentMessage",
    "AgentSession",
    "CancelSessionTool",
    "CollectResultsTool",
    "CreateSessionTool",
    "GetSessionStatusTool",
    "InjectMessageTool",
    "ListSessionsTool",
    "SessionManager",
    "SpawnAgentTool",
    "SessionStatus",
    "StartSessionTool",
    "WaitSessionTool",
    "get_default_manager",
]
