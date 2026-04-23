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
    add_agent_finish_hook,
    set_manager,
    cancel_agent_session,
    collect_agent_results,
    create_agent_session,
    get_agent_session_status,
    inject_agent_message,
    list_agent_sessions,
    notify_parent,
    spawn_agent,
    start_agent_session,
    wait_agent_session,
)

__all__ = [
    "AgentMessage",
    "AgentSession",
    "SessionManager",
    "SessionStatus",
    "add_agent_finish_hook",
    "cancel_agent_session",
    "collect_agent_results",
    "create_agent_session",
    "get_agent_session_status",
    "get_default_manager",
    "inject_agent_message",
    "list_agent_sessions",
    "notify_parent",
    "set_manager",
    "spawn_agent",
    "start_agent_session",
    "wait_agent_session",
]
