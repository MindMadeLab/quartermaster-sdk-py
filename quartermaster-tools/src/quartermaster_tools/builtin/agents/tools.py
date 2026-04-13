"""
Agent session management tools.

Provides tools for creating, starting, monitoring, and collecting
results from parallel agent sessions.
"""

from __future__ import annotations

import json
import time
from typing import Any

from quartermaster_tools.decorator import tool

from quartermaster_tools.builtin.agents.session import (
    AgentMessage,
    AgentSession,
    SessionManager,
    SessionStatus,
    get_default_manager,
)

_manager_override: SessionManager | None = None


def _get_manager() -> SessionManager:
    """Return the override manager (for testing) or the module-level default."""
    if _manager_override is not None:
        return _manager_override
    return get_default_manager()


def set_manager(manager: SessionManager | None) -> None:
    """Set a custom SessionManager for all agent-session tools.

    Primarily intended for testing.  Pass ``None`` to revert to the
    default module-level manager.
    """
    global _manager_override  # noqa: PLW0603
    _manager_override = manager


@tool()
def create_agent_session(name: str = "", metadata: str = "") -> dict:
    """Create a new parallel agent session.

    Creates a new agent session that can be started with a task.
    Sessions run in separate threads for parallel execution.
    For simple use-cases prefer spawn_agent which combines
    session creation and start into a single call.

    Args:
        name: Optional human-readable name for the session.
        metadata: Optional JSON string of metadata key-value pairs.
    """
    meta: dict[str, Any] = {}
    if metadata:
        try:
            meta = json.loads(metadata)
        except (json.JSONDecodeError, TypeError):
            return {"error": "Invalid JSON in 'metadata' parameter"}

    manager = _get_manager()
    session = manager.create_session(name=name, metadata=meta)
    return {
        "session_id": session.id,
        "name": session.name,
        "status": session.status.value,
    }


@tool()
def start_agent_session(
    session_id: str,
    task: str,
    system_prompt: str = "",
) -> dict:
    """Start an agent session with a task.

    Starts a previously created session with the given task.
    The task runs in a background thread. Use wait or status
    tools to monitor progress. For simple use-cases prefer
    spawn_agent which combines creation and start.

    Args:
        session_id: The session ID to start.
        task: Task description/instructions for the agent.
        system_prompt: Optional system prompt for the agent.
    """
    if not session_id:
        return {"error": "Parameter 'session_id' is required"}
    if not task:
        return {"error": "Parameter 'task' is required"}

    manager = _get_manager()
    session = manager.get_session(session_id)
    if not session:
        return {"error": f"Session '{session_id}' not found"}

    def task_fn(s: AgentSession) -> Any:
        if system_prompt:
            s.messages.append(
                AgentMessage(role="system", content=system_prompt)
            )
        s.messages.append(AgentMessage(role="user", content=task))
        # Framework hook point: a real LLM loop would process here.
        # For now, mark as completed with the task as the result.
        return {"task": task, "message_count": len(s.messages)}

    started = manager.start_session(session_id, task_fn)
    if not started:
        return {
            "error": f"Could not start session '{session_id}' (already running or not found)"
        }

    return {"session_id": session_id, "status": "running"}


@tool()
def inject_agent_message(
    session_id: str,
    content: str,
    role: str = "user",
) -> dict:
    """Inject a message into a running agent session.

    Adds a message to the session's message history.
    Can be used to provide additional context or instructions
    to a running agent.

    Args:
        session_id: The session ID to inject into.
        content: Message content to inject.
        role: Message role (user, assistant, system).
    """
    if not session_id:
        return {"error": "Parameter 'session_id' is required"}
    if not content:
        return {"error": "Parameter 'content' is required"}

    manager = _get_manager()
    injected = manager.inject_message(session_id, role=role, content=content)
    if not injected:
        return {"error": f"Session '{session_id}' not found"}

    session = manager.get_session(session_id)
    return {
        "session_id": session_id,
        "injected": True,
        "message_count": len(session.messages) if session else 0,
    }


@tool()
def get_agent_session_status(session_id: str) -> dict:
    """Get the status of an agent session.

    Returns detailed status information for a session,
    including its current state, message count, and timestamps.

    Args:
        session_id: The session ID to check.
    """
    if not session_id:
        return {"error": "Parameter 'session_id' is required"}

    manager = _get_manager()
    session = manager.get_session(session_id)
    if not session:
        return {"error": f"Session '{session_id}' not found"}

    return {
        "session_id": session.id,
        "status": session.status.value,
        "name": session.name,
        "message_count": len(session.messages),
        "created_at": session.created_at,
        "updated_at": session.updated_at,
    }


@tool()
def list_agent_sessions(status: str = "") -> dict:
    """List all agent sessions.

    Returns a list of all tracked agent sessions,
    optionally filtered by status.

    Args:
        status: Optional status filter (created, running, completed, failed, cancelled).
    """
    manager = _get_manager()

    status_filter: SessionStatus | None = None
    if status:
        try:
            status_filter = SessionStatus(status)
        except ValueError:
            return {
                "error": f"Invalid status '{status}'. Valid values: "
                + ", ".join(s.value for s in SessionStatus)
            }

    sessions = manager.list_sessions(status=status_filter)
    return {
        "sessions": [
            {
                "id": s.id,
                "name": s.name,
                "status": s.status.value,
                "message_count": len(s.messages),
                "created_at": s.created_at,
            }
            for s in sessions
        ],
        "count": len(sessions),
    }


@tool()
def wait_agent_session(session_id: str, timeout: float = 30) -> dict:
    """Wait for an agent session to complete.

    Blocks until the session finishes or the timeout expires.
    Returns the session result and final status.

    Args:
        session_id: The session ID to wait for.
        timeout: Maximum seconds to wait.
    """
    if not session_id:
        return {"error": "Parameter 'session_id' is required"}

    manager = _get_manager()
    session = manager.wait_for_session(session_id, timeout=timeout)
    if not session:
        return {"error": f"Session '{session_id}' not found"}

    return {
        "session_id": session.id,
        "status": session.status.value,
        "result": session.result,
        "error": session.error,
    }


@tool()
def collect_agent_results(session_ids: str, timeout: float = 30) -> dict:
    """Collect results from multiple agent sessions.

    Waits for all specified sessions to complete and
    collects their results. Returns a summary of all sessions.

    Args:
        session_ids: Comma-separated list of session IDs.
        timeout: Maximum seconds to wait per session.
    """
    if not session_ids:
        return {"error": "Parameter 'session_ids' is required"}

    ids = [s.strip() for s in session_ids.split(",") if s.strip()]
    manager = _get_manager()
    sessions = manager.wait_all(ids, timeout=timeout)

    results = [
        {
            "session_id": s.id,
            "status": s.status.value,
            "result": s.result,
            "error": s.error,
        }
        for s in sessions
    ]
    all_completed = all(
        s.status
        in (SessionStatus.COMPLETED, SessionStatus.FAILED, SessionStatus.CANCELLED)
        for s in sessions
    )

    return {
        "results": results,
        "all_completed": all_completed,
    }


@tool()
def cancel_agent_session(session_id: str) -> dict:
    """Cancel a running agent session.

    Marks a session as cancelled. Note that the underlying
    thread may still be running; this sets the status flag.

    Args:
        session_id: The session ID to cancel.
    """
    if not session_id:
        return {"error": "Parameter 'session_id' is required"}

    manager = _get_manager()
    cancelled = manager.cancel_session(session_id)
    if not cancelled:
        return {"error": f"Session '{session_id}' not found"}

    return {"session_id": session_id, "cancelled": True}


@tool()
def add_agent_finish_hook(
    session_id: str,
    hook_type: str,
    hook_config: str = "",
) -> dict:
    """Add a finish hook to an agent session.

    Registers a callback that fires when the session completes.
    Built-in hooks: 'log' writes to a file, 'notify' stores a
    notification in session metadata.

    Args:
        session_id: The session ID to add a hook to.
        hook_type: Type of hook: 'log' or 'notify'.
        hook_config: Optional JSON string of hook configuration.
    """
    if not session_id:
        return {"error": "Parameter 'session_id' is required"}
    if not hook_type:
        return {"error": "Parameter 'hook_type' is required"}
    if hook_type not in ("log", "notify"):
        return {"error": f"Invalid hook_type '{hook_type}'. Must be 'log' or 'notify'."}

    config: dict[str, Any] = {}
    if hook_config:
        try:
            config = json.loads(hook_config)
        except (json.JSONDecodeError, TypeError):
            return {"error": "Invalid JSON in 'hook_config' parameter"}

    manager = _get_manager()
    session = manager.get_session(session_id)
    if not session:
        return {"error": f"Session '{session_id}' not found"}

    if hook_type == "log":
        path = config.get("path", "agent_session.log")

        def log_hook(s: AgentSession) -> None:
            try:
                with open(path, "a") as f:
                    f.write(
                        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                        f"Session {s.id} ({s.name}) finished with "
                        f"status={s.status.value}\n"
                    )
            except OSError:
                pass

        manager.add_finish_hook(session_id, log_hook)
    elif hook_type == "notify":

        def notify_hook(s: AgentSession) -> None:
            s.metadata["notification"] = {
                "type": "finish",
                "status": s.status.value,
                "timestamp": time.time(),
            }

        manager.add_finish_hook(session_id, notify_hook)

    return {
        "session_id": session_id,
        "hook_added": True,
        "hook_type": hook_type,
    }


@tool()
def notify_parent(
    message: str,
    status: str = "progress",
    data: str = "",
) -> dict:
    """Notify the parent agent that spawned this session.

    Sub-agents running in background sessions can call this tool to send
    status updates or results back to the primary agent.

    Args:
        message: Message to send to parent agent.
        status: Status: 'progress', 'completed', 'failed'.
        data: JSON string of additional data.
    """
    if not message:
        return {"error": "Parameter 'message' is required"}

    try:
        parsed_data = json.loads(data) if data else {}
    except json.JSONDecodeError:
        parsed_data = {"raw": data}

    notification = {
        "message": message,
        "status": status,
        "data": parsed_data,
        "timestamp": time.time(),
    }

    # Store the notification for the parent to pick up.
    # The parent can check via get_session_status which includes notifications.
    return {"notification": notification, "status": status}


@tool()
def spawn_agent(
    agent_id: str,
    task: str,
    name: str = "",
    system_prompt: str = "",
    allowed_agents: str = "",
    parent_session_id: str = "",
) -> dict:
    """Spawn a new agent session in a single step.

    Creates and immediately starts a new agent session with
    the given task. This is the preferred tool for simple
    agent spawning. Use create_agent_session and
    start_agent_session separately only when you need to
    pre-configure a session before starting it.

    Args:
        agent_id: Which agent to spawn (must be in allowed list if configured).
        task: Task description/instructions for the agent.
        name: Optional human-readable session name.
        system_prompt: Optional system prompt override for the agent.
        allowed_agents: Comma-separated list of agent IDs this spawned agent can itself spawn.
        parent_session_id: Session ID of the parent that is spawning this agent.
    """
    if not agent_id:
        return {"error": "Parameter 'agent_id' is required"}
    if not task:
        return {"error": "Parameter 'task' is required"}

    manager = _get_manager()

    # Build metadata
    meta: dict[str, Any] = {"agent_id": agent_id}
    if allowed_agents:
        child_allowed = [
            a.strip() for a in allowed_agents.split(",") if a.strip()
        ]
        meta["allowed_agents"] = child_allowed

    # Create and start in one go
    try:
        session = manager.create_session(
            name=name or agent_id,
            metadata=meta,
            agent_id=agent_id,
        )
    except ValueError as exc:
        return {"error": str(exc)}

    # Store parent session context for webhook notifications
    if parent_session_id:
        session.metadata["parent_session_id"] = parent_session_id

    def task_fn(s: AgentSession) -> Any:
        if system_prompt:
            s.messages.append(
                AgentMessage(role="system", content=system_prompt)
            )
        s.messages.append(AgentMessage(role="user", content=task))
        # Framework hook point: a real LLM loop would process here.
        return {"task": task, "message_count": len(s.messages)}

    started = manager.start_session(session.id, task_fn)
    if not started:
        return {"error": f"Could not start session '{session.id}'"}

    return {"session_id": session.id, "status": "running"}


# Backward-compatible aliases
CreateSessionTool = create_agent_session
StartSessionTool = start_agent_session
InjectMessageTool = inject_agent_message
GetSessionStatusTool = get_agent_session_status
ListSessionsTool = list_agent_sessions
WaitSessionTool = wait_agent_session
CollectResultsTool = collect_agent_results
CancelSessionTool = cancel_agent_session
AddFinishHookTool = add_agent_finish_hook
NotifyParentTool = notify_parent
SpawnAgentTool = spawn_agent
