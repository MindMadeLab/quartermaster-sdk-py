"""
Agent session management tools.

Provides tools for creating, starting, monitoring, and collecting
results from parallel agent sessions.
"""

from __future__ import annotations

import json
import time
from typing import Any

from quartermaster_tools.base import AbstractTool
from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolResult

from quartermaster_tools.builtin.agents.session import (
    AgentMessage,
    AgentSession,
    SessionManager,
    SessionStatus,
    get_default_manager,
)


class _SessionManagerMixin:
    """Mixin providing access to a shared SessionManager."""

    _manager: SessionManager | None = None

    def __init__(self, manager: SessionManager | None = None) -> None:
        if manager is not None:
            self._manager = manager

    def _get_manager(self) -> SessionManager:
        if self._manager is not None:
            return self._manager
        return get_default_manager()


class CreateSessionTool(_SessionManagerMixin, AbstractTool):
    """Create a new agent session.

    For simple use-cases prefer ``SpawnAgentTool`` (``spawn_agent``) which
    combines session creation and start into a single call.  This tool is
    useful for advanced patterns where you need to pre-configure a session
    (e.g. inject messages or add hooks) before starting it.
    """

    def name(self) -> str:
        return "create_agent_session"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="name",
                description="Optional human-readable name for the session.",
                type="string",
                required=False,
                default="",
            ),
            ToolParameter(
                name="metadata",
                description="Optional JSON string of metadata key-value pairs.",
                type="string",
                required=False,
                default="",
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Create a new parallel agent session.",
            long_description=(
                "Creates a new agent session that can be started with a task. "
                "Sessions run in separate threads for parallel execution."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        name = kwargs.get("name", "")
        metadata_str = kwargs.get("metadata", "")
        metadata: dict[str, Any] = {}
        if metadata_str:
            try:
                metadata = json.loads(metadata_str)
            except (json.JSONDecodeError, TypeError):
                return ToolResult(
                    success=False, error="Invalid JSON in 'metadata' parameter"
                )

        manager = self._get_manager()
        session = manager.create_session(name=name, metadata=metadata)
        return ToolResult(
            success=True,
            data={
                "session_id": session.id,
                "name": session.name,
                "status": session.status.value,
            },
        )


class StartSessionTool(_SessionManagerMixin, AbstractTool):
    """Start a session with a task.

    For simple use-cases prefer ``SpawnAgentTool`` (``spawn_agent``) which
    combines session creation and start into a single call.  This tool is
    useful for advanced patterns where you need to pre-configure a session
    (e.g. inject messages or add hooks) before starting it.
    """

    def name(self) -> str:
        return "start_agent_session"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="session_id",
                description="The session ID to start.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="task",
                description="Task description/instructions for the agent.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="system_prompt",
                description="Optional system prompt for the agent.",
                type="string",
                required=False,
                default="",
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Start an agent session with a task.",
            long_description=(
                "Starts a previously created session with the given task. "
                "The task runs in a background thread. Use wait or status "
                "tools to monitor progress."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        session_id = kwargs.get("session_id", "")
        task = kwargs.get("task", "")
        system_prompt = kwargs.get("system_prompt", "")

        if not session_id:
            return ToolResult(
                success=False, error="Parameter 'session_id' is required"
            )
        if not task:
            return ToolResult(success=False, error="Parameter 'task' is required")

        manager = self._get_manager()
        session = manager.get_session(session_id)
        if not session:
            return ToolResult(
                success=False, error=f"Session '{session_id}' not found"
            )

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
            return ToolResult(
                success=False,
                error=f"Could not start session '{session_id}' (already running or not found)",
            )

        return ToolResult(
            success=True,
            data={"session_id": session_id, "status": "running"},
        )


class InjectMessageTool(_SessionManagerMixin, AbstractTool):
    """Inject a message into a running session."""

    def name(self) -> str:
        return "inject_agent_message"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="session_id",
                description="The session ID to inject into.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="content",
                description="Message content to inject.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="role",
                description="Message role (user, assistant, system).",
                type="string",
                required=False,
                default="user",
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Inject a message into a running agent session.",
            long_description=(
                "Adds a message to the session's message history. "
                "Can be used to provide additional context or instructions "
                "to a running agent."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        session_id = kwargs.get("session_id", "")
        content = kwargs.get("content", "")
        role = kwargs.get("role", "user")

        if not session_id:
            return ToolResult(
                success=False, error="Parameter 'session_id' is required"
            )
        if not content:
            return ToolResult(
                success=False, error="Parameter 'content' is required"
            )

        manager = self._get_manager()
        injected = manager.inject_message(session_id, role=role, content=content)
        if not injected:
            return ToolResult(
                success=False, error=f"Session '{session_id}' not found"
            )

        session = manager.get_session(session_id)
        return ToolResult(
            success=True,
            data={
                "session_id": session_id,
                "injected": True,
                "message_count": len(session.messages) if session else 0,
            },
        )


class GetSessionStatusTool(_SessionManagerMixin, AbstractTool):
    """Get the status of an agent session."""

    def name(self) -> str:
        return "get_agent_session_status"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="session_id",
                description="The session ID to check.",
                type="string",
                required=True,
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Get the status of an agent session.",
            long_description=(
                "Returns detailed status information for a session, "
                "including its current state, message count, and timestamps."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        session_id = kwargs.get("session_id", "")
        if not session_id:
            return ToolResult(
                success=False, error="Parameter 'session_id' is required"
            )

        manager = self._get_manager()
        session = manager.get_session(session_id)
        if not session:
            return ToolResult(
                success=False, error=f"Session '{session_id}' not found"
            )

        return ToolResult(
            success=True,
            data={
                "session_id": session.id,
                "status": session.status.value,
                "name": session.name,
                "message_count": len(session.messages),
                "created_at": session.created_at,
                "updated_at": session.updated_at,
            },
        )


class ListSessionsTool(_SessionManagerMixin, AbstractTool):
    """List all agent sessions."""

    def name(self) -> str:
        return "list_agent_sessions"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="status",
                description="Optional status filter (created, running, completed, failed, cancelled).",
                type="string",
                required=False,
                default="",
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="List all agent sessions.",
            long_description=(
                "Returns a list of all tracked agent sessions, "
                "optionally filtered by status."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        status_str = kwargs.get("status", "")
        manager = self._get_manager()

        status_filter: SessionStatus | None = None
        if status_str:
            try:
                status_filter = SessionStatus(status_str)
            except ValueError:
                return ToolResult(
                    success=False,
                    error=f"Invalid status '{status_str}'. Valid values: "
                    + ", ".join(s.value for s in SessionStatus),
                )

        sessions = manager.list_sessions(status=status_filter)
        return ToolResult(
            success=True,
            data={
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
            },
        )


class WaitSessionTool(_SessionManagerMixin, AbstractTool):
    """Wait for a session to complete."""

    def name(self) -> str:
        return "wait_agent_session"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="session_id",
                description="The session ID to wait for.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="timeout",
                description="Maximum seconds to wait (default: 30).",
                type="number",
                required=False,
                default=30,
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Wait for an agent session to complete.",
            long_description=(
                "Blocks until the session finishes or the timeout expires. "
                "Returns the session result and final status."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        session_id = kwargs.get("session_id", "")
        timeout = float(kwargs.get("timeout", 30))

        if not session_id:
            return ToolResult(
                success=False, error="Parameter 'session_id' is required"
            )

        manager = self._get_manager()
        session = manager.wait_for_session(session_id, timeout=timeout)
        if not session:
            return ToolResult(
                success=False, error=f"Session '{session_id}' not found"
            )

        return ToolResult(
            success=True,
            data={
                "session_id": session.id,
                "status": session.status.value,
                "result": session.result,
                "error": session.error,
            },
        )


class CollectResultsTool(_SessionManagerMixin, AbstractTool):
    """Collect results from multiple sessions."""

    def name(self) -> str:
        return "collect_agent_results"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="session_ids",
                description="Comma-separated list of session IDs.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="timeout",
                description="Maximum seconds to wait per session (default: 30).",
                type="number",
                required=False,
                default=30,
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Collect results from multiple agent sessions.",
            long_description=(
                "Waits for all specified sessions to complete and "
                "collects their results. Returns a summary of all sessions."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        session_ids_str = kwargs.get("session_ids", "")
        timeout = float(kwargs.get("timeout", 30))

        if not session_ids_str:
            return ToolResult(
                success=False, error="Parameter 'session_ids' is required"
            )

        session_ids = [s.strip() for s in session_ids_str.split(",") if s.strip()]
        manager = self._get_manager()
        sessions = manager.wait_all(session_ids, timeout=timeout)

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

        return ToolResult(
            success=True,
            data={
                "results": results,
                "all_completed": all_completed,
            },
        )


class CancelSessionTool(_SessionManagerMixin, AbstractTool):
    """Cancel a running session."""

    def name(self) -> str:
        return "cancel_agent_session"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="session_id",
                description="The session ID to cancel.",
                type="string",
                required=True,
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Cancel a running agent session.",
            long_description=(
                "Marks a session as cancelled. Note that the underlying "
                "thread may still be running; this sets the status flag."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        session_id = kwargs.get("session_id", "")
        if not session_id:
            return ToolResult(
                success=False, error="Parameter 'session_id' is required"
            )

        manager = self._get_manager()
        cancelled = manager.cancel_session(session_id)
        if not cancelled:
            return ToolResult(
                success=False, error=f"Session '{session_id}' not found"
            )

        return ToolResult(
            success=True,
            data={"session_id": session_id, "cancelled": True},
        )


class AddFinishHookTool(_SessionManagerMixin, AbstractTool):
    """Add a finish hook to a session."""

    def name(self) -> str:
        return "add_agent_finish_hook"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="session_id",
                description="The session ID to add a hook to.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="hook_type",
                description="Type of hook: 'log' or 'notify'.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="hook_config",
                description="Optional JSON string of hook configuration.",
                type="string",
                required=False,
                default="",
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Add a finish hook to an agent session.",
            long_description=(
                "Registers a callback that fires when the session completes. "
                "Built-in hooks: 'log' writes to a file, 'notify' stores a "
                "notification in session metadata."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        session_id = kwargs.get("session_id", "")
        hook_type = kwargs.get("hook_type", "")
        hook_config_str = kwargs.get("hook_config", "")

        if not session_id:
            return ToolResult(
                success=False, error="Parameter 'session_id' is required"
            )
        if not hook_type:
            return ToolResult(
                success=False, error="Parameter 'hook_type' is required"
            )
        if hook_type not in ("log", "notify"):
            return ToolResult(
                success=False,
                error=f"Invalid hook_type '{hook_type}'. Must be 'log' or 'notify'.",
            )

        hook_config: dict[str, Any] = {}
        if hook_config_str:
            try:
                hook_config = json.loads(hook_config_str)
            except (json.JSONDecodeError, TypeError):
                return ToolResult(
                    success=False, error="Invalid JSON in 'hook_config' parameter"
                )

        manager = self._get_manager()
        session = manager.get_session(session_id)
        if not session:
            return ToolResult(
                success=False, error=f"Session '{session_id}' not found"
            )

        if hook_type == "log":
            path = hook_config.get("path", "agent_session.log")

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

        return ToolResult(
            success=True,
            data={
                "session_id": session_id,
                "hook_added": True,
                "hook_type": hook_type,
            },
        )


class NotifyParentTool(_SessionManagerMixin, AbstractTool):
    """Notify the parent agent that spawned this session.

    Sub-agents running in background sessions can call this tool to send
    status updates or results back to the primary agent via webhook.
    """

    def name(self) -> str:
        return "notify_parent"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="message",
                description="Message to send to parent agent",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="status",
                description="Status: 'progress', 'completed', 'failed'",
                type="string",
                required=False,
                default="progress",
            ),
            ToolParameter(
                name="data",
                description="JSON string of additional data",
                type="string",
                required=False,
                default="",
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Notify the parent agent that spawned this session",
            long_description=(
                "Send status updates or results from a sub-session back to "
                "the parent agent."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        message = kwargs.get("message", "")
        status = kwargs.get("status", "progress")
        data_str = kwargs.get("data", "{}")

        if not message:
            return ToolResult(success=False, error="message is required")

        try:
            data = json.loads(data_str) if data_str else {}
        except json.JSONDecodeError:
            data = {"raw": data_str}

        notification = {
            "message": message,
            "status": status,
            "data": data,
            "timestamp": time.time(),
        }

        # Store the notification for the parent to pick up.
        # The parent can check via get_session_status which includes notifications.
        return ToolResult(
            success=True,
            data={"notification": notification, "status": status},
        )


class SpawnAgentTool(_SessionManagerMixin, AbstractTool):
    """Spawn a new agent session in a single step.

    Combines session creation and start into one call.  This is the
    preferred tool for simple agent spawning.  Use ``create_agent_session``
    and ``start_agent_session`` separately only when you need to
    pre-configure a session before starting it.
    """

    _allowed_agents: set[str]

    def __init__(
        self,
        manager: SessionManager | None = None,
        allowed_agents: list[str] | None = None,
    ) -> None:
        super().__init__(manager=manager)
        self._allowed_agents = set(allowed_agents) if allowed_agents else set()

    def name(self) -> str:
        return "spawn_agent"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="agent_id",
                description="Which agent to spawn (must be in allowed list).",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="task",
                description="Task description/instructions for the agent.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="name",
                description="Optional human-readable session name.",
                type="string",
                required=False,
                default="",
            ),
            ToolParameter(
                name="system_prompt",
                description="Optional system prompt override for the agent.",
                type="string",
                required=False,
                default="",
            ),
            ToolParameter(
                name="allowed_agents",
                description=(
                    "Comma-separated list of agent IDs this spawned agent "
                    "can itself spawn (for recursive control)."
                ),
                type="string",
                required=False,
                default="",
            ),
            ToolParameter(
                name="parent_session_id",
                description=(
                    "Session ID of the parent that is spawning this agent. "
                    "Stored in metadata so the child can send notifications back."
                ),
                type="string",
                required=False,
                default="",
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Spawn a new agent session in a single step.",
            long_description=(
                "Creates and immediately starts a new agent session with "
                "the given task.  The agent_id must be in the allowed "
                "agents list (if one is configured).  Returns the "
                "session_id and a status of 'running'."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        agent_id = kwargs.get("agent_id", "")
        task = kwargs.get("task", "")
        session_name = kwargs.get("name", "")
        system_prompt = kwargs.get("system_prompt", "")
        allowed_agents_str = kwargs.get("allowed_agents", "")
        parent_session_id = kwargs.get("parent_session_id", "")

        if not agent_id:
            return ToolResult(
                success=False, error="Parameter 'agent_id' is required"
            )
        if not task:
            return ToolResult(success=False, error="Parameter 'task' is required")

        # Validate agent_id against allowed list (empty = allow all)
        if self._allowed_agents and agent_id not in self._allowed_agents:
            return ToolResult(
                success=False,
                error=(
                    f"Agent '{agent_id}' is not in the allowed agents list. "
                    f"Allowed: {', '.join(sorted(self._allowed_agents))}"
                ),
            )

        manager = self._get_manager()

        # Build metadata
        metadata: dict[str, Any] = {"agent_id": agent_id}
        if allowed_agents_str:
            child_allowed = [
                a.strip() for a in allowed_agents_str.split(",") if a.strip()
            ]
            metadata["allowed_agents"] = child_allowed

        # Create and start in one go
        try:
            session = manager.create_session(
                name=session_name or agent_id,
                metadata=metadata,
                agent_id=agent_id,
            )
        except ValueError as exc:
            return ToolResult(success=False, error=str(exc))

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
            return ToolResult(
                success=False,
                error=f"Could not start session '{session.id}'",
            )

        return ToolResult(
            success=True,
            data={"session_id": session.id, "status": "running"},
        )
