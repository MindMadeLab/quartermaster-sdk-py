"""
Agent session management for parallel agent execution.

Provides a SessionManager that tracks concurrent agent sessions,
supports message injection, finish hooks, and result collection.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class SessionStatus(Enum):
    CREATED = "created"
    RUNNING = "running"
    WAITING = "waiting"  # waiting for input
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentMessage:
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentSession:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    status: SessionStatus = SessionStatus.CREATED
    messages: list[AgentMessage] = field(default_factory=list)
    result: Any = None
    error: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    _on_finish: list[Callable] = field(default_factory=list, repr=False)
    _task_fn: Callable | None = field(default=None, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)


class SessionManager:
    """Manages multiple concurrent agent sessions.

    Singleton-pattern manager that tracks all active sessions,
    supports hooks, and enables parallel agent execution.
    """

    _sessions: dict[str, AgentSession]

    def __init__(self) -> None:
        self._sessions = {}
        self._lock = threading.Lock()
        self._allowed_agents: set[str] = set()

    def set_allowed_agents(self, agent_ids: list[str]) -> None:
        """Set the list of allowed agent IDs. Empty set means allow all."""
        self._allowed_agents = set(agent_ids)

    def is_agent_allowed(self, agent_id: str) -> bool:
        """Check if an agent ID is allowed. Empty set means allow all."""
        if not self._allowed_agents:
            return True
        return agent_id in self._allowed_agents

    def create_session(
        self,
        name: str = "",
        metadata: dict[str, Any] | None = None,
        agent_id: str = "",
    ) -> AgentSession:
        """Create a new agent session.

        If *agent_id* is provided it is validated against the allowed-agents
        list (when one has been configured).  A ``ValueError`` is raised for
        disallowed agent IDs.
        """
        if agent_id and not self.is_agent_allowed(agent_id):
            raise ValueError(
                f"Agent '{agent_id}' is not in the allowed agents list"
            )
        meta = metadata or {}
        if agent_id:
            meta["agent_id"] = agent_id
        session = AgentSession(name=name, metadata=meta)
        with self._lock:
            self._sessions[session.id] = session
        return session

    def get_session(self, session_id: str) -> AgentSession | None:
        return self._sessions.get(session_id)

    def list_sessions(
        self, status: SessionStatus | None = None
    ) -> list[AgentSession]:
        sessions = list(self._sessions.values())
        if status:
            sessions = [s for s in sessions if s.status == status]
        return sessions

    def inject_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Inject a message into a running session."""
        session = self._sessions.get(session_id)
        if not session:
            return False
        session.messages.append(
            AgentMessage(role=role, content=content, metadata=metadata or {})
        )
        session.updated_at = time.time()
        return True

    def start_session(
        self, session_id: str, task_fn: Callable[[AgentSession], Any]
    ) -> bool:
        """Start a session with a task function that runs in a thread."""
        session = self._sessions.get(session_id)
        if not session or session.status == SessionStatus.RUNNING:
            return False
        session._task_fn = task_fn
        session.status = SessionStatus.RUNNING
        session.updated_at = time.time()

        def _run() -> None:
            try:
                result = task_fn(session)
                session.result = result
                session.status = SessionStatus.COMPLETED
            except Exception as e:
                session.error = str(e)
                session.status = SessionStatus.FAILED
            finally:
                session.updated_at = time.time()
                for hook in session._on_finish:
                    try:
                        hook(session)
                    except Exception:
                        pass

        thread = threading.Thread(target=_run, daemon=True)
        session._thread = thread
        thread.start()
        return True

    def add_finish_hook(
        self, session_id: str, hook: Callable[[AgentSession], None]
    ) -> bool:
        """Add a callback that fires when session completes."""
        session = self._sessions.get(session_id)
        if not session:
            return False
        session._on_finish.append(hook)
        return True

    def cancel_session(self, session_id: str) -> bool:
        session = self._sessions.get(session_id)
        if not session:
            return False
        session.status = SessionStatus.CANCELLED
        session.updated_at = time.time()
        return True

    def wait_for_session(
        self, session_id: str, timeout: float | None = None
    ) -> AgentSession | None:
        session = self._sessions.get(session_id)
        if not session or not session._thread:
            return session
        session._thread.join(timeout=timeout)
        return session

    def wait_all(
        self, session_ids: list[str], timeout: float | None = None
    ) -> list[AgentSession]:
        """Wait for multiple sessions to complete."""
        results = []
        for sid in session_ids:
            session = self.wait_for_session(sid, timeout=timeout)
            if session:
                results.append(session)
        return results

    def clear_completed(self) -> int:
        """Remove completed/failed/cancelled sessions."""
        to_remove = [
            sid
            for sid, s in self._sessions.items()
            if s.status
            in (
                SessionStatus.COMPLETED,
                SessionStatus.FAILED,
                SessionStatus.CANCELLED,
            )
        ]
        for sid in to_remove:
            del self._sessions[sid]
        return len(to_remove)


# Module-level default manager
_default_manager = SessionManager()


def get_default_manager() -> SessionManager:
    return _default_manager
