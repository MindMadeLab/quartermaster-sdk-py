"""SessionStore protocol for multi-turn chat history (v0.4.0).

Multi-turn chat history is caller-managed by design, but the boilerplate
to fold history into ``user_input`` is identical across integrators. This
module provides a thin adapter protocol that collapses it::

    class DjangoSessionStore(qm.SessionStore):
        def load(self, session_id: str) -> list[qm.ChatTurn]: ...
        def append(self, session_id: str, turn: qm.ChatTurn) -> None: ...

    result = qm.run(
        graph, user_input,
        session=DjangoSessionStore(),
        session_id=request.user.chat_session_id,
    )
    # SDK handles load + fold + append automatically

Added in v0.4.0.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ChatTurn:
    """A single turn in a conversation.

    Attributes:
        role: ``"user"`` or ``"assistant"``.
        content: The text content of this turn.
    """

    role: str
    content: str


class SessionStore(ABC):
    """Abstract protocol for loading and persisting chat history.

    Integrators subclass this to connect their persistence layer
    (Django ORM, Redis, DynamoDB, etc.) — the SDK calls :meth:`load`
    before the run and :meth:`append` after.
    """

    @abstractmethod
    def load(self, session_id: str) -> list[ChatTurn]:
        """Return all prior turns for *session_id*, oldest first."""
        ...

    @abstractmethod
    def append(self, session_id: str, turn: ChatTurn) -> None:
        """Persist a new turn for *session_id*."""
        ...


class InMemorySessionStore(SessionStore):
    """Built-in in-memory store (dict of lists, no persistence).

    Suitable for tests, demos, and short-lived processes. For
    production use, subclass :class:`SessionStore` with your own
    persistence backend.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, list[ChatTurn]] = {}

    def load(self, session_id: str) -> list[ChatTurn]:
        return list(self._sessions.get(session_id, []))

    def append(self, session_id: str, turn: ChatTurn) -> None:
        self._sessions.setdefault(session_id, []).append(turn)


__all__ = [
    "ChatTurn",
    "SessionStore",
    "InMemorySessionStore",
]
