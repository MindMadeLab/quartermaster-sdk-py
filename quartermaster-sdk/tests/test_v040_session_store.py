"""Tests for v0.4.0 SessionStore protocol and InMemorySessionStore.

The session store protocol lets integrators plug their own persistence
backend (Django ORM, Redis, DynamoDB, etc.) while the in-memory
implementation covers tests, demos, and short-lived processes.
"""

from __future__ import annotations

import pytest

from quartermaster_sdk._session import ChatTurn, InMemorySessionStore, SessionStore


# ── 1. Empty session returns empty list ─────────────────────────────


def test_in_memory_store_load_returns_empty_for_new_session() -> None:
    """Loading a session ID that has never been written to must return
    an empty list — not ``None``, not a ``KeyError``.
    """
    store = InMemorySessionStore()
    result = store.load("brand_new_session")

    assert result == []
    assert isinstance(result, list)


# ── 2. Append + load roundtrip ──────────────────────────────────────


def test_in_memory_store_append_and_load_roundtrip() -> None:
    """Appending a turn and loading it back must return the same data."""
    store = InMemorySessionStore()
    turn = ChatTurn(role="user", content="Hello!")

    store.append("sess_1", turn)
    loaded = store.load("sess_1")

    assert len(loaded) == 1
    assert loaded[0].role == "user"
    assert loaded[0].content == "Hello!"


# ── 3. Session isolation ────────────────────────────────────────────


def test_in_memory_store_multiple_sessions_isolated() -> None:
    """Turns appended to session A must not appear in session B."""
    store = InMemorySessionStore()
    store.append("session_a", ChatTurn(role="user", content="A msg"))
    store.append("session_b", ChatTurn(role="user", content="B msg"))

    a_turns = store.load("session_a")
    b_turns = store.load("session_b")

    assert len(a_turns) == 1
    assert a_turns[0].content == "A msg"
    assert len(b_turns) == 1
    assert b_turns[0].content == "B msg"


# ── 4. Protocol enforcement ─────────────────────────────────────────


def test_session_store_protocol_enforced() -> None:
    """A class that does not implement the required abstract methods
    ``load`` and ``append`` must fail with ``TypeError`` on
    instantiation — the ABC contract is strict.
    """

    class BrokenStore(SessionStore):  # type: ignore[abstract]
        pass

    with pytest.raises(TypeError):
        BrokenStore()  # type: ignore[abstract]


# ── 5. Top-level imports ────────────────────────────────────────────


def test_importable_from_top_level() -> None:
    """``from quartermaster_sdk import SessionStore, InMemorySessionStore``
    must resolve and be the same classes as the private-module originals.
    """
    from quartermaster_sdk import InMemorySessionStore as TopISS
    from quartermaster_sdk import SessionStore as TopSS

    assert TopSS is SessionStore
    assert TopISS is InMemorySessionStore
