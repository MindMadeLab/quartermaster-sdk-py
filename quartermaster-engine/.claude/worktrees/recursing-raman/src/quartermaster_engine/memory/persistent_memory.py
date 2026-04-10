"""Persistent memory — cross-flow memory that survives between executions.

Used for agent-level memory: facts the agent learns, user preferences, etc.
The actual storage backend is pluggable via the PersistentMemory protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID


@dataclass
class MemoryEntry:
    """A single entry in persistent memory."""

    key: str
    value: str
    agent_id: UUID
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, str] = field(default_factory=dict)


class PersistentMemory(Protocol):
    """Protocol for cross-flow persistent memory.

    Implementations might use SQLite, PostgreSQL + pgvector, Redis, etc.
    """

    def read(self, agent_id: UUID, key: str) -> str | None:
        """Read a memory entry by key."""
        ...

    def write(self, agent_id: UUID, key: str, value: str) -> None:
        """Write a new memory entry (or overwrite if key exists)."""
        ...

    def update(self, agent_id: UUID, key: str, value: str) -> None:
        """Update an existing memory entry."""
        ...

    def delete(self, agent_id: UUID, key: str) -> None:
        """Delete a memory entry."""
        ...

    def search(self, agent_id: UUID, query: str, limit: int = 10) -> list[MemoryEntry]:
        """Search memory entries (semantic or keyword-based)."""
        ...

    def list_keys(self, agent_id: UUID) -> list[str]:
        """List all memory keys for an agent."""
        ...


class InMemoryPersistence:
    """Simple dict-backed persistent memory. Good for testing."""

    def __init__(self) -> None:
        self._store: dict[UUID, dict[str, MemoryEntry]] = {}

    def read(self, agent_id: UUID, key: str) -> str | None:
        entries = self._store.get(agent_id, {})
        entry = entries.get(key)
        return entry.value if entry else None

    def write(self, agent_id: UUID, key: str, value: str) -> None:
        if agent_id not in self._store:
            self._store[agent_id] = {}
        now = datetime.now(UTC)
        self._store[agent_id][key] = MemoryEntry(
            key=key, value=value, agent_id=agent_id, created_at=now, updated_at=now
        )

    def update(self, agent_id: UUID, key: str, value: str) -> None:
        entries = self._store.get(agent_id, {})
        if key in entries:
            entries[key].value = value
            entries[key].updated_at = datetime.now(UTC)
        else:
            self.write(agent_id, key, value)

    def delete(self, agent_id: UUID, key: str) -> None:
        entries = self._store.get(agent_id, {})
        entries.pop(key, None)

    def search(self, agent_id: UUID, query: str, limit: int = 10) -> list[MemoryEntry]:
        """Simple substring search (no semantic search in memory impl)."""
        entries = self._store.get(agent_id, {})
        query_lower = query.lower()
        results = [
            entry
            for entry in entries.values()
            if query_lower in entry.key.lower() or query_lower in entry.value.lower()
        ]
        return results[:limit]

    def list_keys(self, agent_id: UUID) -> list[str]:
        return list(self._store.get(agent_id, {}).keys())
