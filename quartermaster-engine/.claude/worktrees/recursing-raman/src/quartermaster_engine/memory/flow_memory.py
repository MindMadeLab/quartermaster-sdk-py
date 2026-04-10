"""Flow memory — scoped variable storage for a single flow execution.

Provides a simple key-value interface for nodes to share data during a flow.
Backed by the ExecutionStore so persistence strategy is pluggable.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from quartermaster_engine.stores.base import ExecutionStore


class FlowMemory:
    """Key-value store scoped to a single flow execution.

    Used by FlowMemory1, Var1, Text1 nodes to share data between nodes
    within the same flow run.
    """

    def __init__(self, flow_id: UUID, store: ExecutionStore) -> None:
        self._flow_id = flow_id
        self._store = store

    def set(self, key: str, value: Any) -> None:
        """Store a value."""
        self._store.save_memory(self._flow_id, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value, returning default if not found."""
        val = self._store.get_memory(self._flow_id, key)
        return val if val is not None else default

    def delete(self, key: str) -> None:
        """Remove a value."""
        self._store.delete_memory(self._flow_id, key)

    def list_keys(self) -> list[str]:
        """List all stored keys."""
        return list(self._store.get_all_memory(self._flow_id).keys())

    def get_all(self) -> dict[str, Any]:
        """Get all stored key-value pairs."""
        return self._store.get_all_memory(self._flow_id)

    def clear(self) -> None:
        """Remove all stored values."""
        for key in self.list_keys():
            self.delete(key)
