"""Synchronous dispatcher — executes nodes immediately in the current thread.

Simple, predictable, and great for testing. No true parallelism.
"""

from __future__ import annotations

from collections.abc import Callable
from uuid import UUID


class SyncDispatcher:
    """Execute nodes synchronously in the calling thread.

    This is the simplest dispatcher. Successor nodes are executed one at a time
    in the order they appear. No parallelism, but no concurrency issues either.
    """

    def dispatch(
        self,
        flow_id: UUID,
        node_id: UUID,
        execute_fn: Callable[[UUID, UUID], None],
    ) -> None:
        execute_fn(flow_id, node_id)

    def wait_all(self) -> None:
        pass  # Nothing to wait for — everything is already done

    def shutdown(self) -> None:
        pass  # No resources to clean up
