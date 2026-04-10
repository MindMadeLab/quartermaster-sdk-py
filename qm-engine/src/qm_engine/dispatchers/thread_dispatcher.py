"""Thread-based dispatcher — executes nodes in parallel using a thread pool.

Provides true parallelism for branches. Suitable for I/O-bound node execution
(LLM API calls, tool invocations) which is the common case for agent graphs.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from uuid import UUID


class ThreadDispatcher:
    """Execute nodes in parallel using a thread pool.

    Parallel branches are submitted to a ThreadPoolExecutor.
    `wait_all()` blocks until all pending tasks complete.
    """

    def __init__(self, max_workers: int = 4) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: list[Future[None]] = []

    def dispatch(
        self,
        flow_id: UUID,
        node_id: UUID,
        execute_fn: Callable[[UUID, UUID], None],
    ) -> None:
        future = self._pool.submit(execute_fn, flow_id, node_id)
        self._futures.append(future)

    def wait_all(self) -> None:
        """Block until all dispatched tasks complete, then collect exceptions."""
        exceptions: list[Exception] = []
        for future in self._futures:
            try:
                future.result()
            except Exception as e:
                exceptions.append(e)
        self._futures.clear()
        if exceptions:
            raise ExceptionGroup("Errors in parallel branches", exceptions)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)
        self._futures.clear()
