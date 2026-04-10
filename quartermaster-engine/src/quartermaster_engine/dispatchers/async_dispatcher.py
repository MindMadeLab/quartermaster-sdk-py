"""Async dispatcher — executes nodes in parallel using asyncio.create_task.

Provides concurrent execution for branches using asyncio tasks. Ideal for
I/O-bound node execution in async web applications (FastAPI, aiohttp, etc.).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from uuid import UUID


class AsyncDispatcher:
    """Execute nodes concurrently using asyncio tasks.

    Parallel branches are dispatched as asyncio tasks via ``asyncio.create_task``.
    ``wait_all()`` awaits all pending tasks. Because the underlying
    ``execute_fn`` is synchronous (it comes from FlowRunner), each call is
    wrapped in ``loop.run_in_executor`` so the event loop is never blocked.
    """

    def __init__(self) -> None:
        self._tasks: list[asyncio.Task[None]] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Return the running event loop, or create a new one if needed."""
        if self._loop is None or self._loop.is_closed():
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop

    def dispatch(
        self,
        flow_id: UUID,
        node_id: UUID,
        execute_fn: Callable[[UUID, UUID], None],
    ) -> None:
        """Dispatch a node for execution as an asyncio task.

        The synchronous *execute_fn* is scheduled on the event loop's default
        executor so it does not block the loop.

        Args:
            flow_id: The flow this node belongs to.
            node_id: The node to execute.
            execute_fn: A callable that executes the node — signature (flow_id, node_id).
        """
        loop = self._get_loop()

        async def _run() -> None:
            await loop.run_in_executor(None, execute_fn, flow_id, node_id)

        task = asyncio.ensure_future(_run(), loop=loop)
        self._tasks.append(task)

    def wait_all(self) -> None:
        """Block until all dispatched tasks have completed.

        Gathers all pending asyncio tasks and re-raises any exceptions
        that occurred during branch execution.
        """
        if not self._tasks:
            return

        loop = self._get_loop()

        async def _gather() -> None:
            results = await asyncio.gather(*self._tasks, return_exceptions=True)
            exceptions = [r for r in results if isinstance(r, Exception)]
            self._tasks.clear()
            if exceptions:
                raise ExceptionGroup("Errors in async branches", exceptions)

        if loop.is_running():
            # We are inside an async context — schedule and block via a helper
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, _gather())
                future.result()
        else:
            loop.run_until_complete(_gather())

    def shutdown(self) -> None:
        """Cancel any remaining tasks and clean up resources."""
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()
