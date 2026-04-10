"""
Chain-of-Responsibility pattern for composable data processing pipelines.

Handlers form a processing pipeline where each handler receives data,
processes or transforms it, and passes it to the next handler.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Handler(ABC):
    """A single processing step in a chain.

    Subclasses implement handle() to process data and return the
    (possibly modified) result for the next handler in the chain.
    """

    @abstractmethod
    def handle(self, data: dict[str, Any]) -> dict[str, Any]:
        """Process the data and return the result.

        Args:
            data: Input data dictionary.

        Returns:
            Processed data dictionary (may be modified in-place or a new dict).

        Raises:
            Any exception to halt the chain.
        """
        ...


class Chain:
    """A composable chain of handlers that processes data sequentially.

    Handlers are executed in the order they are added. Each handler
    receives the output of the previous handler. If any handler raises
    an exception, the chain halts and the exception propagates.

    Usage:
        chain = Chain().add_handler(h1).add_handler(h2)
        result = chain.run({"key": "value"})
    """

    def __init__(self) -> None:
        self._handlers: list[Handler] = []

    def add_handler(self, handler: Handler) -> Chain:
        """Add a handler to the end of the chain. Returns self for fluent chaining."""
        self._handlers.append(handler)
        return self

    def run(self, data: dict[str, Any]) -> dict[str, Any]:
        """Execute all handlers in sequence.

        Args:
            data: Initial input data.

        Returns:
            The final processed data after all handlers have run.

        Raises:
            Any exception raised by a handler.
        """
        result = data
        for handler in self._handlers:
            result = handler.handle(result)
        return result

    @property
    def handlers(self) -> list[Handler]:
        """Return a copy of the handler list."""
        return list(self._handlers)

    def __len__(self) -> int:
        return len(self._handlers)
