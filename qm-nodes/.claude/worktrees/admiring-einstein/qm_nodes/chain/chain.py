"""Chain class for composing handlers into processing pipelines."""

from typing import Any, Dict, List

from qm_nodes.chain.handler import Handler


class Chain:
    """Composes handlers into a sequential processing pipeline.

    Example:
        chain = Chain()
        chain.add_handler(ValidateHandler())
        chain.add_handler(TransformHandler())
        chain.add_handler(ExecuteHandler())
        result = chain.run({"input": "data"})
    """

    def __init__(self) -> None:
        self.handlers: List[Handler] = []

    def add_handler(self, handler: Handler) -> "Chain":
        """Add a handler to the chain. Returns self for fluent chaining."""
        self.handlers.append(handler)
        return self

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all handlers in sequence, passing data through each."""
        for handler in self.handlers:
            data = handler.handle(data)
        return data
