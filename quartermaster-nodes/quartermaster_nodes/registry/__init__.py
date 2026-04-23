"""Design-time catalog of node class definitions.

Exports ``NodeCatalog`` — a catalog of
:class:`~quartermaster_nodes.base.AbstractAssistantNode` subclasses.
Not to be confused with the runtime registry
``quartermaster_engine.nodes.NodeRegistry`` (a completely different
Protocol used by ``FlowRunner``).
"""

from quartermaster_nodes.registry.registry import (
    NodeCatalog,
    default_registry,
    register_node,
)

__all__ = [
    "NodeCatalog",
    "default_registry",
    "register_node",
]
