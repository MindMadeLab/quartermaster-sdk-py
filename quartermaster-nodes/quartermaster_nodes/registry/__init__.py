"""Design-time catalog of node class definitions.

Exports:

- ``NodeCatalog`` — the canonical name; a catalog of
  :class:`~quartermaster_nodes.base.AbstractAssistantNode` subclasses.
- ``NodeRegistry`` — backward-compatible alias for ``NodeCatalog``.
  Retained so existing code keeps working; new code should prefer
  ``NodeCatalog`` to avoid confusion with the runtime registry
  ``quartermaster_engine.nodes.NodeRegistry`` (a completely different
  Protocol used by ``FlowRunner``).
"""

from quartermaster_nodes.registry.registry import (
    NodeCatalog,
    NodeRegistry,
    default_registry,
    register_node,
)

__all__ = [
    "NodeCatalog",
    "NodeRegistry",  # deprecated alias — kept for backward compatibility
    "default_registry",
    "register_node",
]
