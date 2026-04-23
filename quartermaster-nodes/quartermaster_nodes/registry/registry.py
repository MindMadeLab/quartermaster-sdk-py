"""Node catalog with auto-discovery and catalog generation.

IMPORTANT — naming gotcha:
=========================

This module exports ``NodeCatalog`` (the design-time catalog for
backward compatibility). It is a *design-time catalog* of node **class
definitions** — not a runtime executor registry.

The ``FlowRunner`` in ``quartermaster_engine`` expects a **different**
registry that maps node types to ``NodeExecutor`` instances. Use
``quartermaster_engine.SimpleNodeRegistry`` (or anything implementing
the ``quartermaster_engine.nodes.NodeRegistry`` Protocol) for that.

Passing a ``NodeCatalog`` / ``quartermaster_nodes.NodeRegistry`` to
``FlowRunner(node_registry=...)`` will raise a ``TypeError`` at the
first node dispatch, pointing here.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, NoReturn, Optional, Type

from quartermaster_nodes.base import AbstractAssistantNode
from quartermaster_nodes.exceptions import NodeNotFoundError

logger = logging.getLogger(__name__)


class NodeCatalog:
    """Design-time catalog of node **class definitions**.

    Tracks which :class:`AbstractAssistantNode` subclasses are
    available to the framework — their names, versions, metadata
    schemas, and flow config. Supports auto-discovery from a package.

    .. note::

        This is NOT the registry ``FlowRunner`` uses at runtime.
        ``FlowRunner`` expects a mapping from node-type strings to
        ``NodeExecutor`` instances — use
        ``quartermaster_engine.SimpleNodeRegistry`` for that. See
        :meth:`get_executor` for the full error message if you've
        wired the wrong one.

    Example::

        catalog = NodeCatalog()
        catalog.register(InstructionNodeV1)
        node_cls = catalog.get("Instruction1")

        # Or use auto-discovery
        catalog.discover("quartermaster_nodes.nodes")
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, Type[AbstractAssistantNode]] = {}
        self._by_version: Dict[str, Dict[str, Type[AbstractAssistantNode]]] = {}

    def register(self, node_class: Type[AbstractAssistantNode]) -> None:
        """Register a node class."""
        name = node_class.name()
        version = node_class.version()

        key = f"{name}:{version}"
        self._nodes[key] = node_class
        self._nodes[name] = node_class  # Latest version wins

        if name not in self._by_version:
            self._by_version[name] = {}
        self._by_version[name][version] = node_class

        logger.debug("Registered node %s v%s", name, version)

    def get(
        self,
        name: str,
        version: Optional[str] = None,
    ) -> Type[AbstractAssistantNode]:
        """Look up a node by name and optional version."""
        if version:
            key = f"{name}:{version}"
            if key in self._nodes:
                return self._nodes[key]
            raise NodeNotFoundError(name, version)

        if name in self._nodes:
            return self._nodes[name]
        raise NodeNotFoundError(name)

    def has(self, name: str, version: Optional[str] = None) -> bool:
        """Check if a node is registered."""
        try:
            self.get(name, version)
            return True
        except NodeNotFoundError:
            return False

    def list_nodes(self) -> List[Dict[str, Any]]:
        """List all registered nodes with their metadata."""
        seen = set()
        result = []
        for key, node_cls in self._nodes.items():
            if ":" not in key:
                continue  # Skip non-versioned entries
            if key in seen:
                continue
            seen.add(key)

            info = node_cls.info()
            config = node_cls.flow_config()
            result.append(
                {
                    "name": node_cls.name(),
                    "version": node_cls.version(),
                    "description": info.description,
                    "deprecated": node_cls.deprecated(),
                    "flow_config": config.asdict(),
                    "metadata_schema": info.metadata,
                }
            )
        return sorted(result, key=lambda x: x["name"])

    def catalog_json(self) -> List[Dict[str, Any]]:
        """Generate a JSON-serializable catalog of all registered nodes."""
        return self.list_nodes()

    def discover(self, package: str) -> int:
        """Auto-discover and register nodes from a package.

        Imports all submodules of the given package and registers any
        AbstractAssistantNode subclasses found.

        Returns the number of nodes registered.
        """
        import importlib
        import pkgutil

        count = 0
        try:
            pkg = importlib.import_module(package)
        except ImportError:
            logger.warning("Could not import package %s", package)
            return 0

        for importer, modname, ispkg in pkgutil.walk_packages(pkg.__path__, prefix=package + "."):
            try:
                module = importlib.import_module(modname)
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, AbstractAssistantNode)
                        and attr is not AbstractAssistantNode
                        and not getattr(attr, "__abstractmethods__", None)
                    ):
                        self.register(attr)
                        count += 1
            except Exception as e:
                logger.warning("Error importing %s: %s", modname, e)

        return count

    @property
    def count(self) -> int:
        """Number of unique registered nodes (name:version pairs)."""
        return sum(1 for k in self._nodes if ":" in k)

    # ------------------------------------------------------------------
    # Wrong-registry guard
    # ------------------------------------------------------------------
    def get_executor(self, node_type: str) -> NoReturn:
        """Guard — this is NOT an executor registry.

        ``FlowRunner._execute_logic_node`` calls ``get_executor`` on the
        registry it was handed. If the hand-off was wrong (passing this
        catalog instead of ``quartermaster_engine.SimpleNodeRegistry``),
        we'd crash with an unhelpful ``AttributeError``. Provide a clear
        redirect instead.
        """
        raise TypeError(
            "quartermaster_nodes.NodeCatalog (aka NodeRegistry) is a "
            "design-time catalog of node CLASS definitions, not a runtime "
            "executor registry. FlowRunner expects a registry mapping node "
            "types to NodeExecutor instances.\n\n"
            "Use quartermaster_engine.SimpleNodeRegistry instead:\n\n"
            "    from quartermaster_engine import FlowRunner\n"
            "    from quartermaster_engine.nodes import SimpleNodeRegistry\n\n"
            "    registry = SimpleNodeRegistry()\n"
            "    registry.register('Instruction1', my_instruction_executor)\n"
            "    runner = FlowRunner(graph=graph, node_registry=registry)\n\n"
            f"(attempted: get_executor({node_type!r}))"
        )


# ------------------------------------------------------------------
# Backward-compatibility alias
# ------------------------------------------------------------------
def register_node(registry: NodeCatalog):
    """Decorator for registering a node class with a catalog.

    Example::

        catalog = NodeCatalog()

        @register_node(catalog)
        class MyNode(AbstractAssistantNode):
            ...
    """

    def decorator(cls: Type[AbstractAssistantNode]) -> Type[AbstractAssistantNode]:
        registry.register(cls)
        return cls

    return decorator


# Global default catalog
default_registry = NodeCatalog()
