"""Node registry with auto-discovery and catalog generation."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Type

from qm_nodes.base import AbstractAssistantNode
from qm_nodes.exceptions import NodeNotFoundError

logger = logging.getLogger(__name__)


class NodeRegistry:
    """Registry for discovering and managing node types.

    Provides registration, lookup, and catalog generation for all
    available node types.

    Example:
        registry = NodeRegistry()
        registry.register(InstructionNodeV1)
        node_cls = registry.get("Instruction1")

        # Or use auto-discovery
        registry.discover("qm_nodes.nodes")
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
            result.append({
                "name": node_cls.name(),
                "version": node_cls.version(),
                "description": info.description,
                "deprecated": node_cls.deprecated(),
                "flow_config": config.asdict(),
                "metadata_schema": info.metadata,
            })
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

        for importer, modname, ispkg in pkgutil.walk_packages(
            pkg.__path__, prefix=package + "."
        ):
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


def register_node(registry: NodeRegistry):
    """Decorator for registering a node class with a registry.

    Example:
        registry = NodeRegistry()

        @register_node(registry)
        class MyNode(AbstractAssistantNode):
            ...
    """

    def decorator(cls: Type[AbstractAssistantNode]) -> Type[AbstractAssistantNode]:
        registry.register(cls)
        return cls

    return decorator


# Global default registry
default_registry = NodeRegistry()
