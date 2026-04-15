"""Regression tests for the NodeRegistry → NodeCatalog rename.

In 0.1.0, ``quartermaster_nodes.NodeRegistry`` and
``quartermaster_engine.nodes.NodeRegistry`` both existed with different
APIs. ``FlowRunner.get_executor`` crashed with ``AttributeError`` when
a user (reasonably) passed the nodes-package registry to the engine.

0.1.1 addresses this two ways:

1. Rename the design-time catalog to ``NodeCatalog`` (canonical name).
   ``NodeRegistry`` remains as a backward-compat alias.
2. Add a ``get_executor`` guard on ``NodeCatalog`` that raises a
   helpful ``TypeError`` instead of ``AttributeError``, pointing users
   to ``quartermaster_engine.SimpleNodeRegistry``.

These tests lock in both behaviours.
"""

from __future__ import annotations

import pytest

from quartermaster_nodes import NodeCatalog, NodeRegistry
from quartermaster_nodes.registry import (
    NodeCatalog as FromRegistry,
)
from quartermaster_nodes.registry import (
    NodeRegistry as NodeRegistryFromRegistry,
)
from quartermaster_nodes.registry import default_registry
from quartermaster_nodes.registry.registry import NodeCatalog as FromInnerModule


def test_noderegistry_is_alias_for_nodecatalog() -> None:
    """Old ``from quartermaster_nodes import NodeRegistry`` must keep working."""
    assert NodeRegistry is NodeCatalog
    assert NodeRegistryFromRegistry is FromRegistry


def test_same_class_across_import_paths() -> None:
    """All import paths resolve to the same class object."""
    assert NodeCatalog is FromRegistry
    assert NodeCatalog is FromInnerModule


def test_default_registry_is_nodecatalog_instance() -> None:
    """The module-level default is a NodeCatalog — and therefore a NodeRegistry."""
    assert isinstance(default_registry, NodeCatalog)
    assert isinstance(default_registry, NodeRegistry)


class TestGetExecutorGuard:
    """The get_executor method must raise TypeError with a clear redirect.

    This prevents the confusing AttributeError that 0.1.0 users hit when
    they passed quartermaster_nodes.NodeRegistry to FlowRunner.
    """

    def test_raises_typeerror_not_attributeerror(self) -> None:
        catalog = NodeCatalog()
        with pytest.raises(TypeError):
            catalog.get_executor("Instruction1")

    def test_error_message_redirects_to_simple_node_registry(self) -> None:
        catalog = NodeCatalog()
        with pytest.raises(TypeError) as exc_info:
            catalog.get_executor("Instruction1")
        msg = str(exc_info.value)
        # Must mention the correct replacement class
        assert "SimpleNodeRegistry" in msg
        assert "quartermaster_engine" in msg
        # Must include the node type that was attempted
        assert "Instruction1" in msg

    def test_guard_also_fires_on_alias(self) -> None:
        """Calling get_executor on the legacy NodeRegistry alias works the same."""
        registry = NodeRegistry()
        with pytest.raises(TypeError) as exc_info:
            registry.get_executor("User1")
        assert "SimpleNodeRegistry" in str(exc_info.value)

    def test_error_message_contains_working_example(self) -> None:
        """The error message should include a code snippet users can copy."""
        catalog = NodeCatalog()
        with pytest.raises(TypeError) as exc_info:
            catalog.get_executor("X")
        msg = str(exc_info.value)
        assert "from quartermaster_engine import FlowRunner" in msg
        assert "SimpleNodeRegistry" in msg
        assert "registry.register" in msg


class TestCatalogAPIUnchanged:
    """The original catalog API (get, has, register, etc.) must still work."""

    def test_empty_catalog_has_no_nodes(self) -> None:
        catalog = NodeCatalog()
        assert catalog.count == 0
        assert catalog.list_nodes() == []

    def test_has_on_empty_catalog_returns_false(self) -> None:
        catalog = NodeCatalog()
        assert catalog.has("anything") is False
