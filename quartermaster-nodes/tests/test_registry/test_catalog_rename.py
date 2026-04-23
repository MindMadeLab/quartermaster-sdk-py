"""Regression tests for the NodeCatalog design-time catalog.

In 0.1.0, ``quartermaster_nodes.NodeRegistry`` and
``quartermaster_engine.nodes.NodeRegistry`` both existed with different
APIs. ``FlowRunner.get_executor`` crashed with ``AttributeError`` when
a user (reasonably) passed the nodes-package registry to the engine.

0.1.1 renamed the design-time catalog to ``NodeCatalog`` and kept
``NodeRegistry`` as a backward-compat alias. **v0.6.0 removes the
alias** — callers must now use ``NodeCatalog`` directly.

These tests lock in:
1. The ``NodeRegistry`` alias is gone from every import path.
2. ``NodeCatalog`` keeps its ``get_executor`` guard that redirects to
   ``quartermaster_engine.SimpleNodeRegistry`` with a helpful message.
3. The rest of the catalog API is unchanged.
"""

from __future__ import annotations

import pytest

from quartermaster_nodes import NodeCatalog
from quartermaster_nodes.registry import NodeCatalog as FromRegistry
from quartermaster_nodes.registry import default_registry
from quartermaster_nodes.registry.registry import NodeCatalog as FromInnerModule


def test_same_class_across_import_paths() -> None:
    """All three import paths resolve to the same class object."""
    assert NodeCatalog is FromRegistry
    assert NodeCatalog is FromInnerModule


def test_default_registry_is_nodecatalog_instance() -> None:
    assert isinstance(default_registry, NodeCatalog)


def test_noderegistry_alias_removed() -> None:
    """v0.6.0 dropped every NodeRegistry alias."""
    with pytest.raises(ImportError):
        from quartermaster_nodes import NodeRegistry  # noqa: F401
    with pytest.raises(ImportError):
        from quartermaster_nodes.registry import NodeRegistry  # noqa: F401
    with pytest.raises(ImportError):
        from quartermaster_nodes.registry.registry import NodeRegistry  # noqa: F401


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
        assert "SimpleNodeRegistry" in msg
        assert "quartermaster_engine" in msg
        assert "Instruction1" in msg

    def test_error_message_contains_working_example(self) -> None:
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
