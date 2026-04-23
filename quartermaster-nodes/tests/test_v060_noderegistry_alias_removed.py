"""Negative regression tests — v0.6.0 dropped the ``NodeRegistry`` alias
that pointed at ``NodeCatalog``. Callers must now use ``NodeCatalog``
directly.

The removal resolves a long-standing naming collision with
``quartermaster_engine.nodes.NodeRegistry`` (a completely different
runtime-executor Protocol). Keeping both around was a recurring source of
confusion: ``FlowRunner(node_registry=quartermaster_nodes.NodeRegistry())``
would crash with ``AttributeError: get_executor``.
"""

from __future__ import annotations

import pytest


def test_node_catalog_still_importable() -> None:
    """Canonical name stays."""
    from quartermaster_nodes import NodeCatalog

    assert NodeCatalog.__name__ == "NodeCatalog"


def test_node_registry_alias_removed_from_top_level() -> None:
    with pytest.raises(ImportError):
        from quartermaster_nodes import NodeRegistry  # noqa: F401


def test_node_registry_alias_removed_from_registry_package() -> None:
    with pytest.raises(ImportError):
        from quartermaster_nodes.registry import NodeRegistry  # noqa: F401


def test_node_registry_alias_removed_from_registry_module() -> None:
    with pytest.raises(ImportError):
        from quartermaster_nodes.registry.registry import NodeRegistry  # noqa: F401


def test_node_registry_not_in_dunder_all() -> None:
    import quartermaster_nodes
    from quartermaster_nodes import registry as registry_pkg

    assert "NodeRegistry" not in quartermaster_nodes.__all__
    assert "NodeRegistry" not in registry_pkg.__all__


def test_engine_node_registry_untouched() -> None:
    """The engine has a completely different ``NodeRegistry`` Protocol.
    Removing the nodes-package alias must NOT affect it."""
    from quartermaster_engine.nodes import NodeRegistry as EngineNodeRegistry

    assert EngineNodeRegistry.__name__ == "NodeRegistry"
