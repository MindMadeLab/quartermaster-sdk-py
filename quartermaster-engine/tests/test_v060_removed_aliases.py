"""Negative regression tests — v0.6.0 dropped the ``_build_registry``
backward-compat shim. The canonical name is ``build_default_registry``.
"""

from __future__ import annotations

import pytest


def test_build_default_registry_still_importable() -> None:
    """The canonical name stays — only the underscore alias went away."""
    from quartermaster_engine.example_runner import build_default_registry

    assert callable(build_default_registry)


def test_private_build_registry_shim_removed() -> None:
    """``_build_registry`` was a v0.1.x-era alias. It's gone in v0.6.0."""
    from quartermaster_engine import example_runner

    assert not hasattr(example_runner, "_build_registry"), (
        "_build_registry was a backward-compat shim for 0.1.x callers; "
        "use build_default_registry instead"
    )


def test_build_default_registry_round_trip() -> None:
    """Calling the canonical name produces a usable registry."""
    from quartermaster_engine.example_runner import build_default_registry
    from quartermaster_engine.nodes import SimpleNodeRegistry
    from quartermaster_providers import ProviderRegistry

    reg = build_default_registry(ProviderRegistry(auto_configure=False))
    assert isinstance(reg, SimpleNodeRegistry)


def test_agentgraph_reexport_removed_from_engine_types() -> None:
    """v0.6.0 also dropped the AgentGraph re-export that quartermaster_engine.types
    forwarded from quartermaster_graph.models."""
    from quartermaster_engine import types

    with pytest.raises(ImportError):
        from quartermaster_engine.types import AgentGraph  # noqa: F401
    assert not hasattr(types, "AgentGraph")
