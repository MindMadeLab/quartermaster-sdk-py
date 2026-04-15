"""Guard tests for the SDK's public surface.

The SDK is a meta-package that re-exports the curated subset of names the
canonical onboarding snippet uses.  These tests fail loudly if anyone
accidentally drops an export or version-skews ``__version__`` against the
sub-packages — that drift was caught manually in the v0.1.4 audit and is the
kind of thing we want a CI-runnable test for from now on.
"""

from __future__ import annotations

import importlib

import pytest

import quartermaster_sdk

# Names that v0.1.4 onwards must be importable from ``quartermaster_sdk``.
# Update this list when adding a new public re-export — the test will tell
# you in CI if you forgot.
_REQUIRED_PUBLIC_NAMES: tuple[str, ...] = (
    # Graph builder + spec
    "Graph",
    "GraphBuilder",
    "GraphSpec",
    "AgentGraph",  # deprecated alias kept for back-compat
    "NodeType",
    # Engine — runner + node-registry surface
    "FlowRunner",
    "FlowResult",
    "NodeRegistry",
    "NodeExecutor",
    "NodeResult",
    "SimpleNodeRegistry",
    "LLMExecutor",
    "AgentExecutor",
    "build_default_registry",
    "run_graph",
    # Providers — config, sync chat result, registry helpers
    "LLMConfig",
    "ChatResult",
    "ProviderRegistry",
    "register_local",
)


class TestPublicSurface:
    """Every name in ``_REQUIRED_PUBLIC_NAMES`` must resolve and be in ``__all__``."""

    @pytest.mark.parametrize("name", _REQUIRED_PUBLIC_NAMES)
    def test_attribute_resolves(self, name: str) -> None:
        assert hasattr(quartermaster_sdk, name), (
            f"quartermaster_sdk.{name} disappeared — check the re-exports "
            "in src/quartermaster_sdk/__init__.py."
        )

    @pytest.mark.parametrize("name", _REQUIRED_PUBLIC_NAMES)
    def test_in_dunder_all(self, name: str) -> None:
        assert name in quartermaster_sdk.__all__, (
            f"{name!r} is importable but missing from __all__; star-imports "
            "(``from quartermaster_sdk import *``) won't pick it up."
        )

    def test_dunder_all_contains_no_dead_names(self) -> None:
        """Every name in ``__all__`` must actually resolve on the module."""
        dead = [
            n for n in quartermaster_sdk.__all__ if not hasattr(quartermaster_sdk, n)
        ]
        assert not dead, f"__all__ lists names that don't exist: {dead}"


class TestVersionAlignment:
    """``quartermaster_sdk.__version__`` must match every sub-package's
    ``__version__``.  The publish workflow seds them all to the same
    value at release time — drift here means either someone hand-edited
    a version string, or the workflow's sed list missed a package."""

    @pytest.mark.parametrize(
        "subpackage",
        [
            "quartermaster_engine",
            "quartermaster_graph",
            "quartermaster_providers",
            "quartermaster_tools",
            "quartermaster_nodes",
        ],
    )
    def test_subpackage_version_matches_sdk(self, subpackage: str) -> None:
        sdk_version = quartermaster_sdk.__version__
        sub = importlib.import_module(subpackage)
        assert sub.__version__ == sdk_version, (
            f"{subpackage}.__version__ = {sub.__version__!r} but "
            f"quartermaster_sdk.__version__ = {sdk_version!r}. "
            "The publish workflow should keep these in lock-step — "
            "see RELEASING.md."
        )


class TestSnippetCompiles:
    """The v0.1.4 release-note snippet must at least import + parse cleanly."""

    def test_snippet_imports(self) -> None:
        # No execution — just prove the symbols exist.  Actual end-to-end
        # behaviour against a mock is in
        # quartermaster-engine/tests/test_smoke_simple_graph.py.
        from quartermaster_sdk import (
            ChatResult,  # noqa: F401
            FlowRunner,  # noqa: F401
            Graph,  # noqa: F401
            register_local,  # noqa: F401
        )
