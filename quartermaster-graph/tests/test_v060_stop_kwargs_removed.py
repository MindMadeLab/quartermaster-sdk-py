"""Lock in the v0.6.0 removal of the legacy ``.end(stop=...)`` kwarg.

v0.3.1 demoted the ``stop`` kwarg to a deprecated no-op (accepted for
v0.3.0 call-site compatibility, silently ignored).  v0.6.0 drops the
shim entirely — ``.end(stop=...)`` now raises ``TypeError`` exactly as
any other unexpected-keyword signature mismatch.

These tests fail fast if anyone ever tries to re-introduce the shim.
"""

from __future__ import annotations

import pytest

from quartermaster_graph.builder import GraphBuilder
from quartermaster_graph.serialization import from_json, to_json


def test_graph_end_stop_true_raises_type_error() -> None:
    """``.end(stop=True)`` on the main GraphBuilder must raise TypeError."""

    g = GraphBuilder("x").start().user().agent()
    with pytest.raises(TypeError):
        g.end(stop=True)


def test_graph_end_stop_false_raises_type_error() -> None:
    """``.end(stop=False)`` on the main GraphBuilder must raise TypeError."""

    g = GraphBuilder("x").start().user().agent()
    with pytest.raises(TypeError):
        g.end(stop=False)


def test_branch_end_stop_true_raises_type_error() -> None:
    """``.end(stop=True)`` on a branch builder must raise TypeError.

    Exercises the sibling removal in the ``_BranchBuilder.end`` shape
    (the branch-scoped .end() that registers a pending endpoint).
    """

    # A user_decision opens a branch builder whose .end() we want to
    # test.  We don't need to call .build() — the TypeError fires
    # before the branch fully closes.
    gb = GraphBuilder("x").start().user()
    branch = gb.user_decision("pick")
    with pytest.raises(TypeError):
        branch.end(stop=True)


def test_start_stop_kwarg_raises_type_error() -> None:
    """``.start(stop=True)`` was never officially supported — stays unsupported.

    The v0.6.0 audit flagged a hypothetical ``.start(stop=...)`` shim;
    none existed in the codebase (only ``.end`` ever carried it).  This
    test locks that in: passing ``stop`` to ``.start()`` must raise.
    """

    gb = GraphBuilder("x")
    with pytest.raises(TypeError):
        gb.start(stop=True)


def test_end_with_no_kwargs_still_works() -> None:
    """Happy path: ``.end()`` builds a terminating End node as before."""

    spec = GraphBuilder("x").start().user().agent().end().build()
    end_names = [n.name for n in spec.nodes]
    assert "End" in end_names


def test_start_with_no_kwargs_still_works() -> None:
    """Happy path: bare ``.start()`` still creates / reuses the Start node."""

    gb = GraphBuilder("x").start()
    # start() is idempotent, so calling twice must not raise either.
    gb.start()


def test_user_agent_end_graph_round_trips_through_json() -> None:
    """Sanity: removing the shim did not break serialisation of a simple graph.

    Builds a ``user() → agent() → end()`` flow and round-trips it via
    ``to_json`` / ``from_json`` to confirm the node and edge shape still
    matches the pre-v0.6.0 wire format.
    """

    spec = GraphBuilder("rt").start().user().agent().end().build()
    payload = to_json(spec)
    restored = from_json(payload)

    assert [n.name for n in restored.nodes] == [n.name for n in spec.nodes]
    assert len(restored.edges) == len(spec.edges)
