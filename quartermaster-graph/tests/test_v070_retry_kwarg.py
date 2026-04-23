"""v0.7.0 — ``retry={"max_attempts": N, "on": <callable>}`` on the builder.

Covers the builder half of the graph-level node-retry primitive:

* ``.agent(retry=...)`` / ``.instruction(retry=...)`` / ``.instruction_form(retry=...)``
  stash ``retry_max_attempts`` on node metadata (integer, serialisable).
* The ``on`` predicate (optional) is stashed in the builder's
  ``_retry_predicates`` side-channel keyed by node name — NOT on node
  metadata, because callables can't survive JSON / YAML round-trips.
* Zero / negative ``max_attempts`` is normalised to 1 (no retries, just
  the initial attempt).
* Omitting ``retry=`` leaves metadata clean — no stale
  ``retry_max_attempts`` key.
* JSON / YAML round-trips preserve the integer; the callable is lost
  because it lives only in the in-memory registry.

Engine wiring is covered separately in
``quartermaster-engine/tests/test_v070_node_retry.py``.
"""

from __future__ import annotations

from quartermaster_graph import Graph
from quartermaster_graph.serialization import from_json, to_json


def _node(graph, name: str):
    spec = graph.build()
    for n in spec.nodes:
        if n.name == name:
            return n
    raise AssertionError(f"node {name!r} not in built graph")


def test_agent_retry_stashes_max_attempts_on_metadata() -> None:
    graph = Graph("g").user().agent("Research", retry={"max_attempts": 3})
    meta = _node(graph, "Research").metadata
    assert meta["retry_max_attempts"] == 3


def test_instruction_retry_stashes_max_attempts_on_metadata() -> None:
    graph = Graph("g").user().instruction("Say", retry={"max_attempts": 4})
    meta = _node(graph, "Say").metadata
    assert meta["retry_max_attempts"] == 4


def test_instruction_form_retry_stashes_max_attempts_on_metadata() -> None:
    graph = (
        Graph("g")
        .user()
        .instruction_form("Form", schema={"type": "object"}, retry={"max_attempts": 2})
    )
    meta = _node(graph, "Form").metadata
    assert meta["retry_max_attempts"] == 2


def test_no_retry_leaves_metadata_clean() -> None:
    """Omitting ``retry=`` must NOT leave a ``retry_max_attempts`` key."""
    graph = Graph("g").user().agent("Plain").instruction("Follow-up")
    for name in ("Plain", "Follow-up"):
        meta = _node(graph, name).metadata
        assert "retry_max_attempts" not in meta, (
            f"node {name!r} acquired retry_max_attempts without retry=; metadata={meta}"
        )


def test_predicate_stashed_in_retry_predicates_side_channel() -> None:
    def _on(capture):
        return "boom" in (capture.output_text or "")

    builder = Graph("g").user().agent("Research", retry={"max_attempts": 3, "on": _on})

    # The builder instance itself holds the side-channel dict (the
    # :class:`GraphBuilder` the ``.agent()`` call returned is the same
    # object we started from — chainable API).
    assert isinstance(builder._retry_predicates, dict)
    assert builder._retry_predicates["Research"] is _on

    # Building propagates the predicates onto the spec for the SDK
    # runner to pick up.  Attribute access — it's NOT a declared
    # Pydantic field (callables can't survive model_dump).
    spec = builder.build()
    assert getattr(spec, "_retry_predicates")["Research"] is _on


def test_predicate_keyed_by_node_name_across_multiple_nodes() -> None:
    def _on_a(capture):
        return False

    def _on_b(capture):
        return False

    builder = (
        Graph("g")
        .user()
        .instruction("A", retry={"max_attempts": 2, "on": _on_a})
        .agent("B", retry={"max_attempts": 3, "on": _on_b})
    )
    assert builder._retry_predicates == {"A": _on_a, "B": _on_b}


def test_retry_without_on_leaves_predicate_registry_empty() -> None:
    builder = Graph("g").user().agent("Research", retry={"max_attempts": 3})
    assert builder._retry_predicates == {}


def test_max_attempts_zero_is_normalised_to_one() -> None:
    graph = Graph("g").user().agent("R", retry={"max_attempts": 0})
    assert _node(graph, "R").metadata["retry_max_attempts"] == 1


def test_max_attempts_negative_is_normalised_to_one() -> None:
    graph = Graph("g").user().agent("R", retry={"max_attempts": -5})
    assert _node(graph, "R").metadata["retry_max_attempts"] == 1


def test_max_attempts_default_when_key_missing_is_one() -> None:
    """If the caller passes an otherwise-empty dict, treat it as no-retry."""
    graph = Graph("g").user().agent("R", retry={})
    # Empty dict is falsy → _apply_retry_spec returns early, leaving
    # metadata untouched (same as retry=None).
    assert "retry_max_attempts" not in _node(graph, "R").metadata


def test_json_round_trip_preserves_max_attempts() -> None:
    """The integer budget survives to_json / from_json; the callable does not.

    Node metadata is JSON-serialisable by contract, so the integer
    ``retry_max_attempts`` key round-trips cleanly.  The ``on=`` predicate
    lives in a builder-side registry and is intentionally lost across
    JSON transport — downstream integrators must re-attach predicates on
    the consumer side via a fresh builder.
    """

    def _on(capture):
        return False

    builder = Graph("g").user().agent("Research", retry={"max_attempts": 3, "on": _on})
    spec = builder.build()
    payload = to_json(spec)
    # ``to_json`` produces a JSON-compatible dict — verify the integer
    # survives into the dump (otherwise a silent drop would still
    # "round trip" to the same missing value).
    agent_node = next(n for n in payload["nodes"] if n["name"] == "Research")
    assert agent_node["metadata"]["retry_max_attempts"] == 3

    restored = from_json(payload)
    restored_meta = next(n for n in restored.nodes if n.name == "Research").metadata
    assert restored_meta["retry_max_attempts"] == 3

    # The callable is NOT on the restored spec (private side-channel was
    # never serialised to begin with).
    assert getattr(restored, "_retry_predicates", None) in (None, {})


def test_non_callable_predicate_raises_type_error() -> None:
    import pytest

    with pytest.raises(TypeError, match="callable predicate"):
        Graph("g").user().agent("R", retry={"max_attempts": 2, "on": "not a callable"})


def test_non_int_max_attempts_raises_value_error() -> None:
    import pytest

    with pytest.raises(ValueError, match="int-coercible"):
        Graph("g").user().agent("R", retry={"max_attempts": "three"})
