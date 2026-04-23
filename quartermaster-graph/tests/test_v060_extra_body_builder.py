"""v0.6.0 — ``agent(..., extra_body={...})`` and friends stash on node metadata.

End-to-end: DSL kwarg → node metadata → engine's ``LLMConfig.extra_body``.
This file covers the builder half; ``test_v060_extra_body.py`` in
quartermaster-providers covers the provider half.
"""

from __future__ import annotations

from quartermaster_graph import Graph


def _meta_for(graph, name: str) -> dict:
    """Fetch the metadata of the first node whose name matches."""
    spec = graph.build()
    for node in spec.nodes:
        if node.name == name:
            return node.metadata
    raise AssertionError(f"node {name!r} not in built graph")


def test_agent_extra_body_stashed_as_llm_extra_body() -> None:
    graph = (
        Graph("g")
        .user()
        .agent(
            "Research",
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    )
    meta = _meta_for(graph, "Research")
    assert meta["llm_extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_instruction_extra_body_stashed_as_llm_extra_body() -> None:
    graph = Graph("g").user().instruction("Say", extra_body={"repetition_penalty": 1.15})
    meta = _meta_for(graph, "Say")
    assert meta["llm_extra_body"] == {"repetition_penalty": 1.15}


def test_instruction_form_extra_body_stashed() -> None:
    graph = (
        Graph("g")
        .user()
        .instruction_form(
            "Extract",
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    )
    meta = _meta_for(graph, "Extract")
    assert meta["llm_extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_no_extra_body_key_when_unused() -> None:
    """Don't pollute metadata with a None or empty-dict key for nodes that
    never opted in."""
    graph = Graph("g").user().agent("A")
    meta = _meta_for(graph, "A")
    assert "llm_extra_body" not in meta


def test_extra_body_dict_is_copied_into_metadata() -> None:
    """Post-build mutation of the caller's dict must not retroactively
    change the node metadata (otherwise caching layers that hash the spec
    break)."""
    payload = {"chat_template_kwargs": {"enable_thinking": False}}
    graph = Graph("g").user().agent("A", extra_body=payload)
    meta = _meta_for(graph, "A")
    assert meta["llm_extra_body"] is not payload


def test_roundtrip_preserved_through_to_json() -> None:
    """extra_body must survive JSON round-trip — it's plain JSON already
    and must stay on the spec."""
    from quartermaster_graph import from_json, to_json

    graph = Graph("g").user().agent("A", extra_body={"top_k": 40, "min_p": 0.05})
    spec = graph.build()
    payload = to_json(spec)
    restored = from_json(payload)

    a_node = next(n for n in restored.nodes if n.name == "A")
    assert a_node.metadata["llm_extra_body"] == {"top_k": 40, "min_p": 0.05}

    # Sanity: the intermediate payload actually carries it — not just
    # something the reconstructor invented. ``to_json`` returns a dict.
    a_payload = next(n for n in payload["nodes"] if n["name"] == "A")
    assert a_payload["metadata"]["llm_extra_body"] == {"top_k": 40, "min_p": 0.05}
