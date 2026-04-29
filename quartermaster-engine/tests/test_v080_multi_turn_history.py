"""v0.8.0 — multi-node graphs propagate proper user/assistant roles
through the wire format instead of squashing every prior node's output
into one giant ``role="user"`` blob.

Pre-v0.8.0 wire format for ``User → Agent1 → Agent2``:

    [system, {role:user, content:"[Agent1]: research notes\\n---\\nOriginal case: <input>"}]

Agent2 had no way to distinguish Agent1's output (input it should act
on) from its own past speech. v0.8.0 rebuilds this as:

    [system,
     {role:user, content:"<input>"},          # original user turn
     {role:user, content:"research notes"},   # Agent1 output, from Agent2's POV
     {role:user, content:"<input>"}]          # the trailing prompt

This file exercises the engine end-to-end: build a graph, run it
through MockProvider, intercept the outbound ``history`` kwarg per
node, and assert the per-node role translation matches the v0.8.0
contract.
"""

from __future__ import annotations

import pytest
from quartermaster_graph import Graph
from quartermaster_providers import register_local
from quartermaster_providers.testing import MockProvider
from quartermaster_providers.types import NativeResponse, TokenResponse

from quartermaster_engine import FlowRunner


@pytest.fixture
def registry_and_mock():
    """A registry whose ``mock`` provider records every (prompt, history)
    pair. Calls to ``mock.calls`` return them in order."""
    registry = register_local("ollama", base_url="http://stub:11434", default_model="m")
    answers = [
        TokenResponse(content="research notes here", stop_reason="stop"),
        TokenResponse(content="final summary", stop_reason="stop"),
    ]
    natives = [
        NativeResponse(
            text_content="research notes here", thinking=[], tool_calls=[], stop_reason="stop"
        ),
        NativeResponse(
            text_content="final summary", thinking=[], tool_calls=[], stop_reason="stop"
        ),
    ]
    mock = MockProvider(responses=answers, native_responses=natives)
    registry.unregister("ollama")
    registry.register_instance("ollama", mock)
    return registry, mock


# ── Test 1: User → Agent1 → Agent2 — the canonical bug from the audit ──


def test_two_agent_pipeline_passes_agent1_output_as_user_role(registry_and_mock) -> None:
    """The bug-of-record: when Agent2 runs, Agent1's output must arrive
    as a ``role="user"`` history entry (input to act on), NOT
    ``role="assistant"`` (its own past speech) and NOT crammed into the
    user prompt with a ``[Agent1]:`` prefix."""
    registry, mock = registry_and_mock

    graph = (
        Graph("two-agent-pipeline")
        .start()
        .user()
        .agent("Researcher")
        .agent("Summariser")
        .end()
        .build()
    )
    runner = FlowRunner(graph=graph, provider_registry=registry)
    result = runner.run("research Acme Corp")
    assert result.success, result.error

    # Two LLM calls — one per agent. The base ``stream_native_response``
    # shim used by MockProvider delegates to ``generate_native_response``,
    # so that's the recorded method.
    llm_calls = [c for c in mock.calls if c["method"] == "generate_native_response"]
    assert len(llm_calls) >= 2, llm_calls

    researcher_call = llm_calls[0]
    summariser_call = llm_calls[1]

    # Researcher (first agent) — no prior conversation, just the user
    # input as the trailing prompt; history is empty.
    assert researcher_call["history"] in (None, [])
    assert researcher_call["prompt"] == "research Acme Corp"

    # Summariser (second agent) — sees Researcher's output as a
    # user-role history entry, NOT as assistant.
    history = summariser_call["history"]
    assert history is not None and len(history) >= 1
    # The Researcher's output ("research notes here") must appear as
    # role="user" — that's the whole fix.
    assert {"role": "user", "content": "research notes here"} in history
    # And the prefix must NOT be present anywhere in the content.
    for entry in history:
        assert "[Researcher]" not in entry["content"]
        assert "[Summariser]" not in entry["content"]
    # The user input is also surfaced as a role=user history entry —
    # by ``role="assistant" if node_name==current_node else "user"``,
    # the user input has node_name=None (it's not a node output) so
    # it surfaces as user; that's correct.
    # Final prompt is still the user input.
    assert summariser_call["prompt"] == "research Acme Corp"


# ── Test 2: same-agent multi-turn (sora chat pattern) ────────────────


def test_same_agent_multi_iteration_keeps_user_assistant_alternation(
    registry_and_mock,
) -> None:
    """Within a single agent node, the node's own past output stays
    ``role="assistant"`` and user-supplied turns stay ``role="user"``.

    Builds the history directly via :func:`_build_history_for_node` to
    isolate the per-node role-translation contract from the rest of
    the engine. The end-to-end FlowRunner path is covered by the
    pipeline test above.
    """
    from quartermaster_engine.example_runner import _build_history_for_node

    pre_seeded_conversation = [
        {"role": "user", "content": "/stranka PIGO", "node_name": None},
        {"role": "assistant", "content": "PIGO d.o.o.", "node_name": "Assistant"},
        {"role": "user", "content": "in status?", "node_name": None},
        {"role": "assistant", "content": "Naročilo SO-123", "node_name": "Assistant"},
    ]
    history = _build_history_for_node(pre_seeded_conversation, "Assistant")
    assert history == [
        {"role": "user", "content": "/stranka PIGO"},
        {"role": "assistant", "content": "PIGO d.o.o."},
        {"role": "user", "content": "in status?"},
        {"role": "assistant", "content": "Naročilo SO-123"},
    ]


# ── Test 3: instruction_form after agent — the enrichment pattern ─────


def test_instruction_form_after_agent_sees_research_as_user_input(registry_and_mock) -> None:
    """Customer-enrichment shape: Research agent → InstructionForm
    schema-extractor. The extractor must see the research notes as a
    user message (input), not as its own past assistant turn."""
    registry, mock = registry_and_mock

    schema = {"type": "object", "properties": {"summary": {"type": "string"}}}
    graph = (
        Graph("research-then-extract")
        .start()
        .user()
        .agent("Research")
        .instruction_form("Extract", schema=schema)
        .end()
        .build()
    )
    runner = FlowRunner(graph=graph, provider_registry=registry)
    runner.run("enrich")
    # The mock's native_responses[1] doesn't return real JSON, so the
    # node's parse may fail — we don't care here, we care about the
    # outbound wire format on the call. The call still happens before
    # parsing.
    assert mock.calls

    extract_calls = [c for c in mock.calls if c["method"] == "generate_native_response"]
    # The extract node calls generate_native_response with a forced
    # tool call. Its history should include the Research agent's output
    # as role="user".
    if extract_calls:
        history = extract_calls[-1]["history"]
        assert history is not None
        assert any(entry == {"role": "user", "content": "research notes here"} for entry in history)
