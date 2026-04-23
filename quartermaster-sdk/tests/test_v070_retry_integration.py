"""v0.7.0 — end-to-end ``.agent(retry={...})`` through ``qm.run``.

Exercises the full surface: builder stashes ``retry_max_attempts`` on
node metadata + the ``on=`` predicate in the ``_retry_predicates``
side-channel; the SDK runner forwards both to :class:`FlowRunner`; the
engine re-runs the node until the predicate returns False or the budget
is exhausted.

The MockProvider scripts two distinct answers — the first should trip
the predicate (``"bad" in output_text``), the second should clear it.
End state: 2 provider calls, final capture text == "good".
"""

from __future__ import annotations

import pytest

import quartermaster_sdk as qm
from quartermaster_providers import ProviderRegistry
from quartermaster_providers.testing import MockProvider
from quartermaster_providers.types import NativeResponse, TokenResponse
from quartermaster_tools import get_default_registry


@pytest.fixture(autouse=True)
def _reset_sdk_state():
    """Reset SDK config + the default tool registry between tests."""
    qm.reset_config()
    get_default_registry().clear()
    yield
    qm.reset_config()
    get_default_registry().clear()


def _scripted_registry(texts: list[str]) -> tuple[ProviderRegistry, MockProvider]:
    """Build a ``ProviderRegistry`` whose ``MockProvider`` replays the
    given completion texts in order.

    Each text drives ONE ``generate_native_response`` call — perfect for
    agent nodes that degenerate to single-shot completions when no tools
    are attached.
    """
    mock = MockProvider(
        responses=[TokenResponse(content=t, stop_reason="stop") for t in texts],
        native_responses=[
            NativeResponse(
                text_content=t,
                thinking=[],
                tool_calls=[],
                stop_reason="stop",
            )
            for t in texts
        ],
    )
    reg = ProviderRegistry(auto_configure=False)
    reg.register_instance("mock", mock)
    reg.set_default_provider("mock")
    reg.set_default_model("mock", "test-model")
    return reg, mock


def test_retry_on_predicate_reruns_then_stops() -> None:
    """First answer ``"bad"`` → predicate True → retry; second answer
    ``"good"`` → predicate False → done.  Exactly 2 provider calls."""
    reg, mock = _scripted_registry(["bad", "good"])
    qm.configure(registry=reg)

    graph = (
        qm.Graph("chat")
        .user()
        .agent(
            "Research",
            capture_as="Research",
            retry={
                "max_attempts": 2,
                "on": lambda r: "bad" in (r.output_text or ""),
            },
        )
    )

    result = qm.run(graph, "go")
    assert result.success, result.error
    # The second attempt's "good" output should be the surviving capture.
    assert result.captures["Research"].output_text == "good"
    # And the mock was called exactly twice (no more, no fewer).
    native_calls = [c for c in mock.calls if c["method"] == "generate_native_response"]
    assert len(native_calls) == 2


def test_retry_budget_not_exceeded_when_predicate_keeps_firing() -> None:
    """Predicate stays True through every attempt — the runner must stop
    at ``max_attempts`` calls, NOT loop forever."""
    reg, mock = _scripted_registry(["bad", "still bad", "always bad"])
    qm.configure(registry=reg)

    graph = (
        qm.Graph("chat")
        .user()
        .agent(
            "Research",
            capture_as="Research",
            retry={
                "max_attempts": 2,
                "on": lambda r: "bad" in (r.output_text or ""),
            },
        )
    )

    qm.run(graph, "go")
    # max_attempts=2 → exactly 2 provider calls, never 3.
    native_calls = [c for c in mock.calls if c["method"] == "generate_native_response"]
    assert len(native_calls) == 2


def test_retry_emits_node_retried_custom_chunk_on_stream() -> None:
    """The ``node.retried`` engine event surfaces as a ``CustomChunk``
    on the streaming surface — integrators can filter it via
    ``stream.custom(name="node.retried")``."""
    reg, _mock = _scripted_registry(["bad", "good"])
    qm.configure(registry=reg)

    graph = (
        qm.Graph("chat")
        .user()
        .agent(
            "Research",
            capture_as="Research",
            retry={
                "max_attempts": 2,
                "on": lambda r: "bad" in (r.output_text or ""),
            },
        )
    )

    retried_chunks: list[dict] = []
    with qm.run.stream(graph, "go") as s:
        for chunk in s:
            if getattr(chunk, "name", None) == "node.retried":
                retried_chunks.append(dict(chunk.payload))

    assert len(retried_chunks) == 1
    assert retried_chunks[0]["node"] == "Research"
    assert retried_chunks[0]["attempt"] == 1
    assert retried_chunks[0]["reason"] == "predicate"
