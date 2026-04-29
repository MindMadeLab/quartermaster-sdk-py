"""v0.8.1 — ``qm.run(history=[...])`` and ``qm.run.stream(history=[...])``
seed multi-turn conversation directly into the engine's
``__conversation__`` so the first LLM node sees prior turns as proper
user/assistant alternation.

Pre-v0.8.1 the only way to surface chat history was the ``session=``
kwarg, which folded prior turns into the ``user_input`` string with
``Uporabnik:``/``Asistent:``-style markers — same single-user-message
wire-format bug that v0.8.0 fixed at the engine layer. v0.8.1 closes
the gap at the SDK runner so chat consumers get clean multi-turn
without doing memory plumbing themselves.
"""

from __future__ import annotations

import quartermaster_sdk as qm
from quartermaster_graph import Graph
from quartermaster_providers import register_local
from quartermaster_providers.testing import MockProvider
from quartermaster_providers.types import NativeResponse, TokenResponse


def _mock_registry():
    registry = register_local("ollama", base_url="http://stub:11434", default_model="m")
    mock = MockProvider(
        responses=[TokenResponse(content="reply", stop_reason="stop")],
        native_responses=[
            NativeResponse(
                text_content="reply", thinking=[], tool_calls=[], stop_reason="stop"
            )
        ],
    )
    registry.unregister("ollama")
    registry.register_instance("ollama", mock)
    return registry, mock


def _chat_graph():
    return Graph("chat").start().user().instruction("Assistant").end().build()


class TestRunHistoryKwarg:
    """``qm.run(history=...)`` pre-seeds ``__conversation__`` with the
    provided turns. The first LLM node's outbound history is the seed
    (translated by ``_build_history_for_node``) and the trailing prompt
    is the new ``user_input``."""

    def test_history_seeds_conversation_for_first_llm_call(self) -> None:
        registry, mock = _mock_registry()
        qm.configure(registry=registry)

        history = [
            {"role": "user", "content": "/stranka PIGO"},
            {"role": "assistant", "content": "PIGO d.o.o."},
        ]

        qm.run(_chat_graph(), "in status?", history=history)

        # Find the LLM call.
        llm_calls = [c for c in mock.calls if c["method"] == "generate_text_response"]
        assert llm_calls, mock.calls
        last = llm_calls[-1]

        # The seed history should appear, with roles preserved.
        assert {"role": "user", "content": "/stranka PIGO"} in last["history"]
        assert {"role": "assistant", "content": "PIGO d.o.o."} in last["history"]
        # And the trailing prompt is the new user message.
        assert last["prompt"] == "in status?"

    def test_history_none_keeps_pre_v081_behaviour(self) -> None:
        """No ``history=`` kwarg → no seeded conversation, single user
        message at the wire format. Existing callers don't break."""
        registry, mock = _mock_registry()
        qm.configure(registry=registry)

        qm.run(_chat_graph(), "hello")

        llm_calls = [c for c in mock.calls if c["method"] == "generate_text_response"]
        # Without history, an empty history list / None is sent.
        assert llm_calls[-1]["history"] in (None, [])

    def test_malformed_history_entries_skipped(self) -> None:
        """Defensive ingest — bad entries get filtered, valid ones
        pass through. Matches the engine-level history splicer."""
        registry, mock = _mock_registry()
        qm.configure(registry=registry)

        qm.run(
            _chat_graph(),
            "now",
            history=[
                {"role": "weird"},  # unknown role → skipped
                {"content": "no role"},  # missing role → skipped
                {"role": "user"},  # missing content → skipped
                {"role": "user", "content": "valid"},
            ],
        )
        last = [c for c in mock.calls if c["method"] == "generate_text_response"][-1]
        assert {"role": "user", "content": "valid"} in last["history"]
        assert all(
            entry.get("content") not in ("no role", "") for entry in last["history"]
        )


class TestStreamHistoryKwarg:
    """``qm.run.stream(history=...)`` plumbs through to the same
    seeding path. The thread-driven streaming runner must accept and
    forward the kwarg verbatim."""

    def test_stream_seeds_conversation_for_first_llm_call(self) -> None:
        registry, mock = _mock_registry()
        qm.configure(registry=registry)

        history = [
            {"role": "user", "content": "earlier turn"},
            {"role": "assistant", "content": "earlier reply"},
        ]
        chunks = list(qm.run.stream(_chat_graph(), "now", history=history))

        # The stream must have completed (DoneChunk last).
        assert chunks[-1].type in ("done", "error"), chunks
        last = [c for c in mock.calls if c["method"] == "generate_text_response"][-1]
        assert {"role": "user", "content": "earlier turn"} in last["history"]
        assert {"role": "assistant", "content": "earlier reply"} in last["history"]


class TestHistoryWinsOverLegacySession:
    """When both ``history=`` and ``session=`` are passed, the explicit
    ``history`` takes precedence — the legacy blob path doesn't fire."""

    def test_explicit_history_overrides_session(self) -> None:
        registry, mock = _mock_registry()
        qm.configure(registry=registry)

        from quartermaster_sdk._session import ChatTurn, InMemorySessionStore

        session = InMemorySessionStore()
        # Pre-populate session with a turn that would otherwise get
        # folded into user_input as a "Assistant: ..." prefix.
        session.append("sid-1", ChatTurn(role="assistant", content="from session"))

        # Pass BOTH history and session.
        qm.run(
            _chat_graph(),
            "now",
            session=session,
            session_id="sid-1",
            history=[{"role": "assistant", "content": "from explicit history"}],
        )

        last = [c for c in mock.calls if c["method"] == "generate_text_response"][-1]
        # Explicit history shows up, session blob does NOT.
        assert {"role": "assistant", "content": "from explicit history"} in last[
            "history"
        ]
        # The session content shouldn't have leaked into prompt either.
        assert "from session" not in last["prompt"]
