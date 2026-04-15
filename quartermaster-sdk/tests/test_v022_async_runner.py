"""Smoke tests for the v0.2.2 async runner.

Covers the new :func:`quartermaster_sdk.arun` coroutine and its
``.stream`` async-iterator counterpart.  The implementation sits on
top of the existing sync ``FlowRunner`` via :func:`asyncio.to_thread`,
so the happy-path assertions here mirror the sync suite in
``test_v020_surface.py`` but exercise the ``await`` / ``async for``
call sites instead.

Provider is stubbed with :class:`MockProvider` — no real LLM is hit.
"""

from __future__ import annotations

import openai  # noqa: F401 — eager import, see test_ollama_chat.py for context

import pytest

import quartermaster_sdk as qm
from quartermaster_providers import ProviderRegistry
from quartermaster_providers.testing import MockProvider
from quartermaster_providers.types import NativeResponse, TokenResponse


# ── Helpers ───────────────────────────────────────────────────────────


def _mock_registry(
    text: str = "canned",
    native_text: str | None = None,
) -> tuple[ProviderRegistry, MockProvider]:
    """Copy of the helper from test_v020_surface.py.

    A ProviderRegistry wired to a MockProvider registered as ``ollama``,
    with both streamed-token and native-response channels primed so the
    engine picks a consistent reply regardless of which path it takes.
    """
    mock = MockProvider(
        responses=[TokenResponse(content=text, stop_reason="stop")],
        native_responses=[
            NativeResponse(
                text_content=native_text if native_text is not None else text,
                thinking=[],
                tool_calls=[],
                stop_reason="stop",
            )
        ],
    )
    reg = ProviderRegistry(auto_configure=False)
    reg.register_instance("ollama", mock)
    reg.set_default_provider("ollama")
    reg.set_default_model("ollama", "mock-model")
    return reg, mock


@pytest.fixture(autouse=True)
def _reset_config():
    """Reset module-level config between tests."""
    qm.reset_config()
    yield
    qm.reset_config()


# ── arun() ────────────────────────────────────────────────────────────


async def test_arun_returns_result():
    """``await qm.arun(graph, input)`` resolves to a populated ``Result``.

    This is the direct async analogue of ``qm.run(...)`` — the sync body
    runs on an ``asyncio.to_thread`` worker but the public contract is
    identical: a :class:`Result` whose ``.text`` contains the mock's
    canned reply and whose ``.success`` is ``True``.
    """
    reg, _ = _mock_registry("async canned")
    qm.configure(registry=reg)

    graph = qm.Graph("x").instruction("One").build()
    result = await qm.arun(graph, "hi")

    assert isinstance(result, qm.Result)
    assert result.success
    assert result.text == "async canned"


async def test_arun_stream_yields_chunks():
    """``async for chunk in qm.arun.stream(graph, input)`` yields the
    same typed :class:`Chunk` sequence as the sync ``run.stream``, and
    terminates with a :class:`DoneChunk` carrying the final result.

    Verifies at least one node-lifecycle pair (``node_start`` →
    ``node_finish``) surfaces between the open and close of the stream,
    so we know events are marshalling across the thread→loop boundary
    and not just the terminal done chunk.
    """
    reg, _ = _mock_registry("streaming async words")
    qm.configure(registry=reg)
    graph = qm.Graph("x").instruction("One").build()

    chunks = [chunk async for chunk in qm.arun.stream(graph, "hi")]
    types = [c.type for c in chunks]

    # Must end with a DoneChunk that carries a populated Result.
    assert types[-1] == "done"
    last = chunks[-1]
    assert isinstance(last, qm.DoneChunk)
    assert last.result.success
    assert last.result.text == "streaming async words"
    # Lifecycle events crossed the thread→loop boundary successfully.
    assert "node_start" in types
    assert "node_finish" in types
