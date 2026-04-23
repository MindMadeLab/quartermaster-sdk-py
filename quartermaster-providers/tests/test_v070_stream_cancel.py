"""v0.7.0 — provider-level stream cancellation.

The provider polls ``should_cancel()`` between streaming chunks. The
engine wraps each provider call with ``set_cancel_check(lambda:
ctx.cancelled)``. When cancellation fires, the provider closes the
openai ``AsyncStream`` (which closes the underlying ``httpx`` response,
which releases the vLLM slot).

These tests exercise the primitive without a live LLM server.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from quartermaster_providers.cancellation import (
    set_cancel_check,
    should_cancel,
)
from quartermaster_providers.config import LLMConfig
from quartermaster_providers.providers.openai import OpenAIProvider


# ── Unit: contextvar push/pop/default ───────────────────────────────


class TestCancelContextVar:
    def test_default_is_false(self) -> None:
        """No cancel check installed → should_cancel() is a safe False."""

        async def _probe() -> bool:
            return should_cancel()

        assert asyncio.run(_probe()) is False

    def test_installed_true(self) -> None:
        async def _probe() -> bool:
            with set_cancel_check(lambda: True):
                return should_cancel()

        assert asyncio.run(_probe()) is True

    def test_installed_false(self) -> None:
        async def _probe() -> bool:
            with set_cancel_check(lambda: False):
                return should_cancel()

        assert asyncio.run(_probe()) is False

    def test_pop_restores_previous(self) -> None:
        """Nested context managers stack — exiting the inner one
        restores the outer's check, not the default."""

        async def _probe() -> tuple[bool, bool, bool]:
            outer_before: bool = False
            outer_during: bool = False
            outer_after: bool = False
            with set_cancel_check(lambda: True):
                outer_before = should_cancel()  # outer says True
                with set_cancel_check(lambda: False):
                    outer_during = should_cancel()  # inner says False
                outer_after = should_cancel()  # outer again True
            return outer_before, outer_during, outer_after

        a, b, c = asyncio.run(_probe())
        assert a is True
        assert b is False
        assert c is True

    def test_raising_predicate_treated_as_no_cancel(self) -> None:
        """A buggy predicate must not bring down the streaming path."""

        def _raise() -> bool:
            raise RuntimeError("bad predicate")

        async def _probe() -> bool:
            with set_cancel_check(_raise):
                return should_cancel()

        assert asyncio.run(_probe()) is False

    def test_none_strips_cancellation(self) -> None:
        """Explicitly passing None inside a nested block suppresses
        an outer cancel — useful for cleanup code that must complete
        even after the outer flow was cancelled."""

        async def _probe() -> tuple[bool, bool]:
            outer: bool = False
            inner: bool = False
            with set_cancel_check(lambda: True):
                outer = should_cancel()
                with set_cancel_check(None):
                    inner = should_cancel()
            return outer, inner

        a, b = asyncio.run(_probe())
        assert a is True
        assert b is False


# ── Integration: openai provider stream cancels mid-iteration ────────


def _make_chunk(text: str, finish: str | None = None) -> MagicMock:
    """Fake one streaming chunk from openai.AsyncStream."""
    delta = MagicMock()
    delta.content = text
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish
    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.usage = None
    return chunk


class _FakeAsyncStream:
    """Mimics openai.AsyncStream: async-iterable + close coroutine."""

    def __init__(self, chunks: list[MagicMock]) -> None:
        self._chunks = list(chunks)
        self.closed = False
        # The number of chunks actually pulled by the consumer before
        # a close. Helps tests assert "we stopped at iteration N".
        self.yielded_count = 0

    def __aiter__(self) -> "_FakeAsyncStream":
        return self

    async def __anext__(self) -> MagicMock:
        if self.closed or not self._chunks:
            raise StopAsyncIteration
        self.yielded_count += 1
        return self._chunks.pop(0)

    async def close(self) -> None:
        self.closed = True


class TestStreamCancellation:
    def test_cancel_mid_stream_closes_response(self) -> None:
        """Check returns True after the 3rd chunk — stream.close() runs,
        consumer sees a ``cancelled`` sentinel, remaining chunks are
        NOT emitted."""
        chunks = [_make_chunk(f"t{i} ", finish=None) for i in range(10)]
        fake_stream = _FakeAsyncStream(chunks)

        provider = OpenAIProvider(api_key="sk-test")
        provider._client = MagicMock()
        provider._client.chat.completions.create = AsyncMock(return_value=fake_stream)

        # Flip True after 3 chunks have been observed.
        state = {"calls": 0}

        def _check() -> bool:
            state["calls"] += 1
            # We poll AFTER yielding; so check-after-3rd-yield is
            # the 3rd invocation → flip True.
            return state["calls"] >= 3

        async def _drive() -> tuple[list[str], bool]:
            out: list[str] = []
            with set_cancel_check(_check):
                config = LLMConfig(model="gpt-4o", provider="openai", stream=True)
                stream = await provider.generate_text_response("hi", config)
                async for token in stream:
                    if token.stop_reason == "cancelled":
                        out.append("<CANCELLED>")
                        break
                    if token.content:
                        out.append(token.content)
            return out, fake_stream.closed

        out, closed = asyncio.run(_drive())
        # We see the first 3 tokens (the check fires after each yield).
        # The cancel sentinel terminates the stream before token 4.
        assert out[:3] == ["t0 ", "t1 ", "t2 "]
        assert out[-1] == "<CANCELLED>"
        assert closed, "Provider must close the underlying AsyncStream on cancel"
        # Fewer than all 10 chunks were pulled.
        assert fake_stream.yielded_count < 10

    def test_no_cancel_check_drains_fully(self) -> None:
        """When no cancel check is installed, the stream drains every
        chunk — the baseline we want the cancellation code to preserve
        for the happy path."""
        chunks = [_make_chunk(f"t{i} ", finish=None) for i in range(5)]
        fake_stream = _FakeAsyncStream(chunks)

        provider = OpenAIProvider(api_key="sk-test")
        provider._client = MagicMock()
        provider._client.chat.completions.create = AsyncMock(return_value=fake_stream)

        async def _drive() -> list[str]:
            out: list[str] = []
            config = LLMConfig(model="gpt-4o", provider="openai", stream=True)
            stream = await provider.generate_text_response("hi", config)
            async for token in stream:
                if token.content:
                    out.append(token.content)
            return out

        out = asyncio.run(_drive())
        assert out == ["t0 ", "t1 ", "t2 ", "t3 ", "t4 "]
        # And the finally-close still fires on natural end-of-stream:
        assert fake_stream.closed

    def test_cancel_check_that_stays_false_is_never_triggered(self) -> None:
        """A check installed but reporting False must behave exactly
        like no check at all — full drain, clean close."""
        chunks = [_make_chunk(f"t{i} ", finish=None) for i in range(4)]
        fake_stream = _FakeAsyncStream(chunks)

        provider = OpenAIProvider(api_key="sk-test")
        provider._client = MagicMock()
        provider._client.chat.completions.create = AsyncMock(return_value=fake_stream)

        async def _drive() -> list[str]:
            out: list[str] = []
            with set_cancel_check(lambda: False):
                config = LLMConfig(model="gpt-4o", provider="openai", stream=True)
                stream = await provider.generate_text_response("hi", config)
                async for token in stream:
                    if token.content:
                        out.append(token.content)
            return out

        out = asyncio.run(_drive())
        assert out == ["t0 ", "t1 ", "t2 ", "t3 "]
        assert fake_stream.closed
