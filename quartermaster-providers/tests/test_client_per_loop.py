"""Regression: OpenAIProvider's cached openai.AsyncOpenAI must not leak
across asyncio event loops.

The pattern this guards against: an outer ``qm.run()`` spins its own loop,
an ``@tool()`` body inside that flow calls ``qm.run()`` again which spins
a *second* loop on a worker thread, both invocations share the same
ProviderRegistry-cached provider instance. Before this fix, the provider
kept a single ``self._client`` whose httpx.AsyncClient and asyncio
primitives bound to whichever loop used it first — leaking
``RuntimeError: Event loop is closed`` and ``RuntimeError: <Event> is
bound to a different event loop`` as soon as the original loop tried its
next LLM call.

The fix: ``_get_client`` now tracks which loop the cached client was
created on and rebuilds the client when called from a different loop.
These tests don't hit a real network — they only assert the cache
behaves loop-locally.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

from quartermaster_providers.providers.openai import OpenAIProvider
from quartermaster_providers.providers.openai_compat import OpenAICompatibleProvider


class TestOpenAIProviderClientPerLoop:
    def test_same_loop_returns_same_client(self):
        """Inside a single loop, repeated _get_client() calls return the
        same cached instance — the fix must not kill connection-pool reuse
        within one run."""
        provider = OpenAIProvider(api_key="sk-test")

        async def _inner():
            return provider._get_client(), provider._get_client()

        a, b = asyncio.run(_inner())
        assert a is b, "client cache must be reused within the same loop"

    def test_new_loop_rebuilds_client(self):
        """Two consecutive ``asyncio.run()`` invocations should get
        distinct clients — the first loop is closed by the time the
        second call happens."""
        provider = OpenAIProvider(api_key="sk-test")

        async def _probe():
            return provider._get_client()

        first = asyncio.run(_probe())
        second = asyncio.run(_probe())

        assert first is not second, (
            "_get_client must rebuild the client when the previous loop "
            "has closed — otherwise asyncio primitives wedge"
        )

    def test_nested_loop_in_worker_thread_rebuilds_and_preserves_outer(self):
        """Simulates the real bug: outer agent's loop A holds a client;
        a tool on a ThreadPoolExecutor worker spins its own loop B and
        calls _get_client() inside; outer loop A then calls _get_client()
        again. The outer client must NOT have been replaced by the inner
        call — per-loop caching lets both coexist."""
        provider = OpenAIProvider(api_key="sk-test")

        async def _outer():
            outer_client = provider._get_client()

            def _tool_body():
                async def _inner_loop():
                    return provider._get_client()

                return asyncio.run(_inner_loop())

            with ThreadPoolExecutor(max_workers=1) as pool:
                inner_client = await asyncio.wrap_future(pool.submit(_tool_body))

            # Outer loop is still alive here; second call from loop A
            # must return the SAME object as the first outer call.
            outer_client_again = provider._get_client()
            return outer_client, inner_client, outer_client_again

        outer, inner, outer_again = asyncio.run(_outer())

        assert inner is not outer, (
            "nested loop's client must not be the outer loop's client — "
            "reusing one AsyncClient across loops wedges httpcore primitives"
        )
        assert outer_again is outer, (
            "outer loop's cached client must survive a nested qm.run() on a "
            "different loop — otherwise every tool invocation trashes the "
            "outer agent's connection pool"
        )


class TestOpenAICompatibleProviderClientPerLoop:
    """Same guarantees for the OpenAI-compatible subclass — it has its own
    ``_get_client`` override to build the inner httpx.AsyncClient for
    basic auth / extra headers, and must also respect loop changes."""

    def test_new_loop_rebuilds_client(self):
        provider = OpenAICompatibleProvider(
            base_url="http://localhost:11434/v1",
            api_key="no-key",
            auth_method="none",
        )

        async def _probe():
            return provider._get_client()

        first = asyncio.run(_probe())
        second = asyncio.run(_probe())
        assert first is not second

    def test_basic_auth_rebuilds_on_loop_change(self):
        """Basic-auth path also goes through a fresh httpx.AsyncClient
        hand-off; confirm that path doesn't leak across loops either."""
        provider = OpenAICompatibleProvider(
            base_url="http://localhost:11434/v1",
            auth_method="basic",
            auth_credentials=("user", "pass"),
        )

        async def _probe():
            return provider._get_client()

        first = asyncio.run(_probe())
        second = asyncio.run(_probe())
        assert first is not second
