"""v0.9.0: close httpx/httpcore2 async pools before the event loop dies.

Regression for GitHub #96 / #97. After a successful ``qm.instruction()``
against an OpenAI-compatible endpoint, stderr printed::

    RuntimeError: generator didn't stop after athrow()

The call succeeded; the noise was ``asyncio.run()`` shutting the loop
while ``openai.AsyncOpenAI`` still held an ``httpx.AsyncClient`` whose
httpcore2 ``PoolByteStream`` async generator was still open.

These tests mock openai/httpx — no network.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from quartermaster_providers.providers.openai import OpenAIProvider, _aclose_http_client
from quartermaster_providers.providers.openai_compat import OpenAICompatibleProvider
from quartermaster_providers.registry import ProviderRegistry
from quartermaster_providers.testing import MockProvider


# ── Fakes ─────────────────────────────────────────────────────────────


class FakeHttpxClient:
    """Stand-in for ``httpx.AsyncClient`` — records ``aclose()``."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.aclose_calls = 0
        self.is_closed = False

    async def aclose(self) -> None:
        self.aclose_calls += 1
        self.is_closed = True


class FakeAsyncOpenAI:
    """Stand-in for ``openai.AsyncOpenAI``.

    Mirrors the real SDK: ``close()`` awaits the underlying httpx
    client's ``aclose()``. Instances are collected so tests can assert
    without holding the provider's private cache.
    """

    instances: list[FakeAsyncOpenAI] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.close_calls = 0
        self._client = kwargs.get("http_client") or FakeHttpxClient()
        type(self).instances.append(self)

    async def close(self) -> None:
        self.close_calls += 1
        aclose = getattr(self._client, "aclose", None)
        if callable(aclose):
            result = aclose()
            if hasattr(result, "__await__"):
                await result


class StubbornPoolStream:
    """Mimics httpcore2 ``PoolByteStream.__aiter__``.

    Yields one chunk then parks. If the event loop shuts down the
    generator via ``athrow(GeneratorExit)`` *without* an explicit
    ``aclose()`` first, it raises the same ``RuntimeError`` httpcore2
    logs. A clean ``aclose()`` sets the flag so GeneratorExit is
    absorbed.
    """

    def __init__(self) -> None:
        self._explicitly_closed = False
        self._gen: object | None = None

    async def start(self) -> bytes:
        async def _pool():
            try:
                yield b"ok"
                await asyncio.Future()  # park until athrow / aclose
            except GeneratorExit:
                if not self._explicitly_closed:
                    raise RuntimeError("generator didn't stop after athrow()") from None
                return

        self._gen = _pool()
        return await self._gen.__anext__()

    async def aclose(self) -> None:
        self._explicitly_closed = True
        gen = self._gen
        if gen is not None:
            await gen.aclose()


# ── Helpers ───────────────────────────────────────────────────────────


def _patch_openai():
    FakeAsyncOpenAI.instances = []
    return patch(
        "openai.AsyncOpenAI",
        FakeAsyncOpenAI,
    )


# ── Unraisable / GeneratorExit ────────────────────────────────────────


class TestFakePoolGeneratorExit:
    """Reproduce the httpcore2 traceback, then show aclose prevents it."""

    def test_leaked_async_gen_emits_unraisable_on_loop_shutdown(self, caplog):
        """Leaving the pool generator open across ``asyncio.run()`` must
        log the RuntimeError — this is the bug the provider ``aclose()``
        exists to prevent. asyncio reports it via the ``asyncio`` logger
        (``an error occurred during closing of asynchronous generator``),
        not always via ``sys.unraisablehook``."""

        async def _leak() -> None:
            stream = StubbornPoolStream()
            await stream.start()
            # Intentionally do not aclose — loop shutdown athrow()s.

        with caplog.at_level(logging.ERROR, logger="asyncio"):
            asyncio.run(_leak())

        assert "an error occurred during closing of asynchronous generator" in caplog.text
        assert "generator didn't stop after athrow()" in caplog.text

    def test_aclose_before_loop_shutdown_is_silent(self, caplog):
        """The same generator, closed before ``asyncio.run()`` returns,
        must not log the httpcore2 athrow RuntimeError."""

        async def _clean() -> None:
            stream = StubbornPoolStream()
            await stream.start()
            await stream.aclose()

        with caplog.at_level(logging.ERROR, logger="asyncio"):
            asyncio.run(_clean())

        assert "generator didn't stop after athrow()" not in caplog.text


# ── Provider aclose ───────────────────────────────────────────────────


class TestOpenAIProviderAclose:
    def test_aclose_invokes_client_close(self):
        provider = OpenAIProvider(api_key="sk-test")
        with _patch_openai():

            async def _inner():
                client = provider._get_client()
                await provider.aclose()
                return client

            client = asyncio.run(_inner())

        assert client.close_calls == 1
        assert client._client.aclose_calls == 1
        assert provider._clients_by_loop == {}

    def test_aclose_then_get_client_rebuilds(self):
        provider = OpenAIProvider(api_key="sk-test")
        with _patch_openai():

            async def _inner():
                first = provider._get_client()
                await provider.aclose()
                second = provider._get_client()
                return first, second

            first, second = asyncio.run(_inner())

        assert first is not second
        assert first.close_calls == 1
        assert second.close_calls == 0

    def test_nested_aclose_preserves_outer_loop_client(self):
        """Inner ``asyncio.run()`` + ``aclose()`` must not close the
        outer loop's cached client — nested ``qm.run()`` from a
        ``@tool()`` body depends on this."""
        provider = OpenAIProvider(api_key="sk-test")
        with _patch_openai():

            async def _outer():
                outer_client = provider._get_client()

                def _tool_body():
                    async def _inner_loop():
                        inner = provider._get_client()
                        await provider.aclose()
                        return inner

                    return asyncio.run(_inner_loop())

                with ThreadPoolExecutor(max_workers=1) as pool:
                    inner_client = await asyncio.wrap_future(pool.submit(_tool_body))

                outer_again = provider._get_client()
                return outer_client, inner_client, outer_again

            outer, inner, outer_again = asyncio.run(_outer())

        assert inner is not outer
        assert inner.close_calls == 1
        assert outer.close_calls == 0
        assert outer_again is outer

    def test_async_context_manager_closes(self):
        provider = OpenAIProvider(api_key="sk-test")
        with _patch_openai():

            async def _inner():
                async with provider:
                    client = provider._get_client()
                return client

            client = asyncio.run(_inner())

        assert client.close_calls == 1

    def test_close_from_running_loop_raises(self):
        provider = OpenAIProvider(api_key="sk-test")

        async def _inner():
            provider.close()

        with pytest.raises(RuntimeError, match="running event loop"):
            asyncio.run(_inner())

    def test_injected_client_is_not_closed_by_aclose(self):
        """Back-compat: tests that assign ``provider._client = fake``
        must not have that fake closed (it is not in the per-loop cache)."""
        provider = OpenAIProvider(api_key="sk-test")
        fake = FakeAsyncOpenAI()

        async def _close_fake():
            raise AssertionError("injected fake must not be aclosed")

        fake.close = _close_fake  # type: ignore[method-assign]
        provider._client = fake

        async def _inner():
            await provider.aclose()
            return provider._get_client()

        got = asyncio.run(_inner())
        assert got is fake


class TestOpenAICompatibleProviderAclose:
    def test_aclose_closes_injected_httpx_client(self):
        """Basic-auth path constructs its own ``httpx.AsyncClient``;
        ``AsyncOpenAI.close()`` must aclose that injected client too."""
        provider = OpenAICompatibleProvider(
            base_url="http://localhost:8000/v1",
            auth_method="basic",
            auth_credentials=("user", "pass"),
        )
        captured: dict[str, FakeHttpxClient] = {}

        class _CapturingOpenAI(FakeAsyncOpenAI):
            def __init__(self, **kwargs: object) -> None:
                http_client = kwargs.get("http_client")
                if isinstance(http_client, FakeHttpxClient):
                    captured["http"] = http_client
                super().__init__(**kwargs)

        FakeAsyncOpenAI.instances = []
        with (
            patch("openai.AsyncOpenAI", _CapturingOpenAI),
            patch("httpx.AsyncClient", FakeHttpxClient),
        ):

            async def _inner():
                client = provider._get_client()
                await provider.aclose()
                return client

            client = asyncio.run(_inner())

        assert client.close_calls == 1
        assert captured["http"].aclose_calls == 1


class TestAcloseHttpClientHelper:
    async def test_aclose_http_client_awaits_close(self):
        fake = FakeAsyncOpenAI()
        await _aclose_http_client(fake)
        assert fake.close_calls == 1
        assert fake._client.aclose_calls == 1

    async def test_aclose_http_client_none_is_noop(self):
        await _aclose_http_client(None)

    async def test_aclose_http_client_falls_back_to_httpx(self):
        class _NoClose:
            def __init__(self) -> None:
                self._client = FakeHttpxClient()

        wrapper = _NoClose()
        await _aclose_http_client(wrapper)
        assert wrapper._client.aclose_calls == 1


# ── Registry ──────────────────────────────────────────────────────────


class TestProviderRegistryAclose:
    def test_aclose_forwards_to_instantiated_providers(self):
        reg = ProviderRegistry(auto_configure=False)
        mock = MockProvider()
        calls: list[str] = []

        async def _track() -> None:
            calls.append("aclose")

        mock.aclose = _track  # type: ignore[method-assign]
        reg.register_instance("mock", mock)

        async def _inner():
            await reg.aclose()

        asyncio.run(_inner())
        assert calls == ["aclose"]

    def test_aclose_skips_uninstantiated_factories(self):
        reg = ProviderRegistry(auto_configure=False)
        reg.register("mock", MockProvider)
        assert reg._providers == {}

        async def _inner():
            await reg.aclose()

        asyncio.run(_inner())
        assert reg._providers == {}

    def test_async_context_manager(self):
        reg = ProviderRegistry(auto_configure=False)
        mock = MockProvider()
        calls: list[str] = []

        async def _track() -> None:
            calls.append("aclose")

        mock.aclose = _track  # type: ignore[method-assign]
        reg.register_instance("mock", mock)

        async def _inner():
            async with reg:
                pass

        asyncio.run(_inner())
        assert calls == ["aclose"]

    def test_close_from_running_loop_raises(self):
        reg = ProviderRegistry(auto_configure=False)

        async def _inner():
            reg.close()

        with pytest.raises(RuntimeError, match="running event loop"):
            asyncio.run(_inner())

    def test_base_provider_aclose_is_noop(self):
        """MockProvider inherits the default no-op aclose — must not raise."""
        mock = MockProvider()

        async def _inner():
            await mock.aclose()

        asyncio.run(_inner())
        mock.close()
