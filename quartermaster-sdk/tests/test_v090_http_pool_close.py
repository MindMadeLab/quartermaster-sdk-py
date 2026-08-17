"""v0.9.0: ``instruction()`` / ``qm.run()`` close async HTTP clients.

Regression for GitHub #96 / #97. The SDK's ``instruction()`` helper
runs a one-node graph through ``FlowRunner``, which uses ``asyncio.run()``.
After a successful call the provider registry must have been
``aclose()``d so openai/httpx pools do not leak
``RuntimeError: generator didn't stop after athrow()`` on loop shutdown.
"""

from __future__ import annotations

import openai  # noqa: F401 — eager import, mirrors other SDK tests

import pytest

import quartermaster_sdk as qm
from quartermaster_providers import ProviderRegistry
from quartermaster_providers.testing import MockProvider
from quartermaster_providers.types import NativeResponse, TokenResponse


def _mock_registry(
    text: str = "canned",
) -> tuple[ProviderRegistry, MockProvider, list[str]]:
    mock = MockProvider(
        responses=[TokenResponse(content=text, stop_reason="stop")],
        native_responses=[
            NativeResponse(
                text_content=text,
                thinking=[],
                tool_calls=[],
                stop_reason="stop",
            )
        ],
    )
    calls: list[str] = []
    original = mock.aclose

    async def _track() -> None:
        calls.append("aclose")
        await original()

    mock.aclose = _track  # type: ignore[method-assign]
    reg = ProviderRegistry(auto_configure=False)
    reg.register_instance("ollama", mock)
    reg.set_default_provider("ollama")
    reg.set_default_model("ollama", "mock-model")
    return reg, mock, calls


@pytest.fixture(autouse=True)
def _reset_config():
    qm.reset_config()
    yield
    qm.reset_config()


def test_instruction_aclose_provider_after_success():
    reg, _mock, calls = _mock_registry("hello from mock")
    qm.configure(registry=reg)

    text = qm.instruction(user="hi", provider_registry=reg)

    assert text == "hello from mock"
    assert "aclose" in calls, (
        "instruction() must aclose the provider while the asyncio loop "
        "is still running so httpx/httpcore2 pools drain cleanly (#97)"
    )


def test_run_aclose_provider_after_success():
    reg, _mock, calls = _mock_registry("graph reply")
    qm.configure(registry=reg)
    graph = qm.Graph("x").instruction("One").build()

    result = qm.run(graph, "hi", provider_registry=reg)

    assert result.success
    assert result.text == "graph reply"
    assert "aclose" in calls
