"""v0.9.0 — ``qm.instruction(..., extra_body=)`` / ``instruction_form``.

The Graph DSL already stashes ``extra_body`` as ``llm_extra_body`` on the
node (#98's gap is the convenience helpers). These tests pin that
``qm.instruction`` / ``qm.instruction_form`` accept the kwarg, forward it
onto the one-node graph (including the ``.vision()`` image path), and
that the engine lands it on ``LLMConfig.extra_body`` for the mock
provider.
"""

from __future__ import annotations

import openai  # noqa: F401 — eager import, matches test_v020_surface.py

import pytest
from pydantic import BaseModel

import quartermaster_sdk as qm
from quartermaster_providers import ProviderRegistry
from quartermaster_providers.testing import MockProvider
from quartermaster_providers.types import NativeResponse, TokenResponse


def _mock_registry(text: str = "canned") -> tuple[ProviderRegistry, MockProvider]:
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
    reg = ProviderRegistry(auto_configure=False)
    reg.register_instance("ollama", mock)
    reg.set_default_provider("ollama")
    reg.set_default_model("ollama", "mock-model")
    return reg, mock


@pytest.fixture(autouse=True)
def _reset_config():
    qm.reset_config()
    yield
    qm.reset_config()


_TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c63f8cfc0500f0000040001212c2e4e0000000049454e44ae426082"
)

_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}


def test_instruction_accepts_extra_body_without_typeerror() -> None:
    """The original bug: unexpected keyword ``extra_body``."""
    reg, _ = _mock_registry("ok")
    qm.configure(registry=reg)
    reply = qm.instruction(user="hi", extra_body=_EXTRA_BODY)
    assert reply == "ok"


def test_instruction_extra_body_lands_on_provider_config() -> None:
    """Helper → graph metadata → engine ``LLMConfig.extra_body``."""
    reg, mock = _mock_registry()
    qm.configure(registry=reg)
    qm.instruction(user="hi", extra_body=_EXTRA_BODY)
    assert mock.last_config is not None
    assert mock.last_config.extra_body == _EXTRA_BODY


def test_instruction_extra_body_dict_is_copied() -> None:
    """Post-call mutation of the caller's dict must not alias into config."""
    reg, mock = _mock_registry()
    qm.configure(registry=reg)
    payload = {"chat_template_kwargs": {"enable_thinking": False}}
    qm.instruction(user="hi", extra_body=payload)
    assert mock.last_config is not None
    assert mock.last_config.extra_body is not payload


def test_instruction_omits_extra_body_when_unused() -> None:
    reg, mock = _mock_registry()
    qm.configure(registry=reg)
    qm.instruction(user="hi")
    assert mock.last_config is not None
    assert mock.last_config.extra_body is None


def test_instruction_form_forwards_extra_body() -> None:
    class _Item(BaseModel):
        name: str

    canned = '{"name":"widget"}'
    reg, mock = _mock_registry(canned)
    qm.configure(registry=reg)
    out = qm.instruction_form(_Item, user="extract", extra_body=_EXTRA_BODY)
    assert out.name == "widget"
    assert mock.last_config is not None
    assert mock.last_config.extra_body == _EXTRA_BODY


def test_instruction_vision_path_forwards_extra_body() -> None:
    """``image=`` flips the helper onto ``.vision()``; extra_body must
    still land on the node / provider config."""
    reg, mock = _mock_registry("ocr")
    qm.configure(registry=reg)
    reply = qm.instruction(
        user="read the image",
        image=_TINY_PNG,
        extra_body=_EXTRA_BODY,
    )
    assert reply == "ocr"
    assert mock.last_config is not None
    assert mock.last_config.vision is True
    assert mock.last_config.extra_body == _EXTRA_BODY


def test_instruction_thinking_level_off_sets_thinking_enabled_false() -> None:
    """Default ``thinking_level="off"`` must reach the provider as
    ``thinking_enabled=False`` so OpenAI-compat (vLLM) can map it to
    ``chat_template_kwargs.enable_thinking=false``."""
    reg, mock = _mock_registry()
    qm.configure(registry=reg)
    qm.instruction(user="hi", thinking_level="off")
    assert mock.last_config is not None
    assert mock.last_config.thinking_enabled is False


def test_graph_instruction_thinking_level_off_sets_thinking_enabled_false() -> None:
    """Graph DSL path — same ``thinking_enabled=False`` landing as the
    convenience helper, so vLLM sees thinking off either way."""
    reg, mock = _mock_registry()
    qm.configure(registry=reg)
    graph = qm.Graph("g").instruction("Say", thinking_level="off")
    result = qm.run(graph, "hi")
    assert result.success, result.error
    assert mock.last_config is not None
    assert mock.last_config.thinking_enabled is False
