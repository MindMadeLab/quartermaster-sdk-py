"""v0.6.1 — Gemma-4 production emits asymmetric-pipe markers and
Python-call syntax, not the symmetric-JSON form PR #63 originally
covered.

Observed in production when vLLM is launched without
``--tool-call-parser gemma4``:

    <|tool_call>call:list_orders(status='odprto')<tool_call|>
    ^pipe-after-<                          ^pipe-before->

Both the marker asymmetry and the Python-call payload format are new
relative to PR #63. This file exercises both, plus the "multiple
concatenated calls in one response" pattern that produces Gemma-4's
retry-loop wedge.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from quartermaster_providers.config import LLMConfig
from quartermaster_providers.providers.openai import (
    OpenAIProvider,
    _coerce_python_call_payload,
    _coerce_text_tool_call_payload,
    _parse_text_form_tool_calls,
)


# ── Unit: the asymmetric marker pair is recognised ───────────────────


class TestAsymmetricMarkers:
    """The Gemma-4 production form ``<|tool_call>...<tool_call|>`` must
    match — the whole reason this patch exists."""

    def test_single_asymmetric_call_with_python_syntax(self) -> None:
        content = "<|tool_call>call:list_orders(status='active')<tool_call|>"
        calls, residual = _parse_text_form_tool_calls(content)
        assert len(calls) == 1
        assert calls[0].tool_name == "list_orders"
        assert calls[0].parameters == {"status": "active"}
        assert residual == ""  # markers fully stripped

    def test_symmetric_form_still_works(self) -> None:
        """Don't regress the symmetric form — PR #63's original case."""
        content = '<|tool_call|>{"name": "f", "arguments": {}}<|tool_call|>'
        calls, _ = _parse_text_form_tool_calls(content)
        assert len(calls) == 1
        assert calls[0].tool_name == "f"

    def test_asymmetric_takes_precedence_over_symmetric(self) -> None:
        """When a string only contains asymmetric markers, we must NOT
        accidentally match half of them as symmetric ``<|tool_call|>``
        blocks. The asymmetric regex is listed first for this reason."""
        content = "<|tool_call>call:a(x=1)<tool_call|><|tool_call>call:b(x=2)<tool_call|>"
        calls, residual = _parse_text_form_tool_calls(content)
        assert [c.tool_name for c in calls] == ["a", "b"]
        assert residual == ""


# ── Unit: the production reproducer (4 concatenated calls) ──────────


class TestProductionReproducer:
    """The exact string that triggered Gemma-4's retry loop on chat. All
    four calls must salvage; no single call lost to the next call's
    marker boundary."""

    PRODUCTION_STRING = (
        "<|tool_call>call:list_orders(status='odprto')<tool_call|>"
        "<|tool_call>call:list_orders(status='poprokan')<tool_call|>"
        "<|tool_call>call:list_orders(status='odprto')<tool_call|>"
        "<|tool_call>call:list_orders(status='poprokan')<tool_call|>"
    )

    def test_four_calls_extracted(self) -> None:
        calls, residual = _parse_text_form_tool_calls(self.PRODUCTION_STRING)
        assert len(calls) == 4
        assert [c.tool_name for c in calls] == ["list_orders"] * 4
        assert [c.parameters["status"] for c in calls] == [
            "odprto",
            "poprokan",
            "odprto",
            "poprokan",
        ]
        assert residual == "", (
            "Residual must be empty when every call was salvaged — "
            "otherwise the assistant's visible text still shows the "
            "literal <|tool_call> / <tool_call|> junk."
        )

    def test_four_calls_have_unique_tool_ids(self) -> None:
        """Agent dispatch loops key by ``tool_id`` — duplicates would
        collapse concurrent calls into one card and misorder results."""
        calls, _ = _parse_text_form_tool_calls(self.PRODUCTION_STRING)
        ids = [c.tool_id for c in calls]
        assert len(ids) == len(set(ids))

    def test_four_calls_preserve_order(self) -> None:
        """The status params alternate odprto/poprokan — we must preserve
        that order so the engine's ``tool_history`` reflects what the
        model actually requested."""
        calls, _ = _parse_text_form_tool_calls(self.PRODUCTION_STRING)
        seen_order = [c.parameters["status"] for c in calls]
        assert seen_order == ["odprto", "poprokan", "odprto", "poprokan"]


# ── Unit: Python-call-syntax payload parser ──────────────────────────


class TestPythonCallCoercer:
    def test_single_string_kwarg(self) -> None:
        assert _coerce_python_call_payload("call:list_orders(status='active')") == (
            "list_orders",
            {"status": "active"},
        )

    def test_without_call_prefix(self) -> None:
        """The ``call:`` prefix is optional — some servers emit it, some
        don't. Both must parse."""
        assert _coerce_python_call_payload("list_orders(status='active')") == (
            "list_orders",
            {"status": "active"},
        )

    def test_multiple_kwargs_mixed_types(self) -> None:
        name, args = _coerce_python_call_payload(
            "call:search(query='foo', limit=5, strict=True, page=None)"
        )
        assert name == "search"
        assert args == {"query": "foo", "limit": 5, "strict": True, "page": None}

    def test_double_quoted_strings(self) -> None:
        name, args = _coerce_python_call_payload('call:f(x="hello world")')
        assert name == "f"
        assert args == {"x": "hello world"}

    def test_list_value(self) -> None:
        name, args = _coerce_python_call_payload("call:f(ids=[1, 2, 3])")
        assert name == "f"
        assert args == {"ids": [1, 2, 3]}

    def test_nested_dict_value(self) -> None:
        name, args = _coerce_python_call_payload("call:f(cfg={'a': 1, 'b': [2, 3]})")
        assert name == "f"
        assert args == {"cfg": {"a": 1, "b": [2, 3]}}

    def test_no_args(self) -> None:
        assert _coerce_python_call_payload("call:ping()") == ("ping", {})

    def test_positional_args_preserved_under_sentinel(self) -> None:
        """Positional args are NOT promoted to kwargs (we don't know the
        tool's signature here). They land under ``__positional__`` so
        tool registries can log / warn about the unusable call."""
        name, args = _coerce_python_call_payload("call:f('a', 42)")
        assert name == "f"
        assert args == {"__positional__": ["a", 42]}

    def test_unquoted_bareword_stashed_as_source(self) -> None:
        """A bareword identifier isn't a literal — ast.literal_eval would
        reject it. We stash the source form so debugging is possible
        instead of dropping the kwarg silently."""
        name, args = _coerce_python_call_payload("call:f(mode=FAST)")
        assert name == "f"
        # The value stays as its source text so the tool registry sees
        # *something* it can match or log.
        assert args == {"mode": "FAST"}

    def test_malformed_returns_none(self) -> None:
        assert _coerce_python_call_payload("not a call") is None
        assert _coerce_python_call_payload("call:") is None
        assert _coerce_python_call_payload("call:f(unclosed") is None

    def test_empty_returns_none(self) -> None:
        assert _coerce_python_call_payload("") is None


# ── Unit: the JSON path still works ──────────────────────────────────


class TestJSONPathStillWorks:
    """Don't regress PR #63's JSON cases."""

    def test_name_arguments_shape(self) -> None:
        out = _coerce_text_tool_call_payload(
            '{"name": "list_orders", "arguments": {"status": "active"}}'
        )
        assert out == ("list_orders", {"status": "active"})

    def test_nested_function_shape(self) -> None:
        out = _coerce_text_tool_call_payload('{"function": {"name": "f", "arguments": {"x": 1}}}')
        assert out == ("f", {"x": 1})


# ── Unit: coercer dispatches JSON first, Python-call fallback ────────


class TestCoercerDispatch:
    def test_json_wins_when_valid(self) -> None:
        """A payload that parses as JSON goes through the JSON shape,
        not the Python-call regex."""
        assert _coerce_text_tool_call_payload('{"name": "f", "arguments": {"x": 1}}') == (
            "f",
            {"x": 1},
        )

    def test_falls_back_to_python_call(self) -> None:
        assert _coerce_text_tool_call_payload("call:f(x=1)") == ("f", {"x": 1})

    def test_both_unparsable_returns_none(self) -> None:
        assert _coerce_text_tool_call_payload("plain prose, no call here") is None


# ── Integration: native-response plumbing with the production string ──


def _fake_response(content: str, tool_calls=None) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = None
    return resp


def _provider_with(client_mock: MagicMock) -> OpenAIProvider:
    provider = OpenAIProvider(api_key="sk-test")
    provider._client = client_mock
    return provider


class TestEndToEndNativeResponse:
    def test_production_reproducer_through_native_response(self) -> None:
        """The full path: server emits the production string in
        ``message.content`` with empty structured ``tool_calls`` →
        ``generate_native_response`` returns a NativeResponse carrying
        four structured ToolCall objects + empty visible text."""
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=_fake_response(
                content=TestProductionReproducer.PRODUCTION_STRING,
                tool_calls=None,
            )
        )
        provider = _provider_with(client)

        config = LLMConfig(model="gemma-4-26b", provider="openai")
        resp = asyncio.run(provider.generate_native_response("hi", tools=None, config=config))

        assert len(resp.tool_calls) == 4
        assert all(c.tool_name == "list_orders" for c in resp.tool_calls)
        assert resp.text_content == ""
        # All four ToolCall ids unique so the agent loop can dispatch
        # them as four independent worker-thread tasks.
        ids = [c.tool_id for c in resp.tool_calls]
        assert len(set(ids)) == 4
