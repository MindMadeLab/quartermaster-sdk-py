"""v0.6.3 — ``InstructionFormExecutor`` parse-hardening.

Three gaps closed vs v0.6.2:

1. **Greedy ``\\{.*\\}`` fallback on the pre-strip raw text.** The old code
   only tried (a) strict ``json.loads`` on fence-stripped text, then
   (b) ``raw_decode`` walking every ``{``/``[`` position on the same
   fence-stripped text. If the fence regex (anchored to the start/end
   of the string) failed to strip because of leading prose, the
   downstream walk still ran — but in some Gemma-4 outputs the brace
   walk confuses a thinking-channel ``{`` with the real answer. A
   greedy-regex last-match pass over the **original raw text** (fences
   and prose untouched) is the third and final tier.

2. **Tool-call payload fallback when ``text_content`` is empty.** On
   the final ``instruction_form`` turn the model occasionally emits
   its entire answer inside a text-form ``<|tool_call|>`` marker
   block. The provider's v0.6.0 / v0.6.1 salvage strips those blocks
   into structured ``tool_calls`` and leaves ``text_content`` empty.
   The executor now checks ``tool_calls`` when text is blank and uses
   the first call's ``parameters`` dict as the JSON candidate.

3. **Diagnostic logging on parse failure.** Previously the failure
   path returned ``success=False`` with the raw text in ``data`` but
   emitted no log line, so ops had to dig into the flow record to see
   what the LLM actually produced. Now we log a WARNING with the first
   500 chars of the raw response, so production grep catches it.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from uuid import uuid4

import pytest
from quartermaster_providers import ProviderRegistry
from quartermaster_providers.testing import MockProvider
from quartermaster_providers.types import NativeResponse, ToolCall

from quartermaster_engine.context.execution_context import ExecutionContext
from quartermaster_engine.example_runner import (
    InstructionFormExecutor,
    _parse_json_progressive,
)
from quartermaster_engine.types import (
    GraphNode,
    GraphSpec,
    NodeType,
)

# ── Unit: ``_parse_json_progressive`` ────────────────────────────────


class TestParseJsonProgressive:
    """Three-tier parser: strict → walk → greedy-regex on raw text."""

    def test_strict_json_loads_happy_path(self) -> None:
        cleaned = '{"city": "Zagreb", "country": "HR"}'
        assert _parse_json_progressive(cleaned, cleaned) == {
            "city": "Zagreb",
            "country": "HR",
        }

    def test_walk_finds_embedded_json_after_preamble(self) -> None:
        """Preamble text before the JSON object — tier 2 brace walk picks
        it up because the first ``{`` is at a known position."""
        raw = 'Here is the answer:\n{"city": "Zagreb"}'
        assert _parse_json_progressive(raw, raw) == {"city": "Zagreb"}

    def test_walk_picks_the_widest_match_when_multiple_candidates(self) -> None:
        """When the text has multiple valid JSON spans, we want the
        outermost / widest one — a reasoning fragment ``{"think": "..."}``
        before the real answer ``{"city": "..."}`` must not win."""
        raw = '{"think": "hmm"} then {"city": "Zagreb", "country": "HR"}'
        out = _parse_json_progressive(raw, raw)
        # Widest by end-offset is the later one.
        assert out == {"city": "Zagreb", "country": "HR"}

    def test_greedy_regex_salvages_json_our_fence_strip_missed(self) -> None:
        """The fence-strip regex in the executor is anchored to the very
        start / end. If the LLM emits prose BEFORE the fence
        (``"Here you go:\\n```json\\n{...}\\n```"``) the fence is not
        stripped, ``json.loads`` fails on the prose, and the brace walk
        would tokenise inside the raw backticks. Tier 3's greedy regex
        on the raw text catches it."""
        raw = 'Sure! Here you go:\n```json\n{"city": "Zagreb"}\n```'
        # Simulate the executor's cleaned = fence-strip-anchored(raw).
        # The anchored regex fails to strip because of leading prose.
        cleaned = raw
        out = _parse_json_progressive(cleaned, raw)
        assert out == {"city": "Zagreb"}

    def test_thinking_channel_then_json(self) -> None:
        """Gemma-4's reasoning-channel prefix before the real answer."""
        raw = (
            "<|channel>thought\nlet me consider this company\n<channel|>\n"
            '{"city": "Zagreb", "industry": "construction"}'
        )
        out = _parse_json_progressive(raw, raw)
        assert out == {"city": "Zagreb", "industry": "construction"}

    def test_gemma_preamble_with_partial_brace_in_reasoning(self) -> None:
        """Thinking content that itself contains ``{`` in prose — must
        not be mistaken for the real JSON and must not short-circuit
        before reaching the actual answer."""
        raw = (
            'Looking at {this partial fragment}\nFinal answer:\n{"city": "Zagreb", "country": "HR"}'
        )
        out = _parse_json_progressive(raw, raw)
        # ``{this partial fragment}`` is not valid JSON → walk skips it
        # and picks up the real object next.
        assert out == {"city": "Zagreb", "country": "HR"}

    def test_returns_none_when_nothing_parses(self) -> None:
        assert _parse_json_progressive("not json at all", "not json at all") is None

    def test_empty_input_returns_none(self) -> None:
        assert _parse_json_progressive("", "") is None


# ── Integration: executor's tool_calls fallback + diagnostic logging ──


def _make_ctx(node_metadata: dict[str, Any] | None = None) -> ExecutionContext:
    node = GraphNode(
        id=uuid4(),
        type=NodeType.INSTRUCTION_FORM,
        name="ExtractCompany",
        metadata=node_metadata or {},
    )
    graph = GraphSpec(
        id=uuid4(),
        agent_id=uuid4(),
        start_node_id=node.id,
        nodes=[node],
        edges=[],
    )
    return ExecutionContext(
        flow_id=uuid4(),
        node_id=node.id,
        graph=graph,
        current_node=node,
        messages=[],
        memory={"__user_input__": "Enrich Makro Mikro"},
        metadata={},
    )


def _registry_with(mock: MockProvider) -> ProviderRegistry:
    reg = ProviderRegistry()
    reg.register_instance("mock", mock)
    # Tell the registry mock is the fallback for any model lookup. The
    # executor reads ``llm_provider`` from metadata and falls back on
    # ``_resolve_provider_and_model``.
    return reg


class TestToolCallPayloadFallback:
    """When the LLM puts the whole answer inside a ``<|tool_call|>``
    block and the provider-level salvage strips it into structured
    ``tool_calls``, ``text_content`` is empty. The executor used to
    return ``success=False`` because it only read ``text_content`` —
    now it falls back to the first tool call's ``parameters`` dict."""

    def test_empty_text_with_tool_call_salvages_to_parsed(self) -> None:
        mock = MockProvider(
            native_responses=[
                NativeResponse(
                    text_content="",
                    thinking=[],
                    tool_calls=[
                        ToolCall(
                            tool_name="SomeSyntheticName",
                            tool_id="call_text_0",
                            parameters={
                                "city": "Zagreb",
                                "country": "HR",
                                "industry": "construction",
                            },
                        )
                    ],
                    stop_reason="stop",
                )
            ]
        )
        ctx = _make_ctx(node_metadata={"llm_provider": "mock", "llm_model": "mock-m"})
        executor = InstructionFormExecutor(_registry_with(mock))

        result = asyncio.run(executor.execute(ctx))

        assert result.success, result.error
        assert result.data["parsed"] == {
            "city": "Zagreb",
            "country": "HR",
            "industry": "construction",
        }

    def test_empty_text_and_no_tool_calls_fails_with_diagnostic_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The empty/empty case is an actual failure — we want the
        diagnostic log line so ops can spot it in production."""
        mock = MockProvider(
            native_responses=[
                NativeResponse(
                    text_content="",
                    thinking=[],
                    tool_calls=[],
                    stop_reason="stop",
                )
            ]
        )
        ctx = _make_ctx(node_metadata={"llm_provider": "mock", "llm_model": "mock-m"})
        executor = InstructionFormExecutor(_registry_with(mock))

        with caplog.at_level(logging.WARNING, logger="quartermaster_engine.example_runner"):
            result = asyncio.run(executor.execute(ctx))

        assert not result.success
        assert "Could not parse JSON" in (result.error or "")
        assert any("could not parse json" in rec.getMessage().lower() for rec in caplog.records), (
            "Diagnostic warning must be logged on parse failure"
        )


class TestGreedyRegexFallbackEndToEnd:
    """The fence-with-preamble case that used to slip through the
    anchored fence strip + brace walk combo — now the greedy-regex
    tier catches it end-to-end through the executor."""

    def test_fenced_json_with_leading_prose_parses_via_regex_tier(self) -> None:
        raw = 'Here you go:\n```json\n{"city": "Zagreb"}\n```'
        mock = MockProvider(
            native_responses=[
                NativeResponse(
                    text_content=raw,
                    thinking=[],
                    tool_calls=[],
                    stop_reason="stop",
                )
            ]
        )
        ctx = _make_ctx(node_metadata={"llm_provider": "mock", "llm_model": "mock-m"})
        executor = InstructionFormExecutor(_registry_with(mock))

        result = asyncio.run(executor.execute(ctx))

        assert result.success, result.error
        assert result.data["parsed"] == {"city": "Zagreb"}


class TestDiagnosticLogging:
    """The WARNING log line on parse failure — production greppability."""

    def test_raw_text_included_in_log(self, caplog: pytest.LogCaptureFixture) -> None:
        mock = MockProvider(
            native_responses=[
                NativeResponse(
                    text_content="no json here, just prose",
                    thinking=[],
                    tool_calls=[],
                    stop_reason="stop",
                )
            ]
        )
        ctx = _make_ctx(node_metadata={"llm_provider": "mock", "llm_model": "mock-m"})
        executor = InstructionFormExecutor(_registry_with(mock))

        with caplog.at_level(logging.WARNING, logger="quartermaster_engine.example_runner"):
            asyncio.run(executor.execute(ctx))

        # The raw text is truncated to 500 chars in the log — the full
        # string we sent is well under 500 so it should appear verbatim.
        assert any("no json here" in rec.getMessage() for rec in caplog.records), (
            "Raw text must be included in the diagnostic log"
        )

    def test_log_truncates_at_500_chars(self, caplog: pytest.LogCaptureFixture) -> None:
        huge = "x" * 2000 + "{bogus}"
        mock = MockProvider(
            native_responses=[
                NativeResponse(
                    text_content=huge,
                    thinking=[],
                    tool_calls=[],
                    stop_reason="stop",
                )
            ]
        )
        ctx = _make_ctx(node_metadata={"llm_provider": "mock", "llm_model": "mock-m"})
        executor = InstructionFormExecutor(_registry_with(mock))

        with caplog.at_level(logging.WARNING, logger="quartermaster_engine.example_runner"):
            asyncio.run(executor.execute(ctx))

        # The log message must not contain the 2000-char filler in full.
        messages = " ".join(rec.getMessage() for rec in caplog.records)
        assert "x" * 1000 not in messages, (
            "Diagnostic log must cap the raw text at 500 chars to avoid "
            "flooding log aggregators on pathological responses."
        )
