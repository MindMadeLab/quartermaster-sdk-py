"""Regression tests for the v0.7.0 sliding-window tool-log truncation.

Surface under test:

* ``_sliding_window_tool_log`` — pure helper that drops the OLDEST
  ``<tool_result>`` blocks until the prompt text fits inside the
  ``llm_max_input_tokens`` budget (approximated at ~4 chars/token).
* ``AgentExecutor`` integration — when the running tool-result log
  would push the next prompt past ``llm_max_input_tokens``, the
  executor trims the log in place and emits a
  ``agent.tool_log_truncated`` ``CustomEvent`` so UIs can surface
  "I had to drop N old scrape results".

Context: on long enrichment flows (15+ iterations, large scraped
pages) the accumulated ``<tool_result>`` blocks would push the
prompt past the model's context window and vLLM would return HTTP
400 ``maximum context length is 32768`` even though the node had
declared ``max_input_tokens=22000``. Pre-v0.7.0 the value was a
hint only; now it's enforced via a sliding-window drop.
"""

from __future__ import annotations

from typing import Any

from quartermaster_engine import FlowRunner
from quartermaster_engine.events import CustomEvent, FlowEvent
from quartermaster_engine.example_runner import _sliding_window_tool_log
from quartermaster_graph import Graph
from quartermaster_providers import ProviderRegistry
from quartermaster_providers.testing import MockProvider
from quartermaster_providers.types import NativeResponse, ToolCall


# ── Unit tests on _sliding_window_tool_log ───────────────────────────


class TestSlidingWindowHelper:
    """Direct unit tests — no graph run needed. Pins down the predicate
    so the integration test can trust the mechanism."""

    def test_max_input_tokens_none_returns_list_unchanged(self):
        """``max_input_tokens=None`` → truncation is disabled entirely.
        The helper returns the same list verbatim and ``dropped=0``.
        This is the pre-v0.7.0 compatibility contract: nodes that
        never set ``llm_max_input_tokens`` keep the old behaviour."""
        blocks = ["<tool_result>A</tool_result>", "<tool_result>B</tool_result>"]
        kept, dropped = _sliding_window_tool_log("base prompt", blocks, None)
        assert kept == blocks
        assert dropped == 0
        # Returns a COPY — the helper must not alias the caller's list
        # since the caller later does an in-place rebind.
        assert kept is not blocks

    def test_no_truncation_when_total_fits(self):
        """If ``len(base_prompt) + sum(blocks)+2`` already fits inside
        ``max_input_tokens*4``, no blocks are dropped."""
        base = "x" * 100
        blocks = ["a" * 50, "b" * 50]
        # budget = 200 tokens * 4 = 800 chars; total ~= 100 + 50+2 + 50+2 = 204
        kept, dropped = _sliding_window_tool_log(base, blocks, 200)
        assert kept == blocks
        assert dropped == 0

    def test_drops_oldest_blocks_when_over_budget(self):
        """FIFO drop: when over budget, the OLDEST block goes first.
        Repeats until the total fits."""
        base = "x" * 100
        # Each block ~1000 chars; budget = 100 tokens * 4 = 400 chars.
        # base alone is 100; one block of 1000 already exceeds 400.
        # But we keep at least the most recent block (single-block
        # bigger-than-budget contract) — so with 4 blocks we expect
        # 3 dropped, 1 kept.
        blocks = [f"block-{i}-" + ("z" * 1000) for i in range(4)]
        kept, dropped = _sliding_window_tool_log(base, blocks, 100)
        assert dropped == 3
        assert len(kept) == 1
        # The most recent block (index 3) is what survives.
        assert kept[0] == blocks[3]

    def test_never_drops_base_prompt(self):
        """The base prompt is counted toward the budget but never
        mutated or stripped — only ``<tool_result>`` blocks are
        eligible for dropping."""
        base = "SYSTEM + USER turn " * 50  # ~1000 chars
        blocks = ["tool-" + ("q" * 500) for _ in range(5)]
        kept, dropped = _sliding_window_tool_log(base, blocks, 200)
        # Some blocks must have been dropped (budget = 800 chars,
        # base alone is ~950), but the helper returns the kept block
        # list — it has no way to touch ``base`` and we assert that by
        # simply verifying the signature still returns the expected
        # shape without the base prompt appearing anywhere in ``kept``.
        assert dropped >= 1
        assert all("SYSTEM + USER" not in b for b in kept)

    def test_returns_tuple_of_kept_and_dropped_count(self):
        """Signature contract: ``(kept: list[str], dropped: int)``.
        Dropped + kept always equals the original block count."""
        base = "x" * 100
        blocks = ["b" * 500 for _ in range(6)]
        kept, dropped = _sliding_window_tool_log(base, blocks, 100)
        assert isinstance(kept, list)
        assert isinstance(dropped, int)
        assert len(kept) + dropped == len(blocks)

    def test_empty_list_in_empty_list_out(self):
        """Empty input → empty output, zero dropped. Defensive: the
        first agent iteration calls the helper before any tool has
        executed, so the list really is empty then."""
        kept, dropped = _sliding_window_tool_log("base", [], 1000)
        assert kept == []
        assert dropped == 0

    def test_fifo_drop_order_preserves_most_recent(self):
        """The most recent block (last in the list) is always the one
        the agent is about to reason about — it must never be dropped
        before older ones. FIFO drop order is the invariant."""
        base = "x" * 50
        blocks = [
            "OLD-" + ("a" * 400),
            "MID-" + ("b" * 400),
            "NEW-" + ("c" * 400),
        ]
        # budget = 100 tokens * 4 = 400 chars. Base is 50. One block
        # is ~404 chars — one block alone doesn't fit with the base,
        # so the helper should drop down to a single block (the
        # newest) per the "keep latest even if over budget" rule.
        kept, dropped = _sliding_window_tool_log(base, blocks, 100)
        assert dropped == 2
        assert len(kept) == 1
        assert kept[0].startswith("NEW-")

    def test_single_block_bigger_than_budget_is_kept(self):
        """When ONE block alone exceeds the budget, we keep it anyway.
        Rationale: the agent needs the latest tool result to make
        progress — truncation is best-effort, not an absolute limit.
        An agent that can't see its own most recent tool output is
        worse than one whose prompt is slightly over budget."""
        base = ""
        huge = "X" * 10_000  # 10 KB single block
        kept, dropped = _sliding_window_tool_log(base, [huge], 100)
        # Not dropped — the helper preserves the latest block even
        # when it alone exceeds the budget.
        assert kept == [huge]
        assert dropped == 0


# ── Integration: AgentExecutor with MockProvider ─────────────────────


class _BigPayloadTool:
    """Tool that returns a ~7 KB payload — enough to blow past a
    small ``max_input_tokens`` budget after 3 iterations.

    Mimics the real failure mode (large scraped web pages) the
    v0.7.0 truncation is guarding against. ``iteration_counter``
    lets each call produce a distinguishable payload so the test
    can identify which blocks survived.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._counter = 0

    def name(self) -> str:
        return "scrape"

    def safe_run(self, **kwargs: Any) -> Any:
        self._counter += 1
        self.calls.append(dict(kwargs))
        marker = f"CALL-{self._counter}"
        # ~7 KB blob tagged with the call index so the test can
        # assert which calls survived truncation. Size is chosen so
        # that 2 blocks fit inside the 20 KB budget the integration
        # test configures but 3 don't — the invariant under test is
        # "drop oldest 2, keep newest 2".
        payload = marker + ("x" * 7_000)

        class R:
            success = True
            data = {"marker": marker, "blob": payload}

        return R()


class _FakeRegistry:
    def __init__(self, tools: list[Any]) -> None:
        self._tools = {t.name(): t for t in tools}

    def get(self, name: str) -> Any:
        if name in self._tools:
            return self._tools[name]
        raise KeyError(name)

    def to_openai_tools(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": "",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for name in self._tools
        ]


def _build_mock_provider(batches: list[list[ToolCall]]) -> MockProvider:
    """Build a MockProvider: each batch drives one iteration; a final
    empty batch terminates the loop."""
    responses: list[NativeResponse] = []
    for batch in batches:
        responses.append(
            NativeResponse(
                text_content="",
                thinking=[],
                tool_calls=list(batch),
                stop_reason="tool_calls",
            )
        )
    responses.append(
        NativeResponse(
            text_content="Final answer.",
            thinking=[],
            tool_calls=[],
            stop_reason="stop",
        )
    )
    return MockProvider(native_responses=responses)


def _make_registry(provider: MockProvider) -> ProviderRegistry:
    reg = ProviderRegistry(auto_configure=False)
    reg.register_instance("mock", provider)
    reg.set_default_provider("mock")
    reg.set_default_model("mock", "test-model")
    return reg


class TestAgentExecutorTruncationIntegration:
    """End-to-end: agent loop with 4 synthetic tool calls, each
    returning a ~7 KB blob, and ``max_input_tokens=5000`` (= 20 KB
    budget). The executor must:

    * Keep the final prompt inside the budget.
    * Emit at least one ``agent.tool_log_truncated`` CustomEvent.
    * Drop the 2 OLDEST calls and keep the 2 newest.
    """

    def test_agent_executor_truncates_and_emits_event(self):
        scrape = _BigPayloadTool()
        registry = _FakeRegistry([scrape])

        # 4 iterations: each emits a single scrape call. Turn 5 is the
        # implicit final-answer turn the helper appends.
        mock = _build_mock_provider(
            [[ToolCall(tool_name="scrape", tool_id=f"c{i}", parameters={"n": i})] for i in range(4)]
        )
        provider_registry = _make_registry(mock)

        graph = (
            Graph("truncation")
            .start()
            .user()
            .agent(
                "research",
                tools=["scrape"],
                capture_as="research",
                max_input_tokens=5000,  # ~20 KB budget
                max_iterations=10,
            )
            .end()
            .build()
        )

        events: list[FlowEvent] = []
        runner = FlowRunner(
            graph=graph,
            provider_registry=provider_registry,
            tool_registry=registry,
            on_event=events.append,
        )
        result = runner.run("go")
        assert result.success, result.error

        # All 4 scrape calls ran — truncation doesn't skip tool
        # execution, it only prunes the LOG between iterations.
        assert len(scrape.calls) == 4

        # ── Final prompt fits inside the budget ─────────────────────
        # MockProvider records every generate_native_response prompt.
        # The LAST tool-iteration prompt is the one that carries the
        # fully-pruned tool log — the final (no-tools) turn is the
        # "Final answer" response which still sees the same pruned log.
        prompts = [c["prompt"] for c in mock.calls if c["method"] == "generate_native_response"]
        # Budget is max_input_tokens * 4 = 20 000 chars. Every recorded
        # prompt AFTER the first iteration must be under budget.
        # (The first iteration's prompt carries no tool_result blocks
        # so it's trivially under budget.)
        budget_chars = 5000 * 4
        for idx, p in enumerate(prompts[1:], start=1):
            assert len(p) <= budget_chars, (
                f"prompt {idx} was {len(p)} chars, expected <= {budget_chars}"
            )

        # ── CustomEvent(name="agent.tool_log_truncated") was emitted ─
        truncation_events = [
            e for e in events if isinstance(e, CustomEvent) and e.name == "agent.tool_log_truncated"
        ]
        assert truncation_events, (
            "expected at least one agent.tool_log_truncated CustomEvent, "
            f"got custom event names: {[e.name for e in events if isinstance(e, CustomEvent)]}"
        )

        # Payload shape — v0.7.0 contract.
        total_dropped = sum(e.payload["dropped"] for e in truncation_events)
        assert total_dropped >= 2, (
            f"expected at least 2 blocks dropped across truncation events, "
            f"got {total_dropped} in payloads {[e.payload for e in truncation_events]}"
        )
        last = truncation_events[-1]
        assert last.payload["max_input_tokens"] == 5000
        assert "iteration" in last.payload
        assert "kept" in last.payload

        # ── Oldest 2 calls dropped, newest 2 survive in the final prompt
        final_prompt = prompts[-1]
        # CALL-1 and CALL-2 are the oldest — they should be gone.
        # CALL-3 and CALL-4 are the most recent two — they should
        # still be visible to the model.
        assert "CALL-3" in final_prompt, "most recent scrape result was lost"
        assert "CALL-4" in final_prompt, "most recent scrape result was lost"
        assert "CALL-1" not in final_prompt, (
            "oldest scrape result should have been dropped by truncation"
        )
        assert "CALL-2" not in final_prompt, (
            "second-oldest scrape result should have been dropped by truncation"
        )
