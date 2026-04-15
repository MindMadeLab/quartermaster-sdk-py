"""Example 17 -- FlowRunner with event streaming.

Demonstrates using FlowRunner directly (not run_graph()) with custom event
handling, showing the production-grade API. Uses a mock executor so no API
keys are needed.

Event types shown:
  - NodeStarted   -- a node begins execution
  - TokenGenerated -- a streaming token is emitted
  - NodeFinished  -- a node completes
  - FlowFinished  -- the entire flow completes

Usage:
    uv run examples/17_streaming_events.py
"""

from __future__ import annotations

import asyncio
import time

from quartermaster_engine import (
    ExecutionContext,
    FlowEvent,
    FlowFinished,
    FlowRunner,
    InMemoryStore,
    NodeFinished,
    NodeResult,
    NodeStarted,
    SimpleNodeRegistry,
    TokenGenerated,
)
from quartermaster_graph import Graph
from quartermaster_graph.enums import NodeType


# ---------------------------------------------------------------------------
# 1. Mock executor -- no LLM needed
# ---------------------------------------------------------------------------


class MockLLMExecutor:
    """A mock node executor that simulates streaming token output."""

    def __init__(
        self, response: str = "This is a mock response showing event streaming."
    ) -> None:
        self._response = response

    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Emit tokens word-by-word, then return the full response."""
        for word in self._response.split():
            context.emit_token(word + " ")
            await asyncio.sleep(0.05)  # simulate network latency
        return NodeResult(success=True, data={}, output_text=self._response)


class PassthroughExecutor:
    """Passes input through unchanged -- used for User and control nodes."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        text = context.messages[-1].content if context.messages else ""
        return NodeResult(success=True, data={}, output_text=text)


# ---------------------------------------------------------------------------
# 2. Build a simple 3-node graph
# ---------------------------------------------------------------------------

graph = (
    Graph("Event Streaming Demo")
    .start()
    .user("Say something")
    .instruction(
        "Respond",
        model="mock",
        system_instruction="You are a helpful assistant.",
    )
    .end()
)

agent_graph = graph.build()

# ---------------------------------------------------------------------------
# 3. Set up node registry with mock executors
# ---------------------------------------------------------------------------

registry = SimpleNodeRegistry()

passthrough = PassthroughExecutor()
mock_llm = MockLLMExecutor()

# Register executors for each node type the graph uses.  Use NodeType
# enum values rather than magic strings — the actual enum values are
# "User1" / "Instruction1" / etc. (the magic-string version below used
# to ship "UserInput" / "Start" / "End" which silently never matched,
# producing a "No executor registered for node type: User1" error at
# runtime).  Start/End are handled by the runner internally so we don't
# strictly need to register them, but doing so is harmless.
registry.register(NodeType.START.value, passthrough)
registry.register(NodeType.END.value, passthrough)
registry.register(NodeType.USER.value, passthrough)
registry.register(NodeType.INSTRUCTION.value, mock_llm)

# ---------------------------------------------------------------------------
# 4. Custom event handler with timestamps
# ---------------------------------------------------------------------------

_flow_start = time.monotonic()
_token_buffer: list[str] = []


def on_event(event: FlowEvent) -> None:
    """Print each event with a relative timestamp."""
    elapsed = time.monotonic() - _flow_start
    ts = f"[{elapsed:6.3f}s]"

    if isinstance(event, NodeStarted):
        print(f"{ts}  NODE STARTED   | {event.node_name} ({event.node_type.value})")
    elif isinstance(event, TokenGenerated):
        _token_buffer.append(event.token)
        # Print tokens inline, flush after each one
        print(f"{ts}  TOKEN          | {event.token.strip()}", flush=True)
    elif isinstance(event, NodeFinished):
        output = event.result[:60] + "..." if len(event.result) > 60 else event.result
        print(f"{ts}  NODE FINISHED  | output={output!r}")
    elif isinstance(event, FlowFinished):
        output = (
            event.final_output[:60] + "..."
            if len(event.final_output) > 60
            else event.final_output
        )
        print(f"{ts}  FLOW FINISHED  | final={output!r}")


# ---------------------------------------------------------------------------
# 5. Run with FlowRunner directly
# ---------------------------------------------------------------------------

print("=" * 60)
print("  FlowRunner -- Event Streaming Demo")
print("=" * 60)
print()

_flow_start = time.monotonic()

runner = FlowRunner(
    graph=agent_graph,
    node_registry=registry,
    store=InMemoryStore(),
    on_event=on_event,
)

result = runner.run("Hello, show me how event streaming works!")

# ---------------------------------------------------------------------------
# 6. Print the FlowResult summary
# ---------------------------------------------------------------------------

print()
print("-" * 60)
print("FlowResult summary:")
print(f"  success:      {result.success}")
print(f"  final_output: {result.final_output!r}")
print(f"  duration:     {result.duration_seconds:.3f}s")
print(f"  node_results: {len(result.node_results)} nodes executed")
print(f"  tokens seen:  {len(_token_buffer)}")

if result.error:
    print(f"  error:        {result.error}")

print()
print("Streamed tokens reassembled:")
print(f"  {''.join(_token_buffer).strip()}")
