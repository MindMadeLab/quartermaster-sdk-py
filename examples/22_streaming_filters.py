"""Example 22 -- Filtered stream iterators (v0.3.0).

One graph, three consumers. ``qm.run.stream(graph, "...")`` returns a
wrapper that is iterable (raw pass-through, for backwards compatibility)
and exposes four filter methods:

    .tokens()           -> yields str -- the model tokens, for typewriter UIs
    .tool_calls()       -> yields ToolCallChunk -- tool dispatch cards
    .progress()         -> yields ProgressChunk -- emit_progress signals
    .custom(name=...)   -> yields CustomChunk -- emit_custom milestones

The filters replace v0.2.x boilerplate like::

    for chunk in qm.run.stream(graph, "hi"):
        if chunk.type == "token":
            print(chunk.content, end="")

IMPORTANT single-pass semantics -- the wrapper owns the underlying
generator. First consumer drains it; a second consumer on the same
stream raises ``RuntimeError("stream already consumed")``. To render
the same graph three ways, build three independent streams (that's
what this example does -- each ``qm.run.stream(...)`` call below runs
the graph fresh).

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."
    uv run examples/22_streaming_filters.py
"""

from __future__ import annotations

import quartermaster_sdk as qm


# ---------------------------------------------------------------------------
# 1. Build a small agent graph
# ---------------------------------------------------------------------------

agent = (
    qm.Graph("Filter Demo")
    .user("Ask a short question")
    .instruction(
        "Respond",
        model="claude-haiku-4-5-20251001",
        provider="anthropic",
        system_instruction=(
            "You are a concise assistant. Answer in 1-2 sentences."
        ),
    )
)

user_input = "What is the tallest mountain in Slovenia?"


# ---------------------------------------------------------------------------
# 2. Way 1 -- typewriter effect (just tokens)
# ---------------------------------------------------------------------------
#
# ``.tokens()`` yields ``str`` (not the chunk wrapper) because
# concatenation is the overwhelming common case. Skip the
# ``.content`` hop the raw-iterator version has.

print("=" * 60)
print("  Way 1: .tokens() -- typewriter")
print("=" * 60)
for token in qm.run.stream(agent, user_input).tokens():
    print(token, end="", flush=True)
print("\n")


# ---------------------------------------------------------------------------
# 3. Way 2 -- dashboard view (tool calls only)
# ---------------------------------------------------------------------------
#
# The trivial instruction graph above doesn't actually call tools, so
# this loop yields zero entries -- but the pattern is what matters.
# In a real agent graph (see example 13 / 21) ``.tool_calls()`` surfaces
# every ``ToolCallChunk`` and skips everything else, giving a clean
# dashboard stream without the ``isinstance`` ladder.

print("=" * 60)
print("  Way 2: .tool_calls() -- dashboard view")
print("=" * 60)
call_count = 0
for call in qm.run.stream(agent, user_input).tool_calls():
    call_count += 1
    print(f"[TOOL] {call.tool}({call.args})")
if call_count == 0:
    print("  (no tool calls in this graph -- see example 21 for a tool-using demo)")
print()


# ---------------------------------------------------------------------------
# 4. Way 3 -- raw stream (debug view)
# ---------------------------------------------------------------------------
#
# Raw iteration still works unchanged -- the wrapper falls through to
# its underlying ``Iterator[Chunk]``. Useful for debugging, logging,
# or any consumer that genuinely needs every chunk type in one place.

print("=" * 60)
print("  Way 3: raw iteration -- debug view")
print("=" * 60)
for chunk in qm.run.stream(agent, user_input):
    # ``type(chunk).__name__`` gives ``TokenChunk`` / ``NodeStartChunk`` / ...
    summary = str(chunk)
    if len(summary) > 120:
        summary = summary[:117] + "..."
    print(f"  {type(chunk).__name__:18} {summary}")
print()


# ---------------------------------------------------------------------------
# 5. Note on single-pass semantics
# ---------------------------------------------------------------------------
#
# Trying to pull two filters off the SAME stream handle raises:
#
#     stream = qm.run.stream(agent, user_input)
#     tokens = list(stream.tokens())         # drains the stream
#     calls  = list(stream.tool_calls())     # RuntimeError: stream already consumed
#
# If you need multiple views of the same run, either:
#
#   (a) call ``qm.run.stream(...)`` multiple times (example above --
#       each run is independent), or
#   (b) iterate the stream once with raw ``for chunk in stream:`` and
#       route chunks to multiple consumers yourself.

print("All three views rendered the same graph independently.")
