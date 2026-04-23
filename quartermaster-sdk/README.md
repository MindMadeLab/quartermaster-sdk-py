# Quartermaster SDK

**Modular AI agent orchestration framework by [MindMade](https://mindmade.io).**

Quartermaster lets you build AI agent workflows as directed graphs — define nodes (LLM calls, decisions, user input, tools), connect them with edges, and execute them with a pluggable engine.

## What's new in v0.5.0

- **Simplified Ollama provider** -- `OllamaProvider` is now a thin subclass of the
  OpenAI-compatible client; the separate `OllamaNativeProvider`, the sync `chat()`
  shim, and the `ollama_tool_protocol` knob are gone. One transport for every
  local and cloud OpenAI-compatible endpoint.
- **Parallel tool execution** -- when a model emits multiple `tool_calls` in a
  single turn, the agent loop dispatches them concurrently; wall-clock drops
  from `O(sum(t))` to `O(max(t))`.
- **`program_runner(program=<callable>)`** -- pass a `@tool()`-decorated function
  directly instead of its name string; the graph builder auto-registers it.
  Parity with `.agent(tools=[...])`.
- **Universal tool-name prefix strip** -- `default_api:foo`, `default_api.foo`,
  `functions:foo`, `mcp:foo` all resolve via `rsplit` on `:` or `.`.
- **`duckduckgo_search` UA fix** -- realistic Chrome UA + `Accept`/`Referer`
  headers.

### What shipped in v0.4.0

- **Application timeouts** -- `qm.configure(timeout=, connect_timeout=, read_timeout=)` + per-call overrides.
- **Stream cancellation** -- `with qm.run.stream(...) as stream:` context-manager; `qm.Cancelled` + `ctx.cancelled`.
- **Per-node tool scoping** -- `agent(tools=[...])` strictly enforced; `tool_scope="permissive"` escape hatch.
- **Inline `@tool` callables** -- `agent(tools=[my_func])` accepts bare callables.
- **`instruction_form`** -- Gemma preamble robustness + dict-schema support.
- **`qm.configure(telemetry=True)`** -- sugar for `qm.telemetry.instrument()`.
- **`qm.configure(auto_redact_pii=True)`** -- automatic PII redaction policy.
- **`Trace.from_jsonl()` / `assert_traces_equal()`** -- round-trip trace serialisation for golden-file tests.
- **`SessionStore` protocol** -- `qm.run(graph, input, session=store, session_id=...)` + `InMemorySessionStore`.
- **`TypedEvent`** -- Pydantic base class for typed custom events.
- **`python -m quartermaster_sdk.lint check`** -- static graph linter (QM001--QM005).
- **`CircuitBreaker`** -- `CircuitBreaker(failure_threshold=, recovery_timeout=)` + `CircuitOpenError`.
- **Local-GPU cost tracker** -- `duration_seconds` + `local_gpu_cost_per_hour` support.

## Quick Install

```bash
# Core framework (graph + providers + tools + nodes + engine)
pip install quartermaster-sdk

# With OpenAI
pip install quartermaster-sdk[openai]

# With everything (all providers, all tools, MCP client, code runner)
pip install quartermaster-sdk[all]
```

## Quick Start (local Ollama, zero config)

```bash
ollama pull gemma4:26b      # or any model you've pulled
```

```python
import quartermaster_sdk as qm

qm.configure(
    provider="ollama",
    base_url="http://localhost:11434",   # or set $OLLAMA_HOST
    default_model="gemma4:26b",
)

# Graph() auto-creates Start; .end() / .build() are both optional when running via qm.run().
result = qm.run(qm.Graph("chat").user().agent(), "Pozdravljen, koliko je ura?")
print(result.text)
```

## Single-shot helpers (no graph visible)

```python
# prompt → str
reply = qm.instruction(system="Respond in Slovenian.", user="Pozdravljen!")

# prompt → Pydantic model (typed JSON extraction)
from pydantic import BaseModel

class Classification(BaseModel):
    category: str
    priority: str

data = qm.instruction_form(Classification, system="Classify.", user=email_body)
```

## Reading specific node outputs with `capture_as=`

```python
graph = (
    qm.Graph("enrich")
    .agent("Research", tools=[...], capture_as="notes")
    .instruction_form(CustomerData, system="Extract.", capture_as="data")
)
result = qm.run(graph, "VT-Treyd Slovenija")
result["notes"].output_text    # agent's free-text research
result["data"].output_text     # extracted JSON
```

## Streaming (v0.3.0 filtered iterators)

`qm.run.stream(...)` returns a wrapper you can iterate raw or pipe
through a filter — one helper per chunk family:

| Filter | Yields | Use for |
|---|---|---|
| `.tokens()` | `str` (the token text) | Typewriter UI — just the text |
| `.tool_calls()` | `ToolCallChunk` | Dashboard cards: `call.tool`, `call.args` |
| `.progress()` | `ProgressChunk` | `prog.message`, `prog.percent`, `prog.data` |
| `.custom(name=...)` | `CustomChunk` | Application-defined milestones |
| (raw `for chunk in ...`) | `Chunk` union | Debugging, pass-through consumers |

```python
# Typewriter effect -- tokens only.
for token in qm.run.stream(graph, "Tell me a story").tokens():
    print(token, end="", flush=True)

# Dashboard view -- just the tool calls.
for call in qm.run.stream(graph, "Research Slovenia").tool_calls():
    ui.tool_card(call.tool, call.args)

# Progress cards interleaved with model tokens.
for prog in qm.run.stream(graph, "Crunch the dataset").progress():
    ui.status(prog.message, prog.percent)

# Subscribe to one milestone name only.
for evt in qm.run.stream(graph, "Research").custom(name="source_found"):
    ui.add_source(evt.payload["url"])
```

Streams are **single-pass** — the wrapper owns its underlying generator,
so picking a second filter (or raw-iterating after a filter) raises
`RuntimeError("stream already consumed")`. Pick one consumer per stream.

The async analogue is available via `qm.arun.stream(...)` with the same
four filter helpers, returning `AsyncIterator[...]`.

## Post-mortem `Result.trace`

Every `Result` (sync or the terminal `DoneChunk.result` of a stream)
carries a structured `Trace` built from the full `FlowEvent` stream:

```python
result = qm.run(graph, "Hello!")

result.trace.text                        # concatenated model output
result.trace.tool_calls                  # list[dict] across every agent node
result.trace.progress                    # list[ProgressEvent]
result.trace.custom(name="source_found") # filtered CustomEvent list
result.trace.by_node["Researcher"].text  # tokens for a single node
print(result.trace.as_jsonl())           # JSONL export for logs / fixtures
```

## Progress events from inside tools

Long-running tools reach the flow's `ExecutionContext` via
`qm.current_context()` and emit structured events that stream back to
the UI alongside model tokens:

```python
from quartermaster_tools import tool

@tool()
def slow_research(topic: str) -> dict:
    ctx = qm.current_context()      # None when called outside a flow -- safe
    if ctx is not None:
        ctx.emit_progress("Gathering sources", percent=0.25, topic=topic)
        ctx.emit_custom("source_found", {"url": "https://example.com"})
    # ... do real work ...
    return {"summary": "..."}
```

Both sync and async tool bodies work. The context is carried through
`contextvars.copy_context()` into the agent loop's worker threads, so
`qm.current_context()` returns the right `ExecutionContext` even inside
tools that were dispatched via `.agent(tools=[...])` in parallel.

### SSE / Django example

```python
# views.py
from django.http import StreamingHttpResponse
import json
import quartermaster_sdk as qm

def enrich_sse(request):
    graph = build_enrichment_graph()

    def event_stream():
        with qm.run.stream(graph, request.GET["company"]) as stream:
            for chunk in stream.progress():
                payload = {
                    "message": chunk.message,
                    "percent": chunk.percent,
                    # ``tool_id`` lets the UI key concurrent cards correctly
                    # when the agent emits parallel tool_calls in one turn.
                    "tool_id": chunk.data.get("tool_id"),
                    **chunk.data,
                }
                yield f"event: progress\ndata: {json.dumps(payload)}\n\n"

    return StreamingHttpResponse(event_stream(), content_type="text/event-stream")
```

### Parallel tool-dispatch caveat

When the agent emits multiple `tool_calls` in a single model turn
(since v0.5.0), each tool runs on its own worker thread via
`asyncio.gather(asyncio.to_thread(...))`. Progress events from concurrent
tools are interleaved in arrival order — deterministic per-tool, not
globally. If your UI renders live cards per tool, key them by the
`tool_id` you pass into `emit_progress(..., data={"tool_id": ...})` so
two simultaneous `list_orders()` and `fetch_invoices()` calls don't
collapse into one card.

### Cancellation interaction

Tools can poll `ctx.cancelled` between progress emissions. If the SSE
client disconnects and the caller exits the `with qm.run.stream(...):`
block, `ctx.cancelled` flips True on the next poll:

```python
@tool()
def long_list_orders() -> list[dict]:
    ctx = qm.current_context()
    orders: list[dict] = []
    for i, row in enumerate(db.iter_orders()):
        if ctx and ctx.cancelled:
            raise qm.Cancelled()      # propagates as ErrorChunk(cancelled)
        if ctx and i % 25 == 0:
            ctx.emit_progress(f"Loaded {i} orders", percent=i / TOTAL)
        orders.append(row)
    return orders
```

## OpenTelemetry instrumentation

```bash
pip install 'quartermaster-sdk[telemetry]'
```

```python
from quartermaster_sdk import telemetry

telemetry.instrument()     # uses the global tracer provider
qm.run(graph, "Hello!")    # every node + tool call is now a span
```

Spans follow the OpenTelemetry GenAI semantic conventions
(`gen_ai.system`, `gen_ai.operation.name`, `gen_ai.tool.name`,
`gen_ai.usage.input_tokens`, …). Point your exporter at Jaeger, Tempo,
Honeycomb, Logfire, Phoenix, or any OTLP collector.

## Quick Start (cloud provider)

```python
agent = (
    qm.Graph("My Agent")
    .user("What can I help you with?")
    .instruction("Respond", model="gpt-4o", system_instruction="You are a helpful assistant.")
)
result = qm.run(agent, "How does photosynthesis work?")
```

## Packages

| Package | Description |
|---------|-------------|
| `quartermaster-graph` | Graph schema, builder API, validation |
| `quartermaster-providers` | LLM provider abstraction (OpenAI, Anthropic, Google, Groq, local) |
| `quartermaster-tools` | Tool definition, registry, built-in tools |
| `quartermaster-nodes` | Node execution protocols and implementations |
| `quartermaster-engine` | Flow execution, traversal, memory, streaming |
| `quartermaster-mcp-client` | MCP protocol client (standalone) |
| `quartermaster-code-runner` | Docker sandboxed code execution (standalone) |

## Documentation

See the [docs/](https://github.com/MindMadeLab/quartermaster-sdk-py/tree/master/docs) directory:

- [Getting Started](../docs/getting-started.md)
- [Graph Building](../docs/graph-building.md)
- [Architecture](../docs/architecture.md)
- [Tools Catalog](../docs/tools-catalog.md)
- [Providers](../docs/providers.md)
- [Security](../docs/security.md)

## License

Apache 2.0
