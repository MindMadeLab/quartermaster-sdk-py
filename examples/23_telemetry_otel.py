"""Example 23 -- OpenTelemetry instrumentation (v0.3.0).

Demonstrates how to capture every Quartermaster flow event as an OTEL
span using GenAI semantic conventions. Uses an in-memory exporter so
the example runs without a backend -- in production you'd point this
at Jaeger, Tempo, Honeycomb, Logfire, Phoenix, or any OTLP collector.

Span model:
  * ``qm.flow``            -- root span, one per flow run
  * ``qm.node.<name>``     -- one span per node (``NodeStarted`` -> ``NodeFinished``)
  * ``qm.tool.<tool>``     -- one span per tool call, parented to the agent node
  * progress / custom      -- recorded as span events on the active node span

Attributes follow the OpenTelemetry GenAI semantic conventions:
``gen_ai.system``, ``gen_ai.operation.name``, ``gen_ai.tool.name``,
``gen_ai.usage.input_tokens``, ``gen_ai.usage.output_tokens``. See
https://opentelemetry.io/docs/specs/semconv/gen-ai/ .

Install:
    pip install 'quartermaster-sdk[telemetry]'

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."
    uv run examples/23_telemetry_otel.py
"""

from __future__ import annotations

import sys

# ---------------------------------------------------------------------------
# 0. Require the [telemetry] extra -- fail with a friendly message
# ---------------------------------------------------------------------------

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
except ImportError:
    print(
        "OpenTelemetry not installed. Run: pip install 'quartermaster-sdk[telemetry]'"
    )
    sys.exit(1)

import quartermaster_sdk as qm
from quartermaster_sdk import telemetry


# ---------------------------------------------------------------------------
# 1. Set up an in-memory span exporter
# ---------------------------------------------------------------------------
#
# Production exporters are typically OTLPSpanExporter talking to a
# Jaeger / Tempo / Honeycomb / Grafana backend. For an example we
# substitute an in-memory exporter so we can inspect spans inline
# below. Swap ``InMemorySpanExporter`` for ``OTLPSpanExporter`` in a
# real deployment.

exporter = InMemorySpanExporter()
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(provider)


# ---------------------------------------------------------------------------
# 2. One line -- instrument Quartermaster
# ---------------------------------------------------------------------------
#
# After this call, every subsequent ``qm.run(...)`` / ``qm.arun(...)`` /
# ``qm.run.stream(...)`` emits GenAI-conventioned OTEL spans via the
# tracer provider we just set up. No callsite changes needed.

telemetry.instrument()


# ---------------------------------------------------------------------------
# 3. Run a small graph
# ---------------------------------------------------------------------------

graph = (
    qm.Graph("Hello OTEL")
    .user("Say hello")
    .instruction(
        "Respond",
        model="claude-haiku-4-5-20251001",
        provider="anthropic",
        system_instruction="Reply with one short greeting.",
    )
)

result = qm.run(graph, "Say hi in Slovenian")
print("Model reply:", result.text)
print()


# ---------------------------------------------------------------------------
# 4. Inspect the spans
# ---------------------------------------------------------------------------
#
# Every flow, every node, every tool call is now a span. In
# production your exporter ships these to a backend; here we just
# print them.

spans = exporter.get_finished_spans()

print("=" * 60)
print(f"  Captured {len(spans)} OTEL spans")
print("=" * 60)

for span in spans:
    duration_ns = (span.end_time or 0) - (span.start_time or 0)
    duration_ms = duration_ns / 1_000_000
    attrs = dict(span.attributes or {})
    print(f"  {span.name:30s}  duration={duration_ms:7.2f} ms")
    for key, value in attrs.items():
        summary = str(value)
        if len(summary) > 80:
            summary = summary[:77] + "..."
        print(f"      {key:32s} = {summary}")
print()


# ---------------------------------------------------------------------------
# 5. Clean shutdown
# ---------------------------------------------------------------------------
#
# ``uninstrument()`` removes the listener so subsequent flows emit zero
# spans -- useful when toggling telemetry on/off across test suites or
# staged rollouts.

telemetry.uninstrument()

print(
    "Telemetry disabled. In production, swap InMemorySpanExporter for "
    "OTLPSpanExporter pointed at Jaeger / Tempo / Honeycomb / Logfire."
)
