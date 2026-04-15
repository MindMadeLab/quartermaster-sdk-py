"""The simplest possible agent: user asks, LLM responds.

Canonical v0.2.0 ergonomic demo — four imports, four lines, no
``.start()`` / ``.end()`` / ``.build()`` / ``FlowRunner`` boilerplate.

Usage:
    # Either set an API key...
    export ANTHROPIC_API_KEY="sk-ant-..."      # or OPENAI_API_KEY
    uv run examples/01_hello_agent.py

    # ...or run against a local Ollama:
    ollama serve && ollama pull gemma4:26b
    export OLLAMA_HOST=http://localhost:11434
    export QM_DEFAULT_MODEL=gemma4:26b
    uv run examples/01_hello_agent.py
"""

from __future__ import annotations

import quartermaster_sdk as qm

# Single-shot: no graph visible.  `qm.instruction(...)` builds a
# one-node graph internally, runs it, and returns the assistant text.
# Uses the default provider from `qm.configure(...)` or $OLLAMA_HOST /
# $QM_DEFAULT_MODEL — no boilerplate.
reply = qm.instruction(
    system="You are a helpful assistant. Be concise.",
    user="What is the capital of Slovenia?",
    model="claude-haiku-4-5-20251001",
)
print("Single-shot:", reply)

# Full graph path: same semantics as the cloud-auto-detecting v0.1.x
# `run_graph()`, but now without the `.start().end().build()` dance.
# `qm.run_graph()` finalises the builder internally and prints as it
# streams.
agent = (
    qm.Graph("Hello Agent")
    .user("Ask me anything")
    .instruction(
        "Respond",
        model="claude-haiku-4-5-20251001",
        system_instruction="You are a helpful assistant. Be concise.",
    )
)
qm.run_graph(agent, user_input="What is the capital of Slovenia?")
