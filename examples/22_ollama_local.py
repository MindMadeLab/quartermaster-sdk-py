"""Example 22 -- Local LLM with Ollama (Gemma 4).

Runs entirely on your machine — no cloud API keys needed.
Demonstrates using Ollama as a local provider with Google's Gemma 4 model.

Prerequisites:
    # Install Ollama: https://ollama.com
    ollama serve
    ollama pull gemma3:4b

Usage:
    uv run examples/22_ollama_local.py
"""

from __future__ import annotations

from quartermaster_graph import Graph
from quartermaster_providers import ProviderRegistry
from quartermaster_engine import FlowRunner, InMemoryStore
from quartermaster_engine.nodes import SimpleNodeRegistry, NodeExecutor, NodeResult
from quartermaster_engine.context.execution_context import ExecutionContext
from quartermaster_engine.events import FlowEvent, NodeStarted, NodeFinished, TokenGenerated, FlowError
from quartermaster_providers.config import LLMConfig
from quartermaster_graph.enums import NodeType


OLLAMA_MODEL = "gemma3:4b"


# -- Ollama LLM executor (talks to local Ollama server) --------------------

class OllamaExecutor(NodeExecutor):
    """Calls a local Ollama model with streaming."""

    def __init__(self, registry: ProviderRegistry, model: str):
        self._registry = registry
        self._model = model

    async def execute(self, context: ExecutionContext) -> NodeResult:
        system_instruction = context.get_meta("llm_system_instruction", "")
        prompt = str(context.memory.get("__user_input__", "Hello"))

        provider = self._registry.get("ollama")
        config = LLMConfig(
            model=self._model,
            provider="ollama",
            system_message=system_instruction,
            temperature=0.7,
            stream=True,
        )

        try:
            stream = await provider.generate_text_response(prompt, config)
            chunks = []
            async for token in stream:
                if token.content:
                    chunks.append(token.content)
                    context.emit_token(token.content)
            text = "".join(chunks)
            return NodeResult(success=True, data={}, output_text=text)
        except Exception as e:
            return NodeResult(success=False, data={}, error=str(e))


# -- Build graph ------------------------------------------------------------

agent = (
    Graph("Local Gemma Agent")
    .start()
    .user("Ask me anything")
    .instruction(
        "Respond",
        model=OLLAMA_MODEL,
        provider="ollama",
        system_instruction=(
            "You are a helpful local AI assistant powered by Gemma. "
            "Be concise and friendly. You run entirely on the user's machine."
        ),
    )
    .end()
)


# -- Set up Ollama provider and run ----------------------------------------

def main():
    graph = agent.build()

    # Register Ollama provider (no API key needed)
    registry = ProviderRegistry()
    registry.register_local("ollama")

    # Build node registry with our Ollama executor
    node_registry = SimpleNodeRegistry()
    executor = OllamaExecutor(registry, OLLAMA_MODEL)
    node_registry.register(NodeType.INSTRUCTION.value, executor)
    node_registry.register(NodeType.USER.value, _UserPassthrough())

    # Event handler with streaming
    def on_event(event: FlowEvent) -> None:
        if isinstance(event, NodeStarted):
            if event.node_type.value not in ("Start1", "End1", "User1"):
                print(f"\n  [{event.node_type.value:15s}] {event.node_name}", flush=True)
        elif isinstance(event, TokenGenerated):
            print(event.token, end="", flush=True)
        elif isinstance(event, NodeFinished):
            if event.node_type.value not in ("Start1", "End1", "User1"):
                print(flush=True)
        elif isinstance(event, FlowError):
            print(f"\n  [ERROR] {event.error}", flush=True)

    runner = FlowRunner(
        graph=graph,
        node_registry=node_registry,
        store=InMemoryStore(),
        on_event=on_event,
    )

    print(f"Ollama Local Agent ({OLLAMA_MODEL})")
    print("Make sure Ollama is running: ollama serve")
    print(f"Model required: ollama pull {OLLAMA_MODEL}")
    print()

    result = runner.run("Explain the difference between compiled and interpreted languages in 3 sentences.")

    print(f"\nDuration: {result.duration_seconds:.2f}s")
    if not result.success:
        print(f"Error: {result.error}")
        print("\nIs Ollama running? Try: ollama serve")


class _UserPassthrough(NodeExecutor):
    async def execute(self, context: ExecutionContext) -> NodeResult:
        return NodeResult(success=True, data={}, output_text=str(context.memory.get("__user_input__", "")))


if __name__ == "__main__":
    main()
