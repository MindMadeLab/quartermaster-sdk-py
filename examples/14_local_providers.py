"""Example 14 -- Using local / self-hosted LLM providers.

Quartermaster works fully offline with your own infrastructure.  No
external API keys needed -- just point to your Ollama, vLLM, LM Studio,
TGI, LocalAI, or llama.cpp server.

This example shows three setups:

1. **Ollama only** -- simplest local setup, one line
2. **Mixed** -- cloud (Anthropic) for smart tasks + local (vLLM) for fast ones
3. **Fully private** -- all traffic stays on your infrastructure

Usage:
    # For Ollama graphs: ollama must be running locally (ollama serve)
    # For vLLM graphs: vLLM server must be running on your GPU box
    # For mixed: export ANTHROPIC_API_KEY="sk-ant-..."

    uv run examples/14_local_providers.py
"""

from quartermaster_graph import Graph
from quartermaster_providers import ProviderRegistry
from _runner import run_graph

# =====================================================================
# Setup 1: Ollama only  (one line!)
# =====================================================================

registry = ProviderRegistry(auto_configure=False)
registry.register_local("ollama")

# Route common open-source model names to Ollama
registry.add_model_pattern(r"llama3.*", "ollama")
registry.add_model_pattern(r"mistral.*", "ollama")
registry.add_model_pattern(r"phi-.*", "ollama")
registry.add_model_pattern(r"qwen.*", "ollama")
registry.add_model_pattern(r"deepseek.*", "ollama")
registry.add_model_pattern(r"codellama.*", "ollama")
registry.add_model_pattern(r"gemma.*", "ollama")

# Now model references resolve automatically:
#   registry.get_for_model("llama3.1:70b")   -> OllamaProvider
#   registry.get_for_model("phi-3-mini")     -> OllamaProvider

graph_ollama = (
    Graph("LocalAssistant")
    .start()
    .user("Ask me anything")
    .agent(
        "Assistant",
        model="llama3.1:70b",       # Ollama will serve this
        provider="ollama",           # explicit provider name
        system_instruction="You are a helpful local assistant.",
        tools=["web_search", "calculator"],
        max_iterations=10,
    )
    .end()
)

# =====================================================================
# Setup 2: Mixed -- cloud for smart, local for fast
# =====================================================================

mixed_registry = ProviderRegistry(auto_configure=False)

# Cloud provider for complex reasoning
from quartermaster_providers.providers.anthropic import AnthropicProvider
import os
mixed_registry.register("anthropic", AnthropicProvider, api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

# Local vLLM on a GPU box for fast inference
mixed_registry.register_local(
    "vllm",
    base_url="http://gpu-box:8000/v1",
    models=[r"llama3.*", r"mistral.*"],
)

graph_mixed = (
    Graph("HybridPipeline")
    .start()
    .user("Describe the problem")
    # Decision node -- LLM picks which path to take
    .decision("Classify")
    .on("simple")
        .instruction(
            "Quick Answer",
            model="llama3.1:70b",   # bigger local model
            provider="vllm",
            system_instruction="Give a direct answer.",
        )
    .end()
    .on("complex")
        .agent(
            "Deep Analysis",
            model="claude-haiku-4-5-20251001",
            provider="anthropic",
            system_instruction="Perform thorough analysis.",
            tools=["web_search", "code_interpreter"],
            max_iterations=15,
        )
    .end()
    # No merge -- decision picks one branch.
    .end()
)

# =====================================================================
# Setup 3: Fully private -- no external calls
# =====================================================================

private_registry = ProviderRegistry(auto_configure=False)

# Main inference via vLLM cluster
private_registry.register_local(
    "vllm",
    base_url="http://vllm-cluster.internal:8000/v1",
    api_key="internal-key",
    default=True,  # catch-all for any model name
)

# Embedding / lightweight tasks via local Ollama
private_registry.register_local(
    "ollama",
    base_url="http://ollama.internal:11434/v1",
    models=[r"nomic-embed.*", r"all-minilm.*"],
)

# Completely custom internal endpoint
private_registry.register_local(
    "custom",
    base_url="https://llm-gateway.corp.internal/v1",
    name="corp-gateway",
    api_key="corp-secret",
    models=[r"corp-.*"],
)

graph_private = (
    Graph("PrivateAgent")
    .allowed_agents("researcher", "writer")
    .start()
    .user("What should we build?")
    .agent(
        "Planner",
        model="llama3.1:70b",           # resolves to vllm (default)
        provider="vllm",
        system_instruction=(
            "Plan the project.  Spawn researcher and writer agents."
        ),
        tools=["spawn_agent", "collect_agent_results"],
        max_iterations=10,
    )
    .instruction(
        "Summary",
        model="corp-custom-model-v2",    # resolves to corp-gateway
        provider="corp-gateway",
        system_instruction="Synthesise the final plan.",
    )
    .end()
)


# -- Execute graphs -----------------------------------------------------------
# NOTE: Ollama must be running locally for graph_ollama to work.
# NOTE: vLLM must be running on gpu-box:8000 for graph_mixed/graph_private.
# We run only graph_ollama here as it's the most common local setup.
# Comment/uncomment as needed for your infrastructure.

if __name__ == "__main__":
    print("This example demonstrates local provider configuration.")
    print(f"Graphs built: ollama={len(graph_ollama.nodes)} nodes, mixed={len(graph_mixed.nodes)} nodes, private={len(graph_private.nodes)} nodes")
    print()
    print("To run with Ollama:  ollama serve && ollama pull llama3.1:70b")
    print("Then uncomment the run_graph() call below.")
    print()
    # run_graph(graph_ollama, user_input="Explain the difference between TCP and UDP", provider="ollama")

    # Uncomment to run the mixed graph (requires ANTHROPIC_API_KEY + vLLM):
    # run_graph(graph_mixed, user_input="Why is the sky blue?")

    # Uncomment to run the fully-private graph (requires vLLM + Ollama + corp-gateway):
    # run_graph(graph_private, user_input="Design a real-time data pipeline for IoT sensor data")
