"""Example 14 — Using local / self-hosted LLM providers.

Quartermaster works fully offline with your own infrastructure.  No
external API keys needed — just point to your Ollama, vLLM, LM Studio,
TGI, LocalAI, or llama.cpp server.

This example shows three setups:

1. **Ollama only** — simplest local setup, one line
2. **Mixed** — cloud (OpenAI) for smart tasks + local (vLLM) for fast ones
3. **Fully private** — all traffic stays on your infrastructure
"""

from quartermaster_graph import Graph
from quartermaster_providers import ProviderRegistry

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
#   registry.get_for_model("llama3.1:70b")   → OllamaProvider
#   registry.get_for_model("phi-3-mini")     → OllamaProvider

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
# Setup 2: Mixed — cloud for smart, local for fast
# =====================================================================

mixed_registry = ProviderRegistry(auto_configure=False)

# Cloud provider for complex reasoning
from quartermaster_providers.providers import OpenAIProvider
mixed_registry.register("openai", OpenAIProvider, api_key="sk-...")

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
    # Decision node — LLM picks which path to take
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
            model="gpt-4o",          # cloud for heavy reasoning
            provider="openai",
            system_instruction="Perform thorough analysis.",
            tools=["web_search", "code_interpreter"],
            max_iterations=15,
        )
    .end()
    # No merge — decision picks one branch.
    .end()
)

# =====================================================================
# Setup 3: Fully private — no external calls
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


# ── Print graph info ─────────────────────────────────────────────────
if __name__ == "__main__":
    for name, g, reg in [
        ("Ollama-only", graph_ollama, registry),
        ("Mixed", graph_mixed, mixed_registry),
        ("Fully-private", graph_private, private_registry),
    ]:
        print(f"\n{'='*60}")
        print(f" {name}")
        print(f"{'='*60}")
        print(f"  Nodes: {len(g.nodes)}")
        print(f"  Edges: {len(g.edges)}")
        print(f"  Providers: {reg.list_providers()}")
