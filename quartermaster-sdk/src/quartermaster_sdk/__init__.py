"""Quartermaster SDK — modular AI agent orchestration framework.

Install with: pip install quartermaster-sdk

This meta-package installs the five core Quartermaster packages:

- quartermaster-graph     — Graph schema, builder API, validation
- quartermaster-providers — LLM provider abstraction and registry
- quartermaster-tools     — Tool definition, registry, JSON schema export
- quartermaster-nodes     — Node execution protocols and type contracts
- quartermaster-engine    — Flow execution, traversal, memory, streaming

Optional extras:
- pip install quartermaster-sdk[openai]        — OpenAI provider SDK
- pip install quartermaster-sdk[anthropic]     — Anthropic provider SDK
- pip install quartermaster-sdk[mcp]           — MCP protocol client
- pip install quartermaster-sdk[code-runner]   — Docker sandboxed code execution
- pip install quartermaster-sdk[all]           — Everything
"""

__version__ = "0.2.0"

# Re-export the most common entry points for convenience.  Pre-0.1.4 the SDK
# re-exported only the graph-builder surface, forcing downstream callers to
# `from quartermaster_engine import FlowRunner` and `from quartermaster_providers
# import register_local` separately even though the SDK is the recommended
# install.  These re-exports cover the v0.1.4 release-notes snippet so a single
# `from quartermaster_sdk import …` line is enough to wire up the simplest graph.
from quartermaster_engine import (  # noqa: F401
    AgentExecutor,
    FlowResult,
    FlowRunner,
    LLMExecutor,
    NodeExecutor,
    NodeRegistry,
    NodeResult,
    SimpleNodeRegistry,
    build_default_registry,
    run_graph,
)
from quartermaster_graph import (  # noqa: F401
    AgentGraph,  # deprecated alias for GraphSpec — kept for backward compat
    Graph,
    GraphBuilder,
    GraphSpec,
)
from quartermaster_graph.enums import NodeType  # noqa: F401
from quartermaster_providers import (  # noqa: F401
    ChatResult,
    LLMConfig,
    ProviderRegistry,
    register_local,
)


__all__ = [
    # Version
    "__version__",
    # Graph builder + spec
    "Graph",
    "GraphBuilder",
    "GraphSpec",
    "AgentGraph",  # deprecated
    "NodeType",
    # Engine — runner + node-registry surface
    "FlowRunner",
    "FlowResult",
    "NodeRegistry",
    "NodeExecutor",
    "NodeResult",
    "SimpleNodeRegistry",
    "LLMExecutor",
    "AgentExecutor",
    "build_default_registry",
    "run_graph",
    # Providers — config, sync chat result, registry helpers
    "LLMConfig",
    "ChatResult",
    "ProviderRegistry",
    "register_local",
]
