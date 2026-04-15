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

__version__ = "0.3.0"

# ── v0.2.0 primary API ────────────────────────────────────────────────
#
# The recommended path is a four-import line for 90% of callsites:
#
#     from quartermaster_sdk import Graph, run, configure, instruction
#
# * ``Graph("x").user().agent().build()`` — auto-Start, optional End
# * ``run(graph, user_input)`` / ``run.stream(...)`` — no ``FlowRunner``
# * ``configure(provider="ollama", default_model=...)`` — boot once
# * ``instruction(system=..., user=...)`` — single-shot prompt → text
# * ``instruction_form(schema, system=..., user=...)`` — prompt → typed JSON
#
# Legacy v0.1.x exports (``FlowRunner``, ``build_default_registry``, …)
# remain available for integrators who need the low-level surface.
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
from quartermaster_engine.context.current_context import (  # noqa: F401
    current_context,
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

from ._chunks import (  # noqa: F401
    AwaitInputChunk,
    Chunk,
    CustomChunk,
    DoneChunk,
    ErrorChunk,
    NodeFinishChunk,
    NodeStartChunk,
    ProgressChunk,
    TokenChunk,
    ToolCallChunk,
    ToolResultChunk,
)
from ._config import (  # noqa: F401
    configure,
    get_default_model,
    get_default_registry,
    reset_config,
)
from ._async_runner import arun  # noqa: F401
from ._helpers import instruction, instruction_form  # noqa: F401
from ._result import Result  # noqa: F401
from ._runner import run  # noqa: F401
from ._trace import NodeTrace, Trace  # noqa: F401


__all__ = [
    # Version
    "__version__",
    # ── v0.2.0 primary surface ──
    "configure",
    "run",
    "arun",
    "instruction",
    "instruction_form",
    "Result",
    # Structured post-mortem trace — v0.3.0
    "Trace",
    "NodeTrace",
    # Typed streaming chunks
    "Chunk",
    "TokenChunk",
    "NodeStartChunk",
    "NodeFinishChunk",
    "ToolCallChunk",
    "ToolResultChunk",
    "AwaitInputChunk",
    "ProgressChunk",
    "CustomChunk",
    "DoneChunk",
    "ErrorChunk",
    # Context reach — tools call current_context() to emit progress
    "current_context",
    # Graph builder + spec
    "Graph",
    "GraphBuilder",
    "GraphSpec",
    "AgentGraph",  # deprecated
    "NodeType",
    # Engine — runner + node-registry surface (low-level)
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
    "get_default_registry",
    "get_default_model",
    "reset_config",
]
