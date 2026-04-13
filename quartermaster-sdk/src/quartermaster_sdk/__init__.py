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

__version__ = "0.1.0"

# Re-export the most common entry points for convenience
from quartermaster_graph import AgentGraph, Graph, GraphBuilder  # noqa: F401
from quartermaster_graph.enums import NodeType  # noqa: F401
