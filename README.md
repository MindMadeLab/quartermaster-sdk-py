# Quartermaster

Open-source AI agent orchestration framework. Build, compose, and run multi-step AI agent workflows with any LLM provider.

Built by [MindMade](https://mindmade.si) in Slovenia.

## Packages

| Package | Description | Extractability | Status |
|---|---|---|---|
| [`qm-mcp-client`](./qm-mcp-client/) | MCP Protocol client (SSE + Streamable HTTP) | 9.5/10 | Tier 1 — Week 1-2 |
| [`qm-code-runner`](./qm-code-runner/) | Docker-based sandboxed code execution (Python, Node, Go, Rust, Deno, Bun) | 10/10 | Tier 1 — Week 1-2 |
| [`qm-providers`](./qm-providers/) | Multi-LLM provider abstraction (OpenAI, Anthropic, Google, Groq, xAI) | 8/10 | Tier 1 — Week 3-4 |
| [`qm-tools`](./qm-tools/) | Tool definition, registry, and execution framework | 8/10 | Tier 2 — Week 5-6 |
| [`qm-nodes`](./qm-nodes/) | 38+ composable node types (LLM, control flow, memory, user interaction) | 8/10 | Tier 2 — Week 5-6 |
| [`qm-graph`](./qm-graph/) | Agent graph schema and builder (DAG-based flow definitions) | 7/10 | Tier 2 — Month 2 |
| [`qm-engine`](./qm-engine/) | Flow execution engine (traversal, branching, memory, streaming) | 6/10 | Tier 3 — Month 2-3 |

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │            Your Application             │
                    └─────────────┬───────────────────────────┘
                                  │
                    ┌─────────────▼───────────────────────────┐
                    │            qm-engine                     │
                    │  (Flow execution, traversal, streaming)  │
                    └──┬──────────┬──────────────┬────────────┘
                       │          │              │
          ┌────────────▼──┐  ┌───▼──────┐  ┌───▼─────────┐
          │   qm-graph    │  │ qm-nodes │  │  qm-tools   │
          │  (Schema,     │  │ (38 node │  │ (Registry,  │
          │   builder)    │  │  types)  │  │  execution) │
          └───────────────┘  └────┬─────┘  └──────┬──────┘
                                  │               │
                    ┌─────────────▼───────────────▼────────────┐
                    │           qm-providers                    │
                    │  (OpenAI, Anthropic, Google, Groq, xAI)  │
                    └──────────────────────────────────────────┘

          ┌──────────────────┐    ┌──────────────────┐
          │  qm-mcp-client   │    │  qm-code-runner  │
          │  (Standalone)    │    │  (Standalone)     │
          └──────────────────┘    └──────────────────┘
```

## Quick Start

Each package can be used independently or together:

```bash
# Use just the MCP client
pip install quartermaster-mcp-client

# Use just the LLM provider abstraction  
pip install quartermaster-providers

# Use the full framework
pip install quartermaster
```

## Extraction Roadmap

### Tier 1 — Immediate (Week 1-4)
Standalone packages with zero/minimal coupling. Can be published to PyPI immediately.

- **qm-mcp-client** — Zero-dependency MCP client. Only needs httpx.
- **qm-code-runner** — Already a standalone FastAPI service. Just needs cleanup.
- **qm-providers** — Clean abstraction, needs Django decoupling.

### Tier 2 — Minor Refactoring (Week 5-8)
Well-structured code that needs some dependency replacement.

- **qm-tools** — Replace Django models with dataclasses, clean up registry.
- **qm-nodes** — Extract 38 node types, replace FlowContext with protocol.
- **qm-graph** — Convert Django ORM models to Pydantic, add builder API.

### Tier 3 — Significant Refactoring (Month 2-3)
Core engine that requires architectural changes for standalone use.

- **qm-engine** — Replace Django ORM + Celery with pluggable stores and dispatchers.

### Meta-Package (Month 3)
- **quartermaster** — Installs all packages together as unified framework.

## What Stays Proprietary

The following components remain part of the Quartermaster Platform (commercial product):

- Visual Agent Graph Editor (React/Electron frontend)
- Multi-tenant platform (user management, workspaces, billing)
- Real-time collaboration (WebSocket, Django Channels)
- Agent marketplace and distribution
- Monitoring, observability, analytics dashboard
- Enterprise SSO, RBAC, audit logging
- MindMade self-hosted LLM infrastructure
- Voice sessions and advanced TTS/STT
- Managed deployment and scaling
- EU AI Act compliance layer

## License

Apache 2.0

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) (coming soon)
