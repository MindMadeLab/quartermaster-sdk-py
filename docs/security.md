# Security Considerations

This document covers security features built into Quartermaster and best practices for production deployments.

## Code Runner Sandboxing

The `qm-code-runner` package executes user-provided code inside Docker containers with multiple layers of isolation.

### Container Isolation

Every code execution request runs in a fresh, ephemeral Docker container that is destroyed after completion. The container configuration supports:

| Control | Configuration | Description |
|---------|--------------|-------------|
| **Memory limit** | `mem_limit` | Maximum memory the container can use (e.g., `"256m"`) |
| **CPU shares** | `cpu_shares` | Relative CPU weight for the container |
| **Disk limit** | `disk_limit` | Maximum disk space available to the container |
| **Network** | `network_disabled` | Disable network access to prevent data exfiltration |
| **Timeout** | `timeout_seconds` | Maximum execution time before the container is killed |

### Supported Runtimes

The code runner supports multiple language runtimes, each using a purpose-built Docker image:

- Python
- Node.js
- Go
- Rust
- Deno
- Bun

### API Authentication

The code runner exposes a FastAPI HTTP API that supports two authentication modes:

```python
from qm_code_runner.security import configure_auth

# API key authentication (X-API-Key header)
configure_auth(api_keys=["key-1", "key-2"])

# Bearer token authentication (Authorization header)
configure_auth(auth_token="secret-token")

# Both modes simultaneously
configure_auth(api_keys=["key-1"], auth_token="secret-token")
```

When no keys or tokens are configured, authentication is disabled. In production, always configure at least one authentication method.

## API Key Management

### Environment Variables

Never hardcode API keys in source code. Use environment variables:

```python
import os
from qm_providers import ProviderRegistry
from qm_providers.providers.openai import OpenAIProvider

registry = ProviderRegistry()
registry.register("openai", OpenAIProvider, api_key=os.environ["OPENAI_API_KEY"])
```

### Key Rotation

The `ProviderRegistry` supports re-registration, which replaces the existing provider configuration:

```python
# Initial registration
registry.register("openai", OpenAIProvider, api_key=old_key)

# Rotate key -- re-registering clears the cached instance
registry.register("openai", OpenAIProvider, api_key=new_key)
```

### Secrets in Graph Metadata

Avoid storing API keys or secrets in `GraphNode.metadata`. Graph definitions may be serialized, logged, or stored in databases. Instead, resolve secrets at runtime from environment variables or a secrets manager.

## Tool Execution Security

### Parameter Validation

Always use `safe_run()` instead of `run()` when executing tools with untrusted input. `safe_run()` validates all parameters before execution:

```python
tool = registry.get("my_tool")

# Validates parameters first, returns error result on failure
result = tool.safe_run(**user_provided_params)
if not result:
    print(f"Validation failed: {result.error}")
```

### Custom Validation

Add custom validation functions to `ToolParameter` for domain-specific checks:

```python
from qm_tools.types import ToolParameter

def validate_path(value):
    import os
    if not os.path.isabs(value):
        raise ValueError("Path must be absolute")
    if ".." in value:
        raise ValueError("Path traversal not allowed")

param = ToolParameter(
    name="file_path",
    description="Path to read",
    type="string",
    required=True,
    validation=validate_path,
)
```

### Local Tool Timeouts

`AbstractLocalTool` enforces a configurable timeout on subprocess execution (default: 30 seconds). Override `timeout()` to adjust:

```python
class MyTool(AbstractLocalTool):
    def timeout(self) -> int:
        return 10  # Kill after 10 seconds
```

Timed-out processes return a `ToolResult` with `success=False`.

## Node Error Handling

### Preventing Runaway Retries

When using `ErrorStrategy.RETRY`, always set a reasonable `max_retries` limit:

```python
from qm_graph.models import GraphNode
from qm_graph.enums import NodeType, ErrorStrategy

node = GraphNode(
    type=NodeType.INSTRUCTION,
    name="API Call",
    error_handling=ErrorStrategy.RETRY,
    max_retries=3,       # Do not set this too high
    retry_delay=2.0,     # Back-off between retries
    timeout=30.0,        # Per-attempt timeout
)
```

### Timeout Enforcement

Node-level timeouts prevent a single node from blocking the entire flow indefinitely. Set `timeout` on nodes that call external services:

```python
node = GraphNode(
    type=NodeType.TOOL,
    name="External API",
    timeout=15.0,
    error_handling=ErrorStrategy.SKIP,  # Continue flow even if this times out
)
```

## Graph Validation

The `validate_graph()` function in `qm-graph` catches structural issues that could cause runtime problems:

- **Cycles** -- Unintended cycles could cause infinite execution loops. The validator detects cycles and raises an error (cycles involving `Loop` nodes produce a warning instead).
- **Orphan nodes** -- Unreachable nodes are flagged, which may indicate a broken graph.
- **Missing edge labels** -- Decision nodes without labeled edges could route incorrectly.

Always validate graphs before deploying them to production:

```python
from qm_graph.validation import validate_graph

errors = validate_graph(agent_version)
for error in errors:
    if error.severity == "error":
        raise RuntimeError(f"Graph validation failed: {error.message}")
    else:
        print(f"Warning: {error.message}")
```

## Execution Store Security

### In-Memory Store

The `InMemoryStore` holds all execution state in process memory. This is safe from external access but means:
- State is lost on process restart
- No audit trail
- No cross-process visibility

### SQLite Store

The `SQLiteStore` writes to a local file. Protect this file:
- Set appropriate file permissions (e.g., `chmod 600`)
- Store in a directory not accessible via web server
- Consider encryption at rest for sensitive data

### Custom Stores

When implementing stores backed by databases or caches (Redis, PostgreSQL):
- Use authenticated connections
- Encrypt data in transit (TLS)
- Consider encrypting sensitive flow memory values at rest
- Implement TTL/cleanup for completed flows to avoid unbounded data growth

## Provider Security

### Rate Limiting

LLM providers enforce rate limits. Quartermaster does not implement client-side rate limiting by default. For production deployments:
- Monitor `TokenUsage` from responses
- Implement rate limiting in a custom provider wrapper or middleware
- Use the `estimate_cost()` method to budget API spend

### Request Logging

Be careful when logging LLM requests and responses. They may contain:
- User personal data
- API keys in headers
- Sensitive business information

Log flow events and metadata (node IDs, timing, success/failure) rather than full message content.

## Deployment Checklist

1. **API keys** -- All provider API keys stored in environment variables or a secrets manager, never in code or graph metadata.
2. **Authentication** -- Code runner API has authentication enabled.
3. **Code sandboxing** -- Code execution uses Docker with memory limits, timeouts, and network restrictions.
4. **Graph validation** -- All graphs validated before production deployment.
5. **Error handling** -- Retry limits set on all nodes that call external services.
6. **Timeouts** -- Node-level timeouts configured for all external calls.
7. **Store security** -- Execution store is properly secured (file permissions, TLS, authentication).
8. **Logging** -- Sensitive data excluded from logs.
9. **Monitoring** -- Token usage and cost tracking enabled.
10. **Cleanup** -- TTL or cleanup policy for completed flow state.

## See Also

- [Architecture](architecture.md) -- System design and trust boundaries
- [Engine](engine.md) -- Error handling and timeout configuration
- [Tools](tools.md) -- Tool parameter validation
- [Providers](providers.md) -- Provider authentication and cost estimation
