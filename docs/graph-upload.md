# Graph Upload — Deploy Agent Graphs to Quartermaster Cloud

## Overview

Agent graphs built with the `quartermaster-graph` builder need a way to be
**uploaded and deployed** to Quartermaster Cloud for execution. This document
specifies the upload API and the open-source client that calls it.

---

## What We Need in the Quartermaster Cloud Backend

### 1. Agent CRUD Endpoints

```
POST   /v1/agents                    # Create new agent
GET    /v1/agents                    # List all agents
GET    /v1/agents/{agent_id}         # Get agent details
PUT    /v1/agents/{agent_id}         # Update agent metadata
DELETE /v1/agents/{agent_id}         # Delete agent
```

**Create Agent:**
```json
POST /v1/agents
Authorization: Bearer qm-xxxxxxxxxxxxxxxx
Content-Type: application/json

{
    "name": "Customer Support Agent",
    "description": "Handles customer inquiries with department routing",
    "tags": ["support", "production"]
}
```

**Response:**
```json
{
    "id": "agt_abc123",
    "name": "Customer Support Agent",
    "description": "Handles customer inquiries with department routing",
    "tags": ["support", "production"],
    "current_version": null,
    "created_at": "2026-04-13T10:00:00Z",
    "updated_at": "2026-04-13T10:00:00Z"
}
```

### 2. Version Upload Endpoint

```
POST /v1/agents/{agent_id}/versions
Authorization: Bearer qm-xxxxxxxxxxxxxxxx
Content-Type: application/json
```

**Request body** — the full graph serialized as JSON:
```json
{
    "version": "1.2.0",
    "graph": {
        "start_node_id": "uuid-...",
        "nodes": [
            {
                "id": "uuid-...",
                "type": "Start1",
                "name": "Start",
                "traverse_in": "AwaitAll",
                "traverse_out": "SpawnAll",
                "metadata": {},
                "position": {"x": 0, "y": 0}
            },
            {
                "id": "uuid-...",
                "type": "Instruction1",
                "name": "Analyze",
                "metadata": {
                    "system_instruction": "Analyze the input",
                    "model": "gpt-4o",
                    "provider": "openai",
                    "temperature": 0.7
                }
            }
        ],
        "edges": [
            {
                "id": "uuid-...",
                "source_id": "uuid-...",
                "target_id": "uuid-...",
                "label": "",
                "is_main": true
            }
        ]
    },
    "features": "department routing, quality check, memory",
    "publish": false
}
```

**Response:**
```json
{
    "id": "ver_xyz789",
    "agent_id": "agt_abc123",
    "version": "1.2.0",
    "node_count": 15,
    "edge_count": 18,
    "is_published": false,
    "validation": {
        "valid": true,
        "warnings": [],
        "errors": []
    },
    "created_at": "2026-04-13T10:05:00Z"
}
```

### 3. Version Management

```
GET    /v1/agents/{agent_id}/versions                # List versions
GET    /v1/agents/{agent_id}/versions/{version_id}   # Get specific version
POST   /v1/agents/{agent_id}/versions/{version_id}/publish   # Publish version
POST   /v1/agents/{agent_id}/versions/{version_id}/rollback  # Rollback to version
DELETE /v1/agents/{agent_id}/versions/{version_id}   # Delete version
GET    /v1/agents/{agent_id}/versions/{v1}/diff/{v2} # Diff two versions
```

### 4. Execution Endpoints

```
POST /v1/agents/{agent_id}/run
Authorization: Bearer qm-xxxxxxxxxxxxxxxx

{
    "version": "1.2.0",          # optional, defaults to published version
    "input": "Hello, I need help with my order",
    "variables": {               # pre-set variables
        "user_id": "usr_123",
        "language": "en"
    },
    "stream": true               # stream execution events
}
```

**Streaming response** (SSE):
```
data: {"event": "node_start", "node_id": "uuid-...", "node_name": "Analyze", "node_type": "Instruction1"}
data: {"event": "node_output", "node_id": "uuid-...", "content": "The user is asking about..."}
data: {"event": "node_complete", "node_id": "uuid-..."}
data: {"event": "decision", "node_id": "uuid-...", "chosen": "support"}
data: {"event": "node_start", "node_id": "uuid-...", "node_name": "Support Handler"}
...
data: {"event": "flow_complete", "result": "Your order #12345 has been updated."}
data: [DONE]
```

---

## What We Need in Quartermaster Open-Source

### 1. Graph Upload Client

New module: `quartermaster-graph/src/quartermaster_graph/cloud.py`

```python
class QuartermasterCloud:
    """Client for uploading and managing agent graphs on Quartermaster Cloud."""

    def __init__(self, api_key: str | None = None, base_url: str = "https://api.quartermaster.ai"):
        self.api_key = api_key or os.environ.get("QUARTERMASTER_API_KEY")
        self.base_url = base_url

    def upload(self, graph: GraphBuilder, name: str | None = None,
               version: str = "0.1.0", publish: bool = False) -> dict:
        """Upload a graph to Quartermaster Cloud.

        Returns the created version metadata.
        """
        ...

    def list_agents(self) -> list[dict]:
        """List all agents in the account."""
        ...

    def get_agent(self, agent_id: str) -> dict:
        """Get agent details."""
        ...

    def publish(self, agent_id: str, version_id: str) -> dict:
        """Publish a specific version."""
        ...

    def run(self, agent_id: str, input_text: str,
            variables: dict | None = None,
            version: str | None = None) -> dict:
        """Execute an agent and return the result."""
        ...
```

### 2. CLI Upload Command

A simple script or entry point for uploading from the command line:

```bash
# Upload a graph from a Python file
quartermaster upload my_agent.py --name "My Agent" --version 1.0.0

# Or from Python
python -m quartermaster_graph.cloud upload my_agent.py
```

### 3. Fluent Upload from Builder

```python
from quartermaster_graph import Graph
from quartermaster_graph.cloud import QuartermasterCloud

agent = (
    Graph("My Agent")
    .start()
    .user("Input")
    .instruction("Process", system_instruction="Handle the request")
    .end()
)

cloud = QuartermasterCloud()  # uses QUARTERMASTER_API_KEY env var
result = cloud.upload(agent, version="1.0.0", publish=True)
print(f"Uploaded: {result['id']}")
```

---

## Serialization Format

The upload uses the existing `to_json()` serialization from `quartermaster-graph`.
The graph is serialized as a JSON object with the `AgentVersion` schema:

```python
from quartermaster_graph import to_json

graph = Graph("My Agent").start().instruction("Work").end()
version = graph.to_version()
json_data = to_json(version)  # dict ready for upload
```

This already works — the `to_json` / `from_json` functions handle full
round-trip serialization including all node metadata, edge labels, positions, etc.

---

## Validation

Before upload, the client should:
1. Run `validate_graph()` locally to catch errors early
2. Verify all required metadata is present (model, provider on instruction nodes)
3. Check version string format (semver)

The server performs its own validation and returns errors in the response.

---

## Implementation Priority

1. **P0**: `QuartermasterCloud.upload()` — serialize graph and POST to API
2. **P0**: Server-side version storage and validation
3. **P1**: `QuartermasterCloud.run()` — execute uploaded graphs
4. **P1**: Version management (list, publish, rollback)
5. **P2**: CLI upload command
6. **P2**: Streaming execution
7. **P2**: Diff between versions (already implemented in quartermaster-graph)
8. **P3**: Webhooks for execution events
9. **P3**: Scheduled execution

---

## Authentication

Same API key as the proxy endpoint:
```bash
export QUARTERMASTER_API_KEY=qm-xxxxxxxxxxxxxxxx
```

All endpoints require the `Authorization: Bearer qm-xxx` header.
