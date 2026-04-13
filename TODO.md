# Quartermaster — Project Task List

Comprehensive task list for the Quartermaster open-source AI agent orchestration framework.

**Status legend:** `[ ]` = todo, `[x]` = done

---

## 1. Engine — Execution Features

Current state: FlowRunner works with sync/async/threaded dispatchers, InMemory + SQLite stores, streaming via FlowEvents, per-node error handling (retry/skip/stop), per-node timeouts.

### Rate Limiting & Backoff
- [ ] Add provider-level rate limiting (token bucket or sliding window)
- [ ] Implement exponential backoff with jitter on 429/503 responses
- [ ] Add configurable quota management (requests/min, tokens/min per provider)
- **File:** `quartermaster-engine/src/quartermaster_engine/runner/flow_runner.py`

### Cost Tracking
- [ ] Add automatic token counting per flow execution (input + output tokens)
- [ ] Aggregate cost per provider/model using published pricing tables
- [ ] Add budget alerts / hard limits (stop flow if spend exceeds threshold)
- [ ] Emit `CostEvent` in the FlowEvent stream
- **File:** new `quartermaster-engine/src/quartermaster_engine/cost.py`

### Caching
- [ ] Add prompt/response caching layer (hash prompt → cached response)
- [ ] Support TTL-based cache expiration
- [ ] Configurable per-node (some nodes should never cache)
- **File:** new `quartermaster-engine/src/quartermaster_engine/cache.py`

### Redis Store
- [ ] Implement `RedisStore` for distributed/multi-process execution
- [ ] Currently listed as optional dep (`redis>=5.0`) but no implementation exists
- **File:** new `quartermaster-engine/src/quartermaster_engine/stores/redis_store.py`

### Global Flow Timeout
- [ ] Add flow-level timeout (currently only per-node `node.timeout`)
- [ ] Cancel all running nodes when flow timeout expires
- **File:** `quartermaster-engine/src/quartermaster_engine/runner/flow_runner.py`

### Observability & Telemetry
- [ ] Add OpenTelemetry trace exporter (currently spans are in-memory only)
- [ ] Structured logging hooks for latency, token usage, errors per node
- [ ] Export traces to Jaeger, Zipkin, or OTLP collector
- **File:** new `quartermaster-engine/src/quartermaster_engine/telemetry.py`

---

## 2. MCP Client — Protocol Gaps

Current state: JSON-RPC 2.0 over SSE/HTTP, async/sync, tool discovery, error translation.

- [ ] Add WebSocket transport (in addition to SSE/HTTP)
- [ ] Add MCP resource handling (file/blob streaming)
- [ ] Add long-lived server subscriptions
- **File:** `quartermaster-mcp-client/src/quartermaster_mcp_client/transports.py`

---

## 3. Code Runner — Enhancements

Current state: Docker sandboxed execution, 6 runtimes (Python, Node.js, Go, Rust, Deno, Bun), FastAPI API, auth, resource limits.

- [ ] Add `docker-compose.yml` for easy local setup
- [ ] Add execution result caching (same code + same input = cached output)
- [ ] Add file upload/download support for code that reads/writes files
- **File:** `quartermaster-code-runner/`

---

## 4. API Gateway for Engine

Currently the engine has no HTTP API — only the code-runner has a FastAPI service.

- [ ] Add FastAPI wrapper for graph execution:
  - `POST /run` — Execute a graph with input
  - `POST /run/stream` — SSE streaming execution
  - `POST /resume` — Resume a paused flow (user input)
  - `GET /flows/{id}` — Get flow status/result
- [ ] Add WebSocket endpoint for real-time streaming
- [ ] Add authentication middleware (API key, JWT)
- **File:** new `quartermaster-engine/src/quartermaster_engine/api.py` or separate package

---

## 5. Graph — Schema Versioning

Current state: Fluent builder API, Pydantic v2 models, JSON/YAML serialization, comprehensive validation.

- [ ] Add graph schema version field to `AgentVersion` model
- [ ] Add migration helpers for upgrading graphs between schema versions
- **File:** `quartermaster-graph/src/quartermaster_graph/models.py`

---

## 6. Testing & Quality

### Cross-Package Integration Tests
- [ ] Graph → Engine → Provider (mock) end-to-end flow test
- [ ] Build graph → validate → serialize → deserialize round-trip test
- [ ] SDK import smoke test (`from quartermaster_sdk import Graph, NodeType`)

### CI Pipeline
- [ ] Add test coverage reporting (`pytest-cov`) to CI
- [ ] Verify all examples run in CI (currently 16 + run_interactive)

### Code Quality
- [ ] Run `ruff check` + `ruff format` across entire repo
- [ ] Run `mypy` strict mode across all packages
- [ ] Review all `# type: ignore` comments
- [ ] Remove stale worktree directories:
  - `quartermaster-graph/.claude/worktrees/goofy-mahavira/`
  - `quartermaster-engine/.claude/worktrees/recursing-raman/`

---

## 7. Packaging & Release

- [x] Add Apache 2.0 LICENSE to all packages
- [x] Remove 17 unused schema-only node types from `NodeType` enum
- [ ] Add `py.typed` marker files to all packages (only `quartermaster-mcp-client` has one)
- [ ] Standardize PyPI classifiers across all 8 `pyproject.toml`:
  - `quartermaster-providers`: add Python 3.13
  - `quartermaster-tools`: add AI classifier
  - `quartermaster-mcp-client`: remove Python 3.14
  - `quartermaster-sdk`: change from Beta to Alpha
- [ ] Create root `CHANGELOG.md`
- [ ] Review all docs for external users (no proprietary cloud references)
- [ ] Add deployment guide `docs/deployment.md`

### Pre-Release Checklist (v0.1.0)
- [ ] All 1,958+ tests pass
- [ ] All examples run clean
- [ ] Lint + format + type-check clean
- [ ] Every package has: LICENSE, README.md, pyproject.toml, tests
- [ ] CHANGELOG.md with v0.1.0 entry
- [ ] GitHub repo: description, topics, homepage URL set
- [ ] `PYPI_API_TOKEN` secret configured
- [ ] Test `publish.yml` with manual trigger
- [ ] Tag `v0.1.0` → packages on PyPI

---

## 8. Examples & Docs

### New Examples
- [ ] `examples/17_streaming.py` — Token-level streaming demo
- [ ] Simple "real execution" example (like `run_interactive.py` but minimal)

### Documentation
- [ ] `docs/deployment.md` — Running agents in production
- [ ] Verify all doc cross-references are valid

---

## File Reference

| Component | Key Files |
|-----------|-----------|
| Engine / FlowRunner | `quartermaster-engine/src/quartermaster_engine/runner/flow_runner.py` |
| Dispatchers | `quartermaster-engine/src/quartermaster_engine/runner/dispatchers/` |
| Stores | `quartermaster-engine/src/quartermaster_engine/stores/` (InMemory, SQLite) |
| Streaming events | `quartermaster-engine/src/quartermaster_engine/runner/events.py` |
| Provider protocol | `quartermaster-providers/src/quartermaster_providers/providers/base.py` |
| Anthropic provider | `quartermaster-providers/src/quartermaster_providers/providers/anthropic.py` |
| OpenAI provider | `quartermaster-providers/src/quartermaster_providers/providers/openai.py` |
| Google provider | `quartermaster-providers/src/quartermaster_providers/providers/google.py` |
| Node implementations | `quartermaster-nodes/quartermaster_nodes/nodes/` (35 node types) |
| Graph builder | `quartermaster-graph/src/quartermaster_graph/builder.py` |
| Graph validation | `quartermaster-graph/src/quartermaster_graph/validation.py` |
| Graph serialization | `quartermaster-graph/src/quartermaster_graph/serialization.py` |
| NodeType enum | `quartermaster-graph/src/quartermaster_graph/enums.py` |
| Tools (69 built-in) | `quartermaster-tools/src/quartermaster_tools/builtin/` |
| MCP client | `quartermaster-mcp-client/src/quartermaster_mcp_client/client.py` |
| Code runner | `quartermaster-code-runner/src/quartermaster_code_runner/` |
| CI pipeline | `.github/workflows/ci.yml` |
| Publish pipeline | `.github/workflows/publish.yml` |
| All docs (17 files) | `docs/` |
| All examples (17 files) | `examples/` |
