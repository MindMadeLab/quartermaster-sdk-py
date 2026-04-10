# Quartermaster Open-Source — Next Steps

## Repo Housekeeping (Do First)

- [ ] Delete duplicate root-level `quartermaster-*` folders (keep only `packages/` subfolder)
- [ ] Move all packages under `packages/` if not already
- [ ] Add root `LICENSE` file (Apache 2.0)
- [ ] Add root `.gitignore`
- [ ] Initialize git: `git init && git add . && git commit -m "Initial monorepo structure"`
- [ ] Create GitHub repo: `github.com/quartermaster-ai/quartermaster` (or `mindmade-ai/quartermaster`)
- [ ] Add `.github/workflows/ci.yml` — lint + typecheck + test all packages
- [ ] Add root `Makefile` with targets: `test-all`, `lint-all`, `typecheck-all`, `publish-<pkg>`

## Critical Issue: Shared Types

Packages currently define their own types independently. Before v1.0:

- `quartermaster-graph` should be THE source of truth for: `NodeType`, `TraverseIn`, `TraverseOut`, `ThoughtType`, `MessageType`, `ErrorStrategy`, `GraphNode`, `GraphEdge`, `AgentVersion`
- `quartermaster-providers` should be THE source of truth for: `Message`, `MessageRole`, `LLMConfig`, `TokenResponse`, `ToolDefinition`
- `quartermaster-engine` should IMPORT from `quartermaster-graph` and `quartermaster-providers` instead of redefining types in its own `types.py`
- `quartermaster-nodes` should IMPORT enums from `quartermaster-graph` instead of its own `enums.py`
- `quartermaster-tools` should IMPORT `ToolDefinition` from `quartermaster-providers` for JSON Schema bridge

This is the #1 architectural task after cleanup.

## Per-Package Priority (What Agents Should Focus On)

### quartermaster-mcp-client — READY TO PUBLISH
Status: 100% implemented, 98 tests, 96.5% coverage

Next steps:
- [ ] Final review of README examples (make sure they actually run)
- [ ] Test on PyPI Test: `pip install --index-url https://test.pypi.org/simple/ quartermaster-mcp-client`
- [ ] Publish to PyPI
- [ ] Create GitHub release v0.1.0

### quartermaster-providers — READY TO PUBLISH
Status: 100% implemented, full test suite

Next steps:
- [ ] Add integration test instructions (how to run with real API keys)
- [ ] Verify each provider against latest SDK versions (openai, anthropic, google-generativeai)
- [ ] Publish to PyPI
- [ ] Create GitHub release v0.1.0

### quartermaster-code-runner — NEEDS RUNTIME BUILDS
Status: ~85% implemented, API complete

Next steps:
- [ ] Build and test all 6 runtime Docker images locally
- [ ] Create `docker-compose.yml` for standalone deployment
- [ ] Write integration tests that actually execute code in containers
- [ ] Publish Docker images to GitHub Container Registry
- [ ] Publish Python package to PyPI
- [ ] Create GitHub release v0.1.0

### quartermaster-engine — NEARLY READY
Status: ~90% implemented, 126 tests

Next steps:
- [ ] Wire up imports from `quartermaster-graph` types (replace internal `types.py` duplication)
- [ ] Implement `AsyncDispatcher`
- [ ] Add loop test (Start → If → loop back or End)
- [ ] Add sub-agent test (nested flow execution)
- [ ] Add timeout enforcement per node
- [ ] Write architecture guide (how traverse_in → think → traverse_out works)
- [ ] Publish to PyPI as v0.1.0-alpha

### quartermaster-nodes — NEEDS MORE TESTING
Status: ~75% implemented, 38 node types extracted

Next steps:
- [ ] Import enums from `quartermaster-graph` instead of own `enums.py`
- [ ] Import `LLMConfig`, `Message` from `quartermaster-providers` instead of own protocols
- [ ] Write unit tests for top 10 most-used nodes:
  1. InstructionNodeV1
  2. Decision1
  3. StartNodeV1
  4. End1
  5. Merge1
  6. If1
  7. User1
  8. FlowMemory1
  9. Static1
  10. ProgramRunner1
- [ ] Write custom node creation guide
- [ ] Publish to PyPI as v0.1.0-alpha

### quartermaster-tools — NEEDS BUILT-IN TOOLS
Status: ~80% implemented, core abstraction done

Next steps:
- [ ] Import `ToolDefinition` from `quartermaster-providers` for JSON Schema bridge
- [ ] Implement at least 3 built-in tools as examples:
  1. `WebRequestTool` — HTTP GET/POST
  2. `ReadFileTool` — Read file content
  3. `RunPythonTool` — Execute Python via quartermaster-code-runner
- [ ] Add `to_openai_tools()`, `to_anthropic_tools()` format converters
- [ ] Write tests for registry, parameter validation, tool execution
- [ ] Publish to PyPI as v0.1.0-alpha

### quartermaster-graph — NEEDS INTEGRATION TESTING
Status: ~80% implemented, schema + builder done

Next steps:
- [ ] Test GraphBuilder fluent API end-to-end (build → validate → serialize → deserialize)
- [ ] Test YAML round-trip
- [ ] Verify all 38 NodeType enums match original QM source
- [ ] Write graph format specification document
- [ ] Add more graph templates (RAG pipeline, multi-agent, tool-using agent)
- [ ] Publish to PyPI as v0.1.0-alpha

## Publication Order

1. **Week 1:** `quartermaster-mcp-client` + `quartermaster-providers` (standalone, no deps on each other)
2. **Week 2:** `quartermaster-code-runner` + `quartermaster-tools` (standalone)
3. **Week 3:** `quartermaster-graph` (standalone, Pydantic only)
4. **Week 4:** `quartermaster-nodes` (depends on quartermaster-graph, quartermaster-providers, quartermaster-tools)
5. **Week 5:** `quartermaster-engine` (depends on all above)
6. **Week 6:** `quartermaster` meta-package (installs everything)

## Open-Core Alignment Check

Everything aligns with the strategy:

OPEN (framework/SDK):
- quartermaster-mcp-client ✅
- quartermaster-code-runner ✅
- quartermaster-providers ✅
- quartermaster-tools ✅
- quartermaster-nodes ✅
- quartermaster-graph ✅
- quartermaster-engine ✅

PROPRIETARY (platform/revenue):
- Visual Agent Graph Editor (React/Electron) — NOT in this repo ✅
- Multi-tenant platform — NOT in this repo ✅
- Real-time WebSocket collaboration — NOT in this repo ✅
- Agent marketplace — NOT in this repo ✅
- Enterprise SSO/RBAC/billing — NOT in this repo ✅
- MindMade self-hosted LLM infra — NOT in this repo ✅
- Monitoring/observability dashboard — NOT in this repo ✅

No proprietary code has leaked into the open-source packages. The boundary is clean.
