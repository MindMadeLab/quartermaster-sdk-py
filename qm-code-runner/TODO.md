# qm-code-runner — Extraction TODO

Standalone, Docker-based sandboxed code execution service. Runs untrusted code in isolated containers with resource limits, network control, and timeout enforcement. Supports Python, Node.js, Go, Rust, Deno, and Bun.

## Source Files

Extract from `quartermaster/code-runner/`:

| Source File | Purpose |
|---|---|
| `app.py` | FastAPI application, API endpoints (`/run`, image mgmt) |
| `config.py` | Configuration constants (timeouts, resource limits, paths) |
| `execution.py` | Docker container execution logic (the core) |
| `schemas.py` | Pydantic request/response models |
| `security.py` | API key validation middleware |
| `environment_files.py` | Temp environment setup & cleanup |
| `images.py` | Docker image management (build, list, remove) |
| `test_app.py` | Integration tests |
| `runtime/python/sdk.py` | Python runtime SDK (runs inside container) |
| `runtime/python/mcp-client.py` | MCP client for runtime (runs inside container) |

## Extractability: 10/10

This is the easiest extraction — code-runner is already a standalone FastAPI microservice with zero coupling to the Quartermaster backend. It communicates only via HTTP API.

## Phase 1: Direct Copy & Cleanup

### 1.1 Copy Source
- [ ] Copy all files from `quartermaster/code-runner/` verbatim
- [ ] Copy `runtime/` directory with all language runtime files
- [ ] Copy Dockerfiles for each runtime (Python, Node, Go, Rust, Deno, Bun)
- [ ] Copy `pyproject.toml` or `requirements.txt`

### 1.2 Remove QM-Specific References
- [ ] Remove any Quartermaster branding from comments/docs
- [ ] Remove internal API key references (make configurable)
- [ ] Check `config.py` for hardcoded paths pointing to QM infrastructure
- [ ] Replace QM-specific environment variable names with generic ones
- [ ] Audit `security.py` — make API key auth optional/configurable

### 1.3 Configuration
- [ ] Make all resource limits configurable via environment variables:
  - CPU limit (default: 1 core)
  - Memory limit (default: 256MB)
  - Disk limit
  - Network access (on/off)
  - Execution timeout (default: 30s)
  - Max output size
- [ ] Support `.env` file loading
- [ ] Add configuration validation on startup
- [ ] Document all environment variables

## Phase 2: Runtime Images

### 2.1 Python Runtime
- [ ] Dockerfile for Python 3.11+ sandbox
- [ ] Pre-installed packages: numpy, pandas, requests, etc.
- [ ] SDK file (`sdk.py`) that runs inside container
- [ ] Support for pip installing additional packages at runtime
- [ ] Test execution with various Python scripts

### 2.2 Node.js Runtime
- [ ] Dockerfile for Node.js 20+ sandbox
- [ ] Support for npm package installation
- [ ] TypeScript support via ts-node
- [ ] Test execution

### 2.3 Go Runtime
- [ ] Dockerfile for Go 1.21+ sandbox
- [ ] Module support
- [ ] Test execution

### 2.4 Rust Runtime
- [ ] Dockerfile for Rust stable sandbox
- [ ] Cargo support
- [ ] Test execution

### 2.5 Deno Runtime
- [ ] Dockerfile for Deno sandbox
- [ ] Permission flags support
- [ ] Test execution

### 2.6 Bun Runtime
- [ ] Dockerfile for Bun sandbox
- [ ] Test execution

### 2.7 Build System
- [ ] Makefile or script to build all runtime images
- [ ] Image tagging strategy (versioned)
- [ ] Image size optimization (multi-stage builds)
- [ ] Pre-built images on Docker Hub / GitHub Container Registry

## Phase 3: API Design

### 3.1 Core Endpoints
- [ ] `POST /run` — Execute code (language, code, stdin, env vars, timeout)
- [ ] `GET /runtimes` — List available runtimes and versions
- [ ] `GET /health` — Health check
- [ ] `POST /run/stream` — Streaming execution output (nice to have)

### 3.2 Request Schema
```
{
  "language": "python",
  "code": "print('hello')",
  "stdin": "",
  "timeout": 30,
  "env": {"API_KEY": "..."},
  "packages": ["requests"],
  "network": false,
  "memory_mb": 256
}
```

### 3.3 Response Schema
```
{
  "stdout": "hello\n",
  "stderr": "",
  "exit_code": 0,
  "execution_time_ms": 142,
  "timed_out": false
}
```

### 3.4 Security
- [ ] Optional API key authentication
- [ ] Rate limiting (configurable)
- [ ] Request size limits
- [ ] No privilege escalation from container
- [ ] Network isolation by default
- [ ] Resource cleanup on timeout/error (orphan container killing)

## Phase 4: Testing

### 4.1 Unit Tests
- [ ] Config validation tests
- [ ] Schema validation tests
- [ ] Security middleware tests

### 4.2 Integration Tests
- [ ] Execute Python code → verify stdout/stderr/exit_code
- [ ] Execute Node.js code → verify output
- [ ] Execute Go code → compile + run
- [ ] Timeout enforcement test (infinite loop → killed)
- [ ] Memory limit test (allocate too much → OOM killed)
- [ ] Network isolation test (curl fails when network=false)
- [ ] Concurrent execution test (10 simultaneous runs)
- [ ] Error handling test (syntax errors, runtime errors)
- [ ] Large output handling (truncation)
- [ ] Environment variable passing

### 4.3 Security Tests
- [ ] Container escape attempt (should fail)
- [ ] File system access outside sandbox (should fail)
- [ ] Fork bomb protection
- [ ] Process limit enforcement

## Phase 5: Documentation

### 5.1 README
- [ ] Quick start with Docker Compose
- [ ] API reference for all endpoints
- [ ] Configuration reference (all env vars)
- [ ] Runtime-specific notes
- [ ] Self-hosting guide

### 5.2 Docker Compose Example
- [ ] `docker-compose.yml` for standalone deployment
- [ ] Includes code-runner + all runtime images
- [ ] Health checks configured
- [ ] Volume mounts for persistent config

## Phase 6: CI/CD & Distribution

### 6.1 GitHub Actions
- [ ] Lint (ruff)
- [ ] Unit tests
- [ ] Integration tests (needs Docker-in-Docker or similar)
- [ ] Build all runtime images
- [ ] Push images to GitHub Container Registry

### 6.2 Distribution
- [ ] PyPI package for the FastAPI app
- [ ] Docker image on Docker Hub: `quartermasterai/code-runner`
- [ ] Runtime images: `quartermasterai/runtime-python`, etc.
- [ ] Helm chart for Kubernetes deployment (nice to have)

## Architecture Notes

### Why This Is Valuable Open-Source
- Every AI agent framework needs code execution
- Most roll their own insecure `subprocess.run()` solutions
- This provides production-grade sandboxing out of the box
- Multi-language support is rare in open-source alternatives
- Docker isolation is battle-tested security

### Competitors
- E2B (closed-source, hosted only, $$$)
- Code Interpreter API (OpenAI, closed)
- Jupyter kernels (heavy, not designed for sandboxing)
- **This fills the gap: open-source, self-hosted, multi-language, secure**

## Timeline Estimate

- Phase 1 (Copy & Cleanup): 1 day
- Phase 2 (Runtime Images): 2-3 days
- Phase 3 (API Polish): 1 day
- Phase 4 (Testing): 2-3 days
- Phase 5 (Docs): 1 day
- Phase 6 (CI/CD): 1-2 days

**Total: ~1 week**
