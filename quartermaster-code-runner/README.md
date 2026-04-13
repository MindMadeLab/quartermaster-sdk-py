# quartermaster-code-runner

Secure sandboxed code execution service. Runs untrusted code in isolated Docker containers with support for Python, Node.js, Go, Rust, Deno, and Bun.

[![PyPI version](https://img.shields.io/pypi/v/quartermaster-code-runner.svg)](https://pypi.org/project/quartermaster-code-runner/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)

## Features

- **6 Runtimes**: Python, Node.js, Go, Rust, Deno, Bun
- **Docker Isolation**: Each execution runs in its own container with read-only filesystem
- **Resource Limits**: Configurable CPU, memory, and disk quotas per execution
- **Network Control**: Outbound network access toggled per request
- **Timeout Enforcement**: Hard container kill after configurable duration
- **File Injection**: Send multiple source files alongside the main code
- **Prebuilt Images**: Extend base runtimes with custom dependencies
- **Auto Cleanup**: Orphaned containers and old images cleaned up automatically
- **API Key Auth**: Optional authentication via `X-API-Key` header or Bearer token
- **Health Checks**: Built-in `/health` endpoint with Docker connectivity status
- **Standalone**: No dependency on other Quartermaster packages

## Supported Runtimes

| Runtime | Image Name | Sandbox |
|---------|-----------|---------|
| Python | `python` | Alpine Linux |
| Node.js | `node` | Alpine Linux |
| Go | `go` | Alpine Linux |
| Rust | `rust` | Alpine Linux |
| Deno | `deno` | Alpine Linux |
| Bun | `bun` | Alpine Linux |

## Installation

### As a Service (recommended)

```bash
pip install quartermaster-code-runner
uvicorn quartermaster_code_runner.app:app --host 0.0.0.0 --port 8000
```

Requires Docker to be running and accessible.

### Docker Compose

```yaml
services:
  code-runner:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
    environment:
      - LOG_LEVEL=info
      - DEFAULT_TIMEOUT=30
      - DEFAULT_MEMORY=256m
```

## Quick Start

### Execute Python Code

```bash
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{
    "code": "print(sum(range(1, 101)))",
    "image": "python"
  }'
```

Response:

```json
{
  "stdout": "5050\n",
  "stderr": "",
  "exit_code": 0,
  "execution_time": 0.3421
}
```

### Execute Node.js Code

```bash
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{
    "code": "console.log(Array.from({length: 5}, (_, i) => i * i))",
    "image": "node"
  }'
```

### With Environment Variables and Resource Limits

```bash
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{
    "code": "import os; print(os.environ[\"API_KEY\"])",
    "image": "python",
    "timeout": 10,
    "mem_limit": "128m",
    "cpu_shares": 256,
    "allow_network": false,
    "environment": {
      "API_KEY": "secret-123"
    }
  }'
```

### With Additional Files

```bash
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{
    "code": "from helpers import greet; print(greet(\"world\"))",
    "image": "python",
    "files": {
      "helpers.py": "def greet(name): return f\"Hello, {name}!\""
    }
  }'
```

### Python Client Usage

```python
from quartermaster_code_runner import CodeExecutionRequest

request = CodeExecutionRequest(
    code="print('Hello from the sandbox')",
    image="python",
    timeout=10,
    mem_limit="128m",
    allow_network=False,
)
```

## API Reference

### POST /run

Execute code in a sandboxed Docker container.

**Request body** (`CodeExecutionRequest`):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `code` | `string` | required | Source code to execute |
| `image` | `string` | `"python"` | Runtime: `python`, `node`, `go`, `rust`, `deno`, `bun` |
| `files` | `dict[str, str]` | `null` | Additional files `{filename: content}` |
| `entrypoint` | `string` | `null` | Custom entrypoint command |
| `timeout` | `int` | `null` (uses server default) | Execution timeout in seconds |
| `mem_limit` | `string` | `null` (uses server default) | Memory limit, e.g. `"256m"` |
| `cpu_shares` | `int` | `null` (uses server default) | Docker CPU shares |
| `disk_limit` | `string` | `null` (uses server default) | Disk limit for tmpfs |
| `allow_network` | `bool` | `true` | Allow outbound network access |
| `environment` | `dict[str, str]` | `null` | Environment variables to inject |
| `prebuild_spec` | `object` | `null` | Prebuild spec: `{base_image, setup_script}` |

**Response** (`CodeExecutionResponse`):

| Field | Type | Description |
|-------|------|-------------|
| `stdout` | `string` | Standard output |
| `stderr` | `string` | Standard error |
| `exit_code` | `int` | Process exit code |
| `execution_time` | `float` | Wall-clock time in seconds |
| `metadata` | `dict` | Optional metadata from the container |

### GET /health

Returns service health and Docker connectivity.

```json
{
  "status": "ok",
  "docker_connected": true,
  "auth_enabled": false
}
```

### GET /runtimes

List available runtime images with metadata (alias for `/images`).

```json
{
  "images": [
    {
      "id": "code-runner-python",
      "name": "Python",
      "description": "Python runtime",
      "default_entrypoint": "python main.py",
      "file_extension": ".py",
      "main_file": "main.py"
    }
  ],
  "default": "code-runner-python"
}
```

### POST /prebuild

Build a prebuilt image extending a base runtime with custom dependencies.

```bash
curl -X POST http://localhost:8000/prebuild \
  -H "Content-Type: application/json" \
  -d '{
    "tag": "my-scipy",
    "base_image": "python",
    "setup_script": "pip install scipy numpy"
  }'
```

### GET /prebuilds

List all prebuilt images. `DELETE /prebuilds/{tag}` removes one.

## Security Model

Each execution is isolated with multiple layers:

1. **Container isolation** -- separate Docker container per execution
2. **Read-only filesystem** -- containers run with `read_only=True`; writes go to tmpfs
3. **Network control** -- outbound access disabled when `allow_network=false`
4. **Resource limits** -- CPU shares, memory limit, disk quota enforced by Docker
5. **Timeout enforcement** -- containers killed after the configured timeout
6. **Input validation** -- code size capped at 1 MB; reserved env vars blocked; path traversal rejected

Reserved environment variables (`ENCODED_CODE`, `ENCODED_FILES`, `CUSTOM_ENTRYPOINT`) cannot be overridden by callers.

## Configuration

All settings load from environment variables with sensible defaults. Use a `.env` file or set them directly.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | Bind port |
| `LOG_LEVEL` | `info` | Log level (`debug`, `info`, `warning`, `error`) |
| `DEFAULT_TIMEOUT` | `30` | Default execution timeout (seconds) |
| `MAX_TIMEOUT` | `300` | Hard maximum timeout (seconds) |
| `DEFAULT_MEMORY` | `256m` | Default memory limit |
| `MAX_MEMORY_MB` | `2048` | Maximum memory (MB) |
| `DEFAULT_CPU_SHARES` | `512` | Default Docker CPU shares |
| `MAX_CPU_CORES` | `4.0` | Maximum CPU cores |
| `DEFAULT_DISK` | `512m` | Default tmpfs disk limit |
| `MAX_DISK_MB` | `5000` | Maximum disk (MB) |
| `DOCKER_SOCKET` | `/var/run/docker.sock` | Docker socket path |
| `AUTH_TOKEN` | _(none)_ | Bearer token for authentication |
| `CODE_RUNNER_API_KEYS` | _(none)_ | Comma-separated API keys for `X-API-Key` header |
| `CLEANUP_INTERVAL_HOURS` | `24` | Hours between automatic prebuild cleanup |
| `CLEANUP_MAX_AGE_DAYS` | `7` | Max age (days) for prebuilt images before cleanup |
| `RUNTIME_DIR` | _(auto)_ | Path to runtime Dockerfile directories |

When neither `AUTH_TOKEN` nor `CODE_RUNNER_API_KEYS` is set, authentication is disabled.

### Error Types

```python
from quartermaster_code_runner import (
    CodeRunnerError,          # Base exception
    DockerError,              # Docker communication failure
    ExecutionError,           # Code execution failure
    InvalidLanguageError,     # Unsupported runtime
    ResourceExhaustedError,   # Resource limit exceeded
    RuntimeNotAvailableError, # Runtime image not found
    TimeoutError,             # Execution timeout
)
```

## Contributing

Contributions welcome. See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## License

Apache License 2.0. See [LICENSE](../LICENSE) for details.
