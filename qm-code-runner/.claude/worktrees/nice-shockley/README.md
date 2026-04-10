# qm-code-runner

[![Docker Image](https://img.shields.io/badge/Docker-Available-2496ED?logo=docker)](https://github.com/quartermaster-ai/quartermaster/pkgs/container/qm-code-runner)
[![Python Versions](https://img.shields.io/pypi/pyversions/qm-code-runner.svg)](https://pypi.org/project/qm-code-runner/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI Status](https://github.com/quartermaster-ai/quartermaster/actions/workflows/test.yml/badge.svg)](https://github.com/quartermaster-ai/quartermaster/actions)

A production-ready service for secure, sandboxed code execution with support for 6+ programming languages. Designed for AI agents, automated tools, and untrusted code environments.

## What is qm-code-runner?

`qm-code-runner` is a FastAPI-based microservice that executes code in isolated Docker containers. Each execution runs in its own lightweight container with:

- **Resource isolation**: CPU, memory, disk quotas per execution
- **Network isolation**: Controlled outbound access (configurable)
- **Timeout enforcement**: Hard kill after configurable duration
- **Multi-language support**: Python, Node.js, Go, Rust, Deno, Bun
- **Streaming output**: Real-time stdout/stderr streaming
- **Environment variables**: Secure secret injection per execution

## Supported Runtimes

| Language | Version | Package Manager | Sandbox |
|----------|---------|-----------------|---------|
| Python | 3.11+ | pip | Alpine Linux |
| Node.js | 20+ | npm | Alpine Linux |
| Go | 1.21+ | go mod | Alpine Linux |
| Rust | 1.75+ | cargo | Alpine Linux |
| Deno | 1.40+ | native | Alpine Linux |
| Bun | 1.0+ | native | Alpine Linux |

All runtimes run in Alpine Linux containers for minimal footprint (~50-200MB each).

## Features

- **Dual Deployment**: Library + Standalone service
- **Async API**: FastAPI with streaming responses
- **Resource Limits**: CPU, memory, disk constraints per execution
- **Timeout Protection**: Hard timeout with container kill
- **Output Capture**: Realtime stdout/stderr streaming
- **Error Tracking**: Exception details and exit codes
- **Secret Management**: Environment variable injection
- **Auto Cleanup**: Orphaned container detection and cleanup
- **Health Checks**: Built-in health endpoint
- **Comprehensive Logging**: Structured execution logs
- **Type Safe**: Full type hints, mypy strict mode

## Installation

### As a Library

```bash
pip install qm-code-runner
```

### As a Standalone Service

```bash
docker pull ghcr.io/quartermaster-ai/qm-code-runner:latest
docker run -it \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -p 8000:8000 \
  ghcr.io/quartermaster-ai/qm-code-runner:latest
```

### Docker Compose

```yaml
services:
  code-runner:
    image: ghcr.io/quartermaster-ai/qm-code-runner:latest
    ports:
      - "8000:8000"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
    environment:
      - LOG_LEVEL=info
      - DEFAULT_TIMEOUT=30
      - MAX_MEMORY_MB=512
```

## Quick Start

### Running Python Code

```bash
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{
    "language": "python",
    "code": "print(\"Hello, World!\")\nprint(2 + 2)",
    "timeout": 10
  }'
```

Response:
```json
{
  "exitCode": 0,
  "stdout": "Hello, World!\n4\n",
  "stderr": "",
  "duration": 0.25,
  "executionId": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Running Node.js Code

```bash
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{
    "language": "node",
    "code": "console.log(\"Hello from Node\"); console.log(Array(5).fill(0).map((_, i) => i).join(\",\"));",
    "timeout": 10
  }'
```

### With Environment Variables

```bash
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{
    "language": "python",
    "code": "import os; print(os.environ.get(\"API_KEY\"))",
    "timeout": 10,
    "environment": {
      "API_KEY": "secret-key-123"
    }
  }'
```

### With Resource Limits

```bash
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{
    "language": "python",
    "code": "import time; time.sleep(5); print(\"Done\")",
    "timeout": 2,
    "resources": {
      "cpuCores": 1,
      "memoryMb": 128,
      "diskMb": 100
    }
  }'
```

## API Reference

### POST /execute

Execute code in a sandboxed container.

#### Request

```json
{
  "language": "python|node|go|rust|deno|bun",
  "code": "source code to execute",
  "timeout": 30,
  "environment": {
    "VAR_NAME": "value"
  },
  "resources": {
    "cpuCores": 1,
    "memoryMb": 512,
    "diskMb": 1024
  }
}
```

#### Response (Success)

```json
{
  "exitCode": 0,
  "stdout": "output text",
  "stderr": "",
  "duration": 1.25,
  "executionId": "uuid",
  "language": "python"
}
```

#### Response (Timeout)

```json
{
  "exitCode": 124,
  "stdout": "partial output",
  "stderr": "Execution timeout after 10 seconds",
  "duration": 10.05,
  "executionId": "uuid",
  "error": "timeout"
}
```

### GET /health

Health check endpoint.

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "ok",
  "dockerConnected": true,
  "imagesReady": {
    "python": true,
    "node": true,
    "go": true,
    "rust": true,
    "deno": true,
    "bun": true
  }
}
```

### GET /runtimes

List available runtimes and versions.

```bash
curl http://localhost:8000/runtimes
```

Response:
```json
{
  "runtimes": {
    "python": {
      "version": "3.11.7",
      "imageId": "sha256:abc123",
      "sizeBytes": 52428800
    },
    "node": {
      "version": "20.10.0",
      "imageId": "sha256:def456",
      "sizeBytes": 125829120
    }
  }
}
```

## Security Model

### Isolation Layers

1. **Container Isolation**: Each execution runs in a separate Docker container
2. **Filesystem Isolation**: Temporary directories with automatic cleanup
3. **Network Isolation**: No outbound network by default (configurable)
4. **Resource Limits**: CPU and memory quotas prevent DoS
5. **Process Isolation**: PID namespace isolation from host
6. **User Isolation**: Runs as non-root user (uid 1000)

### Attack Surface Mitigation

- **Code injection**: Input validated and escaped before container execution
- **Resource exhaustion**: CPU, memory, disk limits enforced
- **Infinite loops**: Timeout with hard container kill
- **Privilege escalation**: Non-root execution, no sudo access
- **Secret leakage**: Environment variables not logged, container cleanup on exit
- **Container escape**: Alpine Linux base image, no shell access

### Best Practices

```python
# DO: Set strict timeouts
response = await execute_code(
    code="...",
    timeout=5,  # Hard limit in seconds
    resources={"memoryMb": 256}
)

# DO: Use specific environment variables
response = await execute_code(
    code="...",
    environment={"API_KEY": "..."}  # Only what's needed
)

# DON'T: Trust arbitrary code execution
# Always run untrusted code through qm-code-runner

# DON'T: Run without resource limits
# Use resources parameter to prevent DoS
```

## Configuration

### Environment Variables

```bash
# Server
PORT=8000
HOST=0.0.0.0
LOG_LEVEL=info  # debug, info, warning, error

# Execution Defaults
DEFAULT_TIMEOUT=30          # seconds
MAX_TIMEOUT=300             # seconds (hard limit)
DEFAULT_MEMORY_MB=512       # megabytes
MAX_MEMORY_MB=2048          # megabytes (hard limit)
DEFAULT_CPU_CORES=1         # fractional allowed (0.5)
MAX_CPU_CORES=4             # cores (hard limit)
DEFAULT_DISK_MB=500         # megabytes
MAX_DISK_MB=5000            # megabytes (hard limit)

# Docker
DOCKER_SOCKET=/var/run/docker.sock
DOCKER_NETWORK=bridge       # Network mode
CLEANUP_INTERVAL_MINUTES=5  # Orphan container cleanup

# Security
ENABLE_NETWORK=false        # Allow outbound network
ALLOWED_HOSTS=              # Comma-separated hosts (if ENABLE_NETWORK=true)
AUTH_TOKEN=                 # Bearer token for authentication (optional)
```

### Programmatic Configuration

```python
from qm_code_runner.config import ExecutionConfig, ResourceLimits

config = ExecutionConfig(
    default_timeout=30,
    default_memory_mb=512,
    enable_network=False,
    cleanup_interval_minutes=5,
)

limits = ResourceLimits(
    cpuCores=2,
    memoryMb=1024,
    diskMb=2000,
)
```

## Usage as Library

### Async Execution

```python
import asyncio
from qm_code_runner import CodeRunner

async def main():
    runner = CodeRunner()
    await runner.initialize()
    
    result = await runner.execute(
        language="python",
        code="print(sum([1, 2, 3, 4, 5]))",
        timeout=10,
    )
    
    print(f"Exit code: {result.exitCode}")
    print(f"Output: {result.stdout}")
    print(f"Duration: {result.duration}s")
    
    await runner.cleanup()

asyncio.run(main())
```

### Sync Execution

```python
from qm_code_runner import CodeRunner

runner = CodeRunner()
runner.initialize_sync()

result = runner.execute_sync(
    language="python",
    code="import platform; print(platform.python_version())",
    timeout=10,
)

print(result.stdout)
runner.cleanup_sync()
```

### Streaming Output

```python
import asyncio
from qm_code_runner import CodeRunner

async def main():
    runner = CodeRunner()
    await runner.initialize()
    
    async for chunk in runner.execute_streaming(
        language="python",
        code="for i in range(10): print(i); import time; time.sleep(0.1)",
        timeout=30,
    ):
        if chunk.type == "stdout":
            print(f"Output: {chunk.data}", end="")
        elif chunk.type == "stderr":
            print(f"Error: {chunk.data}", end="", file=sys.stderr)
        elif chunk.type == "exit":
            print(f"Exit code: {chunk.exitCode}")

asyncio.run(main())
```

## Development

### Setup

```bash
git clone https://github.com/quartermaster-ai/quartermaster.git
cd packages/qm-code-runner
pip install -e ".[dev]"
```

### Building Runtime Images

```bash
# Build all runtime images
make build-runtimes

# Build specific runtime
make build-runtime lang=python
make build-runtime lang=node
make build-runtime lang=go
```

### Running Tests

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests (requires Docker)
pytest tests/integration/ -v

# With coverage
pytest --cov=qm_code_runner tests/

# Specific language
pytest -k "test_python" -v
```

### Type Checking

```bash
mypy src/qm_code_runner --strict
```

### Linting

```bash
ruff check src/qm_code_runner
ruff format src/qm_code_runner
```

## Performance

### Typical Execution Times

```
Language    | Startup | Overhead | Total (small script)
------------|---------|----------|--------------------
Python 3.11 | 150ms   | 50ms     | ~200ms
Node.js 20  | 200ms   | 75ms     | ~275ms
Go 1.21     | 100ms   | 40ms     | ~140ms (compiled)
Rust 1.75   | 500ms   | 100ms    | ~600ms (compiled)
Deno 1.40   | 250ms   | 60ms     | ~310ms
Bun 1.0     | 100ms   | 40ms     | ~140ms
```

### Optimization Tips

- Pre-warm containers: Reduce startup overhead by keeping warm instances
- Batch executions: Group related executions to share containers
- Use binary format: Go and Rust pre-compiled faster than interpreted languages
- Minimize dependencies: Smaller code = faster startup
- Resource limits: Set realistic limits to prevent unnecessary overhead

## Troubleshooting

### Docker Connection Failed

```
Error: Cannot connect to Docker daemon at unix:///var/run/docker.sock
```

Solution:
1. Ensure Docker is running: `docker info`
2. Check socket permissions: `ls -la /var/run/docker.sock`
3. Add user to docker group: `sudo usermod -aG docker $USER`

### Out of Memory

```
exitCode: 137
error: "OOMKilled"
```

Solution: Increase `MAX_MEMORY_MB` or reduce concurrent executions

### Timeout Not Working

Ensure `DEFAULT_TIMEOUT` and `MAX_TIMEOUT` are set correctly. Timeouts are hard kills.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](https://github.com/quartermaster-ai/quartermaster/blob/main/CONTRIBUTING.md).

### Areas for Contribution

- Additional runtime support (PHP, Ruby, Java)
- Windows/macOS native support
- GPU execution support
- Persistent container pool optimization
- Integration tests with real workloads

## License

Apache License 2.0. See [LICENSE](./LICENSE) for details.

## Related Projects

- [Quartermaster](https://github.com/quartermaster-ai/quartermaster) — AI agent platform
- [E2B Code Interpreter](https://github.com/e2b-dev/code-interpreter) — Similar tool for AI agents
- [Docker SDK for Python](https://github.com/docker/docker-py)

## Support

- GitHub Issues: [Report a bug](https://github.com/quartermaster-ai/quartermaster/issues)
- Documentation: [Full docs](https://quartermaster.dev/docs/code-runner)
- Email: info@mindmade.io
