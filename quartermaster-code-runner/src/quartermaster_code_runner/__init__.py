"""quartermaster-code-runner: Secure sandboxed code execution service.

Executes untrusted code in isolated Docker containers with support
for Python, Node.js, Go, Rust, Deno, and Bun.
"""

from quartermaster_code_runner.config import ResourceLimits, Settings
from quartermaster_code_runner.errors import (
    CodeRunnerError,
    DockerError,
    ExecutionError,
    InvalidLanguageError,
    ResourceExhaustedError,
    RuntimeNotAvailableError,
    TimeoutError,
)
from quartermaster_code_runner.schemas import (
    CodeExecutionRequest,
    CodeExecutionResponse,
    HealthResponse,
)

__all__ = [
    "CodeExecutionRequest",
    "CodeExecutionResponse",
    "CodeRunnerError",
    "DockerError",
    "ExecutionError",
    "HealthResponse",
    "InvalidLanguageError",
    "ResourceExhaustedError",
    "ResourceLimits",
    "RuntimeNotAvailableError",
    "Settings",
    "TimeoutError",
]

__version__ = "0.6.1"
