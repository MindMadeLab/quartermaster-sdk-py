"""
JavaScriptExecutorTool: Execute JavaScript code via Node.js subprocess.

Runs JavaScript code via ``node -e <code>`` with timeout enforcement.
Gracefully handles the case where Node.js is not installed.
"""

from __future__ import annotations

from typing import Any

from quartermaster_tools.base import AbstractLocalTool
from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolResult


# Maximum allowed code length in characters.
MAX_CODE_LENGTH = 100_000

# Default execution timeout in seconds.
DEFAULT_TIMEOUT = 30


class JavaScriptExecutorTool(AbstractLocalTool):
    """Execute JavaScript code in a Node.js subprocess.

    Security notes:
    - Code runs as an unprivileged subprocess of the current process.
    - A hard timeout prevents runaway processes.
    - Code length is capped to prevent memory exhaustion.
    - Gracefully handles missing Node.js installation.

    For production sandboxing, prefer quartermaster-code-runner which
    provides Docker-based isolation.
    """

    def __init__(self, timeout: int = DEFAULT_TIMEOUT) -> None:
        """Initialise the JavaScriptExecutorTool.

        Args:
            timeout: Maximum execution time in seconds.
        """
        self._timeout = timeout

    def name(self) -> str:
        """Return the tool name."""
        return "javascript_executor"

    def version(self) -> str:
        """Return the tool version."""
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        """Return parameter definitions for the tool."""
        return [
            ToolParameter(
                name="code",
                description="JavaScript source code to execute.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="timeout",
                description="Maximum execution time in seconds.",
                type="number",
                required=False,
                default=DEFAULT_TIMEOUT,
            ),
        ]

    def info(self) -> ToolDescriptor:
        """Return metadata describing this tool."""
        return ToolDescriptor(
            name=self.name(),
            short_description="Execute JavaScript code via Node.js.",
            long_description=(
                "Runs JavaScript code via `node -e <code>` in a subprocess. "
                "Captures stdout, stderr, and exit code. Enforces a timeout. "
                "Returns a clear error if Node.js is not installed."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def timeout(self) -> int:
        """Return the configured timeout."""
        return self._timeout

    def execute(self, code: str, timeout: int | None = None) -> ToolResult:
        """Execute JavaScript code and return the result.

        Args:
            code: JavaScript source code to execute.
            timeout: Optional timeout override in seconds.

        Returns:
            ToolResult with stdout, stderr, and exit_code in data.
        """
        old_timeout = self._timeout
        try:
            if timeout is not None:
                self._timeout = timeout
            return self.safe_run(code=code)
        finally:
            self._timeout = old_timeout

    def prepare_command(self, **kwargs: Any) -> list[str]:
        """Build the subprocess command for Node.js execution."""
        code: str = kwargs["code"]
        return ["node", "-e", code]

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute with code-length validation before running."""
        code: str = kwargs.get("code", "")
        if not code or not code.strip():
            return ToolResult(success=False, error="Parameter 'code' is required and must not be empty")
        if len(code) > MAX_CODE_LENGTH:
            return ToolResult(
                success=False,
                error=f"Code too long: {len(code)} chars (limit: {MAX_CODE_LENGTH})",
            )
        return super().run(**kwargs)
