"""
PythonExecutorTool: Execute Python code via subprocess.

Runs Python code in a subprocess with timeout enforcement.
The code is passed via ``python3 -c <code>`` and stdout/stderr
are captured and returned.
"""

from __future__ import annotations

import subprocess

from quartermaster_tools.decorator import tool
from quartermaster_tools.types import ToolResult


# Maximum allowed code length in characters.
MAX_CODE_LENGTH = 100_000

# Default execution timeout in seconds.
DEFAULT_TIMEOUT = 30


@tool()
def python_executor(code: str, timeout: int = DEFAULT_TIMEOUT) -> ToolResult:
    """Execute Python code in a subprocess.

    Runs Python code via `python3 -c <code>` in a subprocess. Captures stdout,
    stderr, and exit code. Enforces a timeout to prevent runaway execution.

    Args:
        code: Python source code to execute.
        timeout: Maximum execution time in seconds.
    """
    if not code or not code.strip():
        return ToolResult(success=False, error="Parameter 'code' is required and must not be empty")
    if len(code) > MAX_CODE_LENGTH:
        return ToolResult(
            success=False,
            error=f"Code too long: {len(code)} chars (limit: {MAX_CODE_LENGTH})",
        )

    cmd = ["python3", "-c", code]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        data = {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
        if result.returncode != 0:
            return ToolResult(
                success=False,
                error=result.stderr or f"Command exited with code {result.returncode}",
                data=data,
            )
        return ToolResult(success=True, data=data)
    except subprocess.TimeoutExpired:
        return ToolResult(success=False, error=f"Command timed out after {timeout} seconds")
    except FileNotFoundError:
        return ToolResult(success=False, error=f"Command not found: {cmd[0]}")
    except OSError as e:
        return ToolResult(success=False, error=f"OS error: {e}")


# Backward-compatible alias
PythonExecutorTool = python_executor
