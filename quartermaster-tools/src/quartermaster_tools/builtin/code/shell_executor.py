"""
ShellExecutorTool: Execute shell commands via subprocess.

Runs shell commands with a blocked-command safety list to prevent
obviously destructive operations.
"""

from __future__ import annotations

import shlex
import subprocess

from quartermaster_tools.decorator import tool
from quartermaster_tools.types import ToolResult


# Maximum allowed command length in characters.
MAX_COMMAND_LENGTH = 100_000

# Default execution timeout in seconds.
DEFAULT_TIMEOUT = 30

# Blocked command patterns -- exact matches and substrings that indicate
# destructive or dangerous operations.
_BLOCKED_COMMANDS: list[str] = [
    "rm -rf /",
    "rm -rf /*",
    "mkfs",
    "dd ",
    ":(){ :|:& };:",
]

# Additional blocked patterns checked as substrings.
_BLOCKED_SUBSTRINGS: list[str] = [
    "mkfs.",
    "mkfs ",
    "> /dev/sd",
    "dd ",
    "dd\t",
    "chmod -R 777 /",
    "chown -R",
    "rm -rf /",
    "rm -rf /*",
]


def _is_command_blocked(command: str) -> bool:
    """Check if a command matches any blocked pattern.

    Args:
        command: The shell command string to check.

    Returns:
        True if the command is blocked.
    """
    stripped = command.strip()

    # Direct match against blocked commands
    for blocked in _BLOCKED_COMMANDS:
        if stripped == blocked:
            return True

    # Substring match against dangerous patterns
    for pattern in _BLOCKED_SUBSTRINGS:
        if pattern in stripped:
            return True

    return False


@tool()
def shell_executor(
    command: str,
    timeout: int = DEFAULT_TIMEOUT,
    working_dir: str = None,
) -> ToolResult:
    """Execute shell commands in a subprocess.

    Runs shell commands via `sh -c <command>` in a subprocess. Captures stdout,
    stderr, and exit code. Blocks known-dangerous commands and enforces a timeout.

    Args:
        command: Shell command to execute.
        timeout: Maximum execution time in seconds.
        working_dir: Working directory for command execution.
    """
    if not command or not command.strip():
        return ToolResult(success=False, error="Parameter 'command' is required and must not be empty")
    if len(command) > MAX_COMMAND_LENGTH:
        return ToolResult(
            success=False,
            error=f"Command too long: {len(command)} chars (limit: {MAX_COMMAND_LENGTH})",
        )
    if _is_command_blocked(command):
        return ToolResult(
            success=False,
            error=f"Command blocked for safety: {shlex.quote(command)}",
        )

    cmd = ["sh", "-c", command]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=working_dir,
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
ShellExecutorTool = shell_executor
