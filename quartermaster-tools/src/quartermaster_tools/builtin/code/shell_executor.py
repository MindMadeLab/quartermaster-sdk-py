"""
ShellExecutorTool: Execute shell commands via subprocess.

Runs shell commands with a blocked-command safety list to prevent
obviously destructive operations.
"""

from __future__ import annotations

import shlex
from typing import Any

from quartermaster_tools.base import AbstractLocalTool
from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolResult


# Maximum allowed command length in characters.
MAX_COMMAND_LENGTH = 100_000

# Default execution timeout in seconds.
DEFAULT_TIMEOUT = 30

# Blocked command patterns — exact matches and substrings that indicate
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


class ShellExecutorTool(AbstractLocalTool):
    """Execute shell commands in a subprocess.

    Security notes:
    - A blocklist prevents obviously destructive commands.
    - A hard timeout prevents runaway processes.
    - Command length is capped.
    - Commands are executed via ``sh -c`` (not a login shell).

    The blocklist is NOT a security boundary. For production use,
    prefer Docker-based isolation via quartermaster-code-runner.
    """

    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT,
        working_dir: str | None = None,
    ) -> None:
        """Initialise the ShellExecutorTool.

        Args:
            timeout: Maximum execution time in seconds.
            working_dir: Default working directory for command execution.
        """
        self._timeout = timeout
        self._working_dir = working_dir

    def name(self) -> str:
        """Return the tool name."""
        return "shell_executor"

    def version(self) -> str:
        """Return the tool version."""
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        """Return parameter definitions for the tool."""
        return [
            ToolParameter(
                name="command",
                description="Shell command to execute.",
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
            ToolParameter(
                name="working_dir",
                description="Working directory for command execution.",
                type="string",
                required=False,
            ),
        ]

    def info(self) -> ToolDescriptor:
        """Return metadata describing this tool."""
        return ToolDescriptor(
            name=self.name(),
            short_description="Execute shell commands in a subprocess.",
            long_description=(
                "Runs shell commands via `sh -c <command>` in a subprocess. "
                "Captures stdout, stderr, and exit code. Blocks known-dangerous "
                "commands and enforces a timeout."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def timeout(self) -> int:
        """Return the configured timeout."""
        return self._timeout

    def working_directory(self) -> str | None:
        """Return the configured working directory."""
        return self._working_dir

    def execute(
        self,
        command: str,
        timeout: int | None = None,
        working_dir: str | None = None,
    ) -> ToolResult:
        """Execute a shell command and return the result.

        Args:
            command: Shell command to execute.
            timeout: Optional timeout override in seconds.
            working_dir: Optional working directory override.

        Returns:
            ToolResult with stdout, stderr, and exit_code in data.
        """
        old_timeout = self._timeout
        old_working_dir = self._working_dir
        try:
            if timeout is not None:
                self._timeout = timeout
            if working_dir is not None:
                self._working_dir = working_dir
            return self.safe_run(command=command)
        finally:
            self._timeout = old_timeout
            self._working_dir = old_working_dir

    def prepare_command(self, **kwargs: Any) -> list[str]:
        """Build the subprocess command for shell execution."""
        command: str = kwargs["command"]
        return ["sh", "-c", command]

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute with safety checks before running."""
        command: str = kwargs.get("command", "")
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
        return super().run(**kwargs)
