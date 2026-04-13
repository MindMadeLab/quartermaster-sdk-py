"""
Abstract base classes for tool definitions.

Provides the core abstractions that all tools must implement:
- AbstractTool: base interface for any tool
- AbstractLocalTool: base for tools that run local subprocess commands
"""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from typing import Any

from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolResult


class AbstractTool(ABC):
    """Base class for all tools.

    A tool is a unit of functionality that an AI agent can invoke.
    Each tool declares its name, version, parameters, and execution logic.

    Subclasses must implement:
        - name(): return tool identifier
        - version(): return semantic version string
        - parameters(): return list of ToolParameter definitions
        - info(): return ToolDescriptor metadata
        - run(**kwargs): execute the tool and return ToolResult
    """

    @abstractmethod
    def name(self) -> str:
        """Return the unique name of this tool."""
        ...

    def version(self) -> str:
        """Return the semantic version of this tool (e.g., '1.0.0')."""
        return "1.0.0"

    @abstractmethod
    def parameters(self) -> list[ToolParameter]:
        """Return the list of parameters this tool accepts."""
        ...

    @abstractmethod
    def info(self) -> ToolDescriptor:
        """Return metadata describing this tool."""
        ...

    @abstractmethod
    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with the given keyword arguments.

        Args:
            **kwargs: Tool-specific parameters as defined by parameters().

        Returns:
            ToolResult with success/error status and data.
        """
        ...

    def validate_params(self, **kwargs: Any) -> list[str]:
        """Validate that required parameters are present and types match.

        Returns a list of error messages. Empty list means validation passed.
        """
        errors: list[str] = []
        for param in self.parameters():
            if param.required and param.name not in kwargs:
                errors.append(f"Missing required parameter: {param.name}")
            if param.name in kwargs and param.validation is not None:
                try:
                    param.validation(kwargs[param.name])
                except (ValueError, TypeError) as e:
                    errors.append(f"Validation failed for '{param.name}': {e}")
        return errors

    def safe_run(self, **kwargs: Any) -> ToolResult:
        """Run with parameter validation. Returns error result if validation fails."""
        errors = self.validate_params(**kwargs)
        if errors:
            return ToolResult(success=False, error="; ".join(errors))
        return self.run(**kwargs)


class AbstractLocalTool(AbstractTool):
    """Base class for tools that execute local subprocess commands.

    Subclasses must implement:
        - prepare_command(**kwargs): return the command list to execute
        - parse_output(stdout, stderr, returncode): convert output to ToolResult

    Optionally override:
        - timeout(): return max execution time in seconds (default: 30)
        - working_directory(): return cwd for the subprocess (default: None)
    """

    @abstractmethod
    def prepare_command(self, **kwargs: Any) -> list[str]:
        """Build the command-line arguments for subprocess execution.

        Returns:
            List of strings to pass to subprocess.run().
        """
        ...

    def parse_output(self, stdout: str, stderr: str, returncode: int) -> ToolResult:
        """Convert subprocess output to a ToolResult.

        Override this for custom output parsing. Default returns stdout as data.
        """
        if returncode != 0:
            return ToolResult(
                success=False,
                error=stderr or f"Command exited with code {returncode}",
                data={"stdout": stdout, "stderr": stderr, "returncode": returncode},
            )
        return ToolResult(
            success=True,
            data={"stdout": stdout, "stderr": stderr, "returncode": returncode},
        )

    def timeout(self) -> int:
        """Maximum execution time in seconds. Override to customize."""
        return 30

    def working_directory(self) -> str | None:
        """Working directory for the subprocess. Override to customize."""
        return None

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the local command and return the result."""
        try:
            cmd = self.prepare_command(**kwargs)
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to prepare command: {e}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout(),
                cwd=self.working_directory(),
            )
            return self.parse_output(result.stdout, result.stderr, result.returncode)
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"Command timed out after {self.timeout()} seconds",
            )
        except FileNotFoundError:
            return ToolResult(
                success=False,
                error=f"Command not found: {cmd[0]}",
            )
        except OSError as e:
            return ToolResult(success=False, error=f"OS error: {e}")
