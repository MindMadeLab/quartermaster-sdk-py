"""Exception types for code runner errors."""


class CodeRunnerError(Exception):
    """Base exception for all code runner errors."""

    pass


class ExecutionError(CodeRunnerError):
    """Raised when code execution fails."""

    def __init__(
        self,
        message: str,
        exit_code: int | None = None,
        stdout: str | None = None,
        stderr: str | None = None,
    ) -> None:
        """Initialize execution error.

        Args:
            message: Error message.
            exit_code: Exit code from execution.
            stdout: Standard output captured before error.
            stderr: Standard error output.
        """
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(message)


class TimeoutError(CodeRunnerError):
    """Raised when code execution times out."""

    def __init__(
        self,
        message: str,
        duration: float | None = None,
        stdout: str | None = None,
    ) -> None:
        """Initialize timeout error.

        Args:
            message: Error message.
            duration: Duration before timeout.
            stdout: Partial output captured before timeout.
        """
        self.duration = duration
        self.stdout = stdout
        super().__init__(message)


class ResourceExhaustedError(CodeRunnerError):
    """Raised when execution exceeds resource limits."""

    def __init__(
        self,
        message: str,
        resource_type: str | None = None,
    ) -> None:
        """Initialize resource exhausted error.

        Args:
            message: Error message.
            resource_type: Type of resource (memory, cpu, disk).
        """
        self.resource_type = resource_type
        super().__init__(message)


class DockerError(CodeRunnerError):
    """Raised when Docker operations fail."""

    pass


class RuntimeNotAvailableError(CodeRunnerError):
    """Raised when requested runtime is not available."""

    pass


class InvalidLanguageError(CodeRunnerError):
    """Raised when language is not supported."""

    pass
