"""Unit tests for error types."""

from __future__ import annotations

from qm_code_runner.errors import (
    CodeRunnerError,
    DockerError,
    ExecutionError,
    InvalidLanguageError,
    ResourceExhaustedError,
    RuntimeNotAvailableError,
    TimeoutError,
)


class TestErrorHierarchy:
    """Tests for error class hierarchy."""

    def test_execution_error_is_code_runner_error(self) -> None:
        err = ExecutionError("test")
        assert isinstance(err, CodeRunnerError)

    def test_timeout_error_is_code_runner_error(self) -> None:
        err = TimeoutError("test")
        assert isinstance(err, CodeRunnerError)

    def test_resource_error_is_code_runner_error(self) -> None:
        err = ResourceExhaustedError("test")
        assert isinstance(err, CodeRunnerError)

    def test_docker_error_is_code_runner_error(self) -> None:
        err = DockerError("test")
        assert isinstance(err, CodeRunnerError)

    def test_runtime_not_available_is_code_runner_error(self) -> None:
        err = RuntimeNotAvailableError("test")
        assert isinstance(err, CodeRunnerError)

    def test_invalid_language_is_code_runner_error(self) -> None:
        err = InvalidLanguageError("test")
        assert isinstance(err, CodeRunnerError)


class TestExecutionError:
    """Tests for ExecutionError attributes."""

    def test_with_all_fields(self) -> None:
        err = ExecutionError(
            "failed",
            exit_code=1,
            stdout="out",
            stderr="err",
        )
        assert str(err) == "failed"
        assert err.exit_code == 1
        assert err.stdout == "out"
        assert err.stderr == "err"

    def test_defaults(self) -> None:
        err = ExecutionError("msg")
        assert err.exit_code is None
        assert err.stdout is None
        assert err.stderr is None


class TestTimeoutError:
    """Tests for TimeoutError attributes."""

    def test_with_all_fields(self) -> None:
        err = TimeoutError("timed out", duration=10.5, stdout="partial")
        assert str(err) == "timed out"
        assert err.duration == 10.5
        assert err.stdout == "partial"

    def test_defaults(self) -> None:
        err = TimeoutError("msg")
        assert err.duration is None
        assert err.stdout is None


class TestResourceExhaustedError:
    """Tests for ResourceExhaustedError attributes."""

    def test_with_resource_type(self) -> None:
        err = ResourceExhaustedError("OOM", resource_type="memory")
        assert err.resource_type == "memory"

    def test_defaults(self) -> None:
        err = ResourceExhaustedError("msg")
        assert err.resource_type is None
