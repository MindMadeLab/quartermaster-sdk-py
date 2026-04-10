"""Tests for qm_tools.base — AbstractTool and AbstractLocalTool."""

from typing import Any

import pytest

from qm_tools import AbstractTool, AbstractLocalTool, ToolDescriptor, ToolParameter, ToolResult


# --- Concrete test implementations ---


class EchoTool(AbstractTool):
    """Simple tool that echoes input."""

    def name(self) -> str:
        return "echo"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="message", description="Message to echo", type="string", required=True
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Echo a message",
            long_description="Returns the input message as output",
            version=self.version(),
            parameters=self.parameters(),
        )

    def run(self, **kwargs: Any) -> ToolResult:
        msg = kwargs.get("message", "")
        return ToolResult(success=True, data={"echo": msg})


class CalculatorTool(AbstractTool):
    """Calculator with validation."""

    def name(self) -> str:
        return "calculator"

    def version(self) -> str:
        return "2.0.0"

    def parameters(self) -> list[ToolParameter]:
        def positive(v):
            if v < 0:
                raise ValueError("Must be non-negative")
            return v

        return [
            ToolParameter(name="a", description="First number", type="number", required=True),
            ToolParameter(name="b", description="Second number", type="number", required=True),
            ToolParameter(
                name="precision",
                description="Decimal places",
                type="integer",
                default=2,
                validation=positive,
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Add two numbers",
            long_description="Adds two numbers with configurable precision",
            version=self.version(),
            parameters=self.parameters(),
        )

    def run(self, **kwargs: Any) -> ToolResult:
        a = kwargs["a"]
        b = kwargs["b"]
        precision = kwargs.get("precision", 2)
        return ToolResult(success=True, data={"result": round(a + b, precision)})


class LocalEchoTool(AbstractLocalTool):
    """Local tool that runs echo command."""

    def name(self) -> str:
        return "local_echo"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="text", description="Text to echo", type="string", required=True),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Echo via shell",
            long_description="Runs echo command locally",
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def prepare_command(self, **kwargs: Any) -> list[str]:
        return ["echo", kwargs["text"]]


class CustomTimeoutTool(AbstractLocalTool):
    """Local tool with custom timeout."""

    def name(self) -> str:
        return "slow"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return []

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Slow tool",
            long_description="Slow",
            version=self.version(),
        )

    def prepare_command(self, **kwargs: Any) -> list[str]:
        return ["sleep", "100"]

    def timeout(self) -> int:
        return 1

    def working_directory(self) -> str | None:
        return "/tmp"


# --- AbstractTool tests ---


class TestAbstractTool:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            AbstractTool()

    def test_echo_tool_basic(self):
        tool = EchoTool()
        assert tool.name() == "echo"
        assert tool.version() == "1.0.0"
        assert len(tool.parameters()) == 1
        assert tool.parameters()[0].name == "message"

    def test_echo_tool_run(self):
        tool = EchoTool()
        result = tool.run(message="hello")
        assert result.success
        assert result.data["echo"] == "hello"

    def test_echo_tool_info(self):
        tool = EchoTool()
        info = tool.info()
        assert info.name == "echo"
        assert info.version == "1.0.0"
        assert "Echo" in info.short_description

    def test_validate_params_missing_required(self):
        tool = EchoTool()
        errors = tool.validate_params()
        assert len(errors) == 1
        assert "message" in errors[0]

    def test_validate_params_ok(self):
        tool = EchoTool()
        errors = tool.validate_params(message="hi")
        assert errors == []

    def test_validate_params_custom_validation_pass(self):
        tool = CalculatorTool()
        errors = tool.validate_params(a=1, b=2, precision=3)
        assert errors == []

    def test_validate_params_custom_validation_fail(self):
        tool = CalculatorTool()
        errors = tool.validate_params(a=1, b=2, precision=-1)
        assert len(errors) == 1
        assert "precision" in errors[0]

    def test_safe_run_with_valid_params(self):
        tool = EchoTool()
        result = tool.safe_run(message="test")
        assert result.success
        assert result.data["echo"] == "test"

    def test_safe_run_with_missing_params(self):
        tool = EchoTool()
        result = tool.safe_run()
        assert not result.success
        assert "Missing required" in result.error

    def test_safe_run_with_invalid_validation(self):
        tool = CalculatorTool()
        result = tool.safe_run(a=1, b=2, precision=-5)
        assert not result.success
        assert "Validation failed" in result.error


# --- AbstractLocalTool tests ---


class TestAbstractLocalTool:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            AbstractLocalTool()

    def test_local_echo_run(self):
        tool = LocalEchoTool()
        result = tool.run(text="hello world")
        assert result.success
        assert "hello world" in result.data["stdout"]

    def test_local_echo_info(self):
        tool = LocalEchoTool()
        info = tool.info()
        assert info.is_local is True

    def test_default_timeout(self):
        tool = LocalEchoTool()
        assert tool.timeout() == 30

    def test_default_working_directory(self):
        tool = LocalEchoTool()
        assert tool.working_directory() is None

    def test_custom_timeout(self):
        tool = CustomTimeoutTool()
        assert tool.timeout() == 1

    def test_custom_working_directory(self):
        tool = CustomTimeoutTool()
        assert tool.working_directory() == "/tmp"

    def test_timeout_handling(self):
        tool = CustomTimeoutTool()
        result = tool.run()
        assert not result.success
        assert "timed out" in result.error

    def test_command_not_found(self):
        class BadTool(AbstractLocalTool):
            def name(self):
                return "bad"

            def version(self):
                return "1.0.0"

            def parameters(self):
                return []

            def info(self):
                return ToolDescriptor(
                    name="bad", short_description="Bad", long_description="Bad", version="1.0.0"
                )

            def prepare_command(self, **kwargs):
                return ["nonexistent_command_xyz_123"]

        tool = BadTool()
        result = tool.run()
        assert not result.success
        assert "not found" in result.error

    def test_command_failure_nonzero_exit(self):
        class FailTool(AbstractLocalTool):
            def name(self):
                return "fail"

            def version(self):
                return "1.0.0"

            def parameters(self):
                return []

            def info(self):
                return ToolDescriptor(
                    name="fail", short_description="Fail", long_description="Fail", version="1.0.0"
                )

            def prepare_command(self, **kwargs):
                return ["false"]  # always exits with 1

        tool = FailTool()
        result = tool.run()
        assert not result.success

    def test_prepare_command_error(self):
        class ExplodingTool(AbstractLocalTool):
            def name(self):
                return "explode"

            def version(self):
                return "1.0.0"

            def parameters(self):
                return []

            def info(self):
                return ToolDescriptor(
                    name="explode",
                    short_description="Boom",
                    long_description="Boom",
                    version="1.0.0",
                )

            def prepare_command(self, **kwargs):
                raise RuntimeError("Cannot build command")

        tool = ExplodingTool()
        result = tool.run()
        assert not result.success
        assert "prepare command" in result.error.lower()

    def test_parse_output_success(self):
        tool = LocalEchoTool()
        result = tool.parse_output("hello\n", "", 0)
        assert result.success
        assert result.data["stdout"] == "hello\n"
        assert result.data["returncode"] == 0

    def test_parse_output_failure(self):
        tool = LocalEchoTool()
        result = tool.parse_output("", "error msg", 1)
        assert not result.success
        assert "error msg" in result.error
