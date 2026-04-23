"""Tests for code execution tools."""

from __future__ import annotations

import math
import shutil
from unittest.mock import patch

import pytest

from quartermaster_tools.builtin.code import (
    eval_math,
    javascript_executor,
    python_executor,
    shell_executor,
)
from quartermaster_tools.types import ToolResult


# ---------------------------------------------------------------------------
# python_executor
# ---------------------------------------------------------------------------


class TestPythonExecutorTool:
    """Tests for python_executor."""

    def setup_method(self) -> None:
        self.tool = python_executor

    def test_hello_world(self) -> None:
        """Execute a simple print statement."""
        result = self.tool.run(code="print('hello world')", timeout=10)
        assert result.success
        assert "hello world" in result.data["stdout"]

    def test_math_expression(self) -> None:
        """Execute a math calculation."""
        result = self.tool.run(code="print(2 + 3)", timeout=10)
        assert result.success
        assert "5" in result.data["stdout"]

    def test_multiline_code(self) -> None:
        """Execute multiline Python code."""
        code = "x = 10\ny = 20\nprint(x + y)"
        result = self.tool.run(code=code, timeout=10)
        assert result.success
        assert "30" in result.data["stdout"]

    def test_syntax_error(self) -> None:
        """Syntax errors produce stderr and non-zero exit code."""
        result = self.tool.run(code="def incomplete(", timeout=10)
        assert not result.success
        assert result.data["returncode"] != 0

    def test_runtime_error(self) -> None:
        """Runtime errors produce stderr."""
        result = self.tool.run(code="1/0", timeout=10)
        assert not result.success
        assert "ZeroDivisionError" in result.data["stderr"]

    def test_timeout(self) -> None:
        """Long-running code is killed by timeout."""
        result = self.tool.run(code="import time; time.sleep(60)", timeout=1)
        assert not result.success
        assert "timed out" in result.error.lower()

    def test_empty_code_rejected(self) -> None:
        """Empty code string is rejected."""
        result = self.tool.run(code="")
        assert not result.success
        assert "required" in result.error.lower()

    def test_whitespace_only_rejected(self) -> None:
        """Whitespace-only code is rejected."""
        result = self.tool.run(code="   \n  ")
        assert not result.success

    def test_code_too_long(self) -> None:
        """Code exceeding the length limit is rejected."""
        result = self.tool.run(code="x" * 200_000)
        assert not result.success
        assert "too long" in result.error.lower()

    def test_info_descriptor(self) -> None:
        """Info returns valid ToolDescriptor."""
        info = self.tool.info()
        assert info.name == "python_executor"
        assert info.is_local is False

    def test_exit_code_captured(self) -> None:
        """Exit code is captured in result data."""
        result = self.tool.run(code="import sys; sys.exit(42)", timeout=10)
        assert not result.success
        assert result.data["returncode"] == 42


# ---------------------------------------------------------------------------
# shell_executor
# ---------------------------------------------------------------------------


class TestShellExecutorTool:
    """Tests for shell_executor."""

    def setup_method(self) -> None:
        self.tool = shell_executor

    def test_echo(self) -> None:
        """Simple echo command."""
        result = self.tool.run(command="echo hello", timeout=10)
        assert result.success
        assert "hello" in result.data["stdout"]

    def test_ls(self) -> None:
        """ls command returns output."""
        result = self.tool.run(command="ls /", timeout=10)
        assert result.success
        assert len(result.data["stdout"]) > 0

    def test_working_dir(self) -> None:
        """working_dir parameter is respected."""
        result = self.tool.run(command="pwd", working_dir="/tmp", timeout=10)
        assert result.success
        # macOS resolves /tmp -> /private/tmp
        assert "tmp" in result.data["stdout"]

    def test_blocked_rm_rf_root(self) -> None:
        """rm -rf / is blocked."""
        result = self.tool.run(command="rm -rf /", timeout=10)
        assert not result.success
        assert "blocked" in result.error.lower()

    def test_blocked_rm_rf_star(self) -> None:
        """rm -rf /* is blocked."""
        result = self.tool.run(command="rm -rf /*", timeout=10)
        assert not result.success
        assert "blocked" in result.error.lower()

    def test_blocked_mkfs(self) -> None:
        """mkfs commands are blocked."""
        result = self.tool.run(command="mkfs.ext4 /dev/sda1", timeout=10)
        assert not result.success
        assert "blocked" in result.error.lower()

    def test_blocked_dd(self) -> None:
        """dd commands are blocked."""
        result = self.tool.run(command="dd if=/dev/zero of=/dev/sda", timeout=10)
        assert not result.success
        assert "blocked" in result.error.lower()

    def test_blocked_fork_bomb(self) -> None:
        """Fork bomb is blocked."""
        result = self.tool.run(command=":(){ :|:& };:", timeout=10)
        assert not result.success
        assert "blocked" in result.error.lower()

    def test_timeout(self) -> None:
        """Long-running commands are killed by timeout."""
        result = self.tool.run(command="sleep 60", timeout=1)
        assert not result.success
        assert "timed out" in result.error.lower()

    def test_empty_command_rejected(self) -> None:
        """Empty command is rejected."""
        result = self.tool.run(command="")
        assert not result.success

    def test_nonzero_exit(self) -> None:
        """Non-zero exit code is captured."""
        result = self.tool.run(command="exit 1", timeout=10)
        assert not result.success
        assert result.data["returncode"] == 1

    def test_pipe_commands(self) -> None:
        """Pipe commands work."""
        result = self.tool.run(command="echo 'abc def' | wc -w", timeout=10)
        assert result.success
        assert "2" in result.data["stdout"]


# ---------------------------------------------------------------------------
# eval_math
# ---------------------------------------------------------------------------


class TestEvalMathTool:
    """Tests for eval_math."""

    def setup_method(self) -> None:
        self.tool = eval_math

    def test_addition(self) -> None:
        """Simple addition."""
        result = self.tool.run(expression="2 + 3")
        assert result.success
        assert result.data["result"] == 5

    def test_subtraction(self) -> None:
        """Subtraction."""
        result = self.tool.run(expression="10 - 4")
        assert result.success
        assert result.data["result"] == 6

    def test_multiplication(self) -> None:
        """Multiplication."""
        result = self.tool.run(expression="6 * 7")
        assert result.success
        assert result.data["result"] == 42

    def test_division(self) -> None:
        """Division returns float."""
        result = self.tool.run(expression="10 / 3")
        assert result.success
        assert abs(result.data["result"] - 3.333333) < 0.001

    def test_floor_division(self) -> None:
        """Floor division."""
        result = self.tool.run(expression="10 // 3")
        assert result.success
        assert result.data["result"] == 3

    def test_modulo(self) -> None:
        """Modulo operator."""
        result = self.tool.run(expression="10 % 3")
        assert result.success
        assert result.data["result"] == 1

    def test_exponentiation(self) -> None:
        """Power operator."""
        result = self.tool.run(expression="2 ** 10")
        assert result.success
        assert result.data["result"] == 1024

    def test_comparison_lt(self) -> None:
        """Less-than comparison."""
        result = self.tool.run(expression="3 < 5")
        assert result.success
        assert result.data["result"] is True

    def test_comparison_eq(self) -> None:
        """Equality comparison."""
        result = self.tool.run(expression="3 == 3")
        assert result.success
        assert result.data["result"] is True

    def test_comparison_chain(self) -> None:
        """Chained comparison."""
        result = self.tool.run(expression="1 < 2 < 3")
        assert result.success
        assert result.data["result"] is True

    def test_abs_function(self) -> None:
        """abs() function."""
        result = self.tool.run(expression="abs(-5)")
        assert result.success
        assert result.data["result"] == 5

    def test_round_function(self) -> None:
        """round() function."""
        result = self.tool.run(expression="round(3.14159, 2)")
        assert result.success
        assert result.data["result"] == 3.14

    def test_min_max(self) -> None:
        """min() and max() functions."""
        result = self.tool.run(expression="min(3, 1, 2)")
        assert result.success
        assert result.data["result"] == 1

        result = self.tool.run(expression="max(3, 1, 2)")
        assert result.success
        assert result.data["result"] == 3

    def test_sqrt(self) -> None:
        """sqrt() function."""
        result = self.tool.run(expression="sqrt(16)")
        assert result.success
        assert result.data["result"] == 4.0

    def test_pi_constant(self) -> None:
        """pi constant is available."""
        result = self.tool.run(expression="pi")
        assert result.success
        assert abs(result.data["result"] - math.pi) < 1e-10

    def test_complex_expression(self) -> None:
        """Complex nested expression."""
        result = self.tool.run(expression="sqrt(abs(-16)) + round(3.7)")
        assert result.success
        assert result.data["result"] == 8.0

    def test_invalid_syntax(self) -> None:
        """Invalid syntax is rejected."""
        result = self.tool.run(expression="2 +")
        assert not result.success

    def test_unknown_function(self) -> None:
        """Unknown functions are rejected."""
        result = self.tool.run(expression="os.system('ls')")
        assert not result.success

    def test_injection_exec(self) -> None:
        """exec() calls are blocked."""
        result = self.tool.run(expression="exec('import os')")
        assert not result.success

    def test_injection_eval(self) -> None:
        """eval() calls are blocked."""
        result = self.tool.run(expression="eval('1+1')")
        assert not result.success

    def test_injection_import(self) -> None:
        """__import__ is blocked."""
        result = self.tool.run(expression="__import__('os')")
        assert not result.success

    def test_injection_attribute_access(self) -> None:
        """Attribute access is blocked."""
        result = self.tool.run(expression="().__class__.__bases__")
        assert not result.success

    def test_empty_expression_rejected(self) -> None:
        """Empty expression is rejected."""
        result = self.tool.run(expression="")
        assert not result.success

    def test_exponent_too_large(self) -> None:
        """Extremely large exponents are rejected."""
        result = self.tool.run(expression="2 ** 100000")
        assert not result.success
        assert "too large" in result.error.lower()

    def test_division_by_zero(self) -> None:
        """Division by zero returns error."""
        result = self.tool.run(expression="1 / 0")
        assert not result.success

    def test_negative_unary(self) -> None:
        """Unary negation works."""
        result = self.tool.run(expression="-5 + 3")
        assert result.success
        assert result.data["result"] == -2

    def test_variable_names_rejected(self) -> None:
        """Arbitrary variable names are rejected."""
        result = self.tool.run(expression="x + 1")
        assert not result.success
        assert "Unknown name" in result.error


# ---------------------------------------------------------------------------
# javascript_executor
# ---------------------------------------------------------------------------


class TestJavaScriptExecutorTool:
    """Tests for javascript_executor."""

    def setup_method(self) -> None:
        self.tool = javascript_executor

    @pytest.mark.skipif(
        shutil.which("node") is None,
        reason="Node.js not installed",
    )
    def test_hello_world(self) -> None:
        """Execute a simple console.log."""
        result = self.tool.run(code="console.log('hello world')", timeout=10)
        assert result.success
        assert "hello world" in result.data["stdout"]

    @pytest.mark.skipif(
        shutil.which("node") is None,
        reason="Node.js not installed",
    )
    def test_math(self) -> None:
        """Execute JS math."""
        result = self.tool.run(code="console.log(2 + 3)", timeout=10)
        assert result.success
        assert "5" in result.data["stdout"]

    @pytest.mark.skipif(
        shutil.which("node") is None,
        reason="Node.js not installed",
    )
    def test_syntax_error(self) -> None:
        """JS syntax errors are captured."""
        result = self.tool.run(code="function(", timeout=10)
        assert not result.success

    def test_missing_node_handling(self) -> None:
        """Clear error when node is not found."""
        with patch(
            "quartermaster_tools.builtin.code.javascript_executor.subprocess.run",
            side_effect=FileNotFoundError("node not found"),
        ):
            result = self.tool.run(code="console.log(1)")
        assert not result.success
        assert "not found" in result.error.lower()

    def test_empty_code_rejected(self) -> None:
        """Empty code is rejected."""
        result = self.tool.run(code="")
        assert not result.success

    def test_code_too_long(self) -> None:
        """Code exceeding the length limit is rejected."""
        result = self.tool.run(code="x" * 200_000)
        assert not result.success
        assert "too long" in result.error.lower()

    @pytest.mark.skipif(
        shutil.which("node") is None,
        reason="Node.js not installed",
    )
    def test_timeout(self) -> None:
        """Long-running JS code is killed by timeout."""
        code = "while(true){}"
        result = self.tool.run(code=code, timeout=1)
        assert not result.success
        assert "timed out" in result.error.lower()

    def test_info_descriptor(self) -> None:
        """Info returns valid ToolDescriptor."""
        info = self.tool.info()
        assert info.name == "javascript_executor"
        assert info.is_local is False
