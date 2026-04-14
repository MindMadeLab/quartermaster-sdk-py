"""
Code execution tools for quartermaster-tools.

Provides tools for executing code in subprocess-based sandboxes:
- python_executor: Execute Python code via subprocess
- shell_executor: Execute shell commands with blocked-command safety
- eval_math: Safe mathematical expression evaluation (no exec/eval)
- javascript_executor: Execute JavaScript via Node.js subprocess
"""

from quartermaster_tools.builtin.code.eval_math import eval_math, EvalMathTool
from quartermaster_tools.builtin.code.javascript_executor import (
    javascript_executor,
    JavaScriptExecutorTool,
)
from quartermaster_tools.builtin.code.python_executor import python_executor, PythonExecutorTool
from quartermaster_tools.builtin.code.shell_executor import shell_executor, ShellExecutorTool

__all__ = [
    "eval_math",
    "javascript_executor",
    "python_executor",
    "shell_executor",
    # Backward-compatible aliases
    "EvalMathTool",
    "JavaScriptExecutorTool",
    "PythonExecutorTool",
    "ShellExecutorTool",
]
