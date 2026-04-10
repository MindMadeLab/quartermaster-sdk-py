"""
EvalMathTool: Safe mathematical expression evaluation.

Uses AST parsing to evaluate mathematical expressions without
exec/eval of arbitrary code.  Supports arithmetic operators,
comparisons, and a small set of safe built-in functions.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from quartermaster_tools.base import AbstractTool
from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolResult


# Maximum expression length in characters.
MAX_EXPRESSION_LENGTH = 10_000

# Maximum AST node depth to prevent stack overflow via deeply nested expressions.
MAX_AST_DEPTH = 50

# Supported binary operators.
_BINARY_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Supported unary operators.
_UNARY_OPS: dict[type, Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Supported comparison operators.
_COMPARE_OPS: dict[type, Any] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}

# Allowed function names and their implementations.
_SAFE_FUNCTIONS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "int": int,
    "float": float,
}


def _get_ast_depth(node: ast.AST, current: int = 0) -> int:
    """Calculate the maximum depth of an AST tree."""
    max_depth = current
    for child in ast.iter_child_nodes(node):
        child_depth = _get_ast_depth(child, current + 1)
        if child_depth > max_depth:
            max_depth = child_depth
    return max_depth


def _safe_eval_node(node: ast.AST) -> Any:
    """Recursively evaluate an AST node in a safe manner.

    Only supports numeric literals, arithmetic, comparisons,
    and a whitelist of safe functions.

    Args:
        node: An AST node to evaluate.

    Returns:
        The computed value.

    Raises:
        ValueError: If the node type is not supported.
    """
    # Numeric and string literals
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

    # Unary operations: -x, +x
    if isinstance(node, ast.UnaryOp):
        op_func = _UNARY_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_func(_safe_eval_node(node.operand))

    # Binary operations: x + y, x ** y, etc.
    if isinstance(node, ast.BinOp):
        op_func = _BINARY_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        # Prevent exponentiation bombs
        if isinstance(node.op, ast.Pow):
            if isinstance(right, (int, float)) and abs(right) > 10_000:
                raise ValueError(f"Exponent too large: {right}")
        return op_func(left, right)

    # Comparison operations: x < y, x == y, etc.
    if isinstance(node, ast.Compare):
        left = _safe_eval_node(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            op_func = _COMPARE_OPS.get(type(op))
            if op_func is None:
                raise ValueError(f"Unsupported comparison: {type(op).__name__}")
            right = _safe_eval_node(comparator)
            if not op_func(left, right):
                return False
            left = right
        return True

    # Function calls: abs(x), sqrt(x), min(x, y), etc.
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are supported (e.g., abs, sqrt)")
        func_name = node.func.id
        func = _SAFE_FUNCTIONS.get(func_name)
        if func is None:
            raise ValueError(
                f"Unknown function: {func_name!r}. "
                f"Allowed: {sorted(_SAFE_FUNCTIONS.keys())}"
            )
        args = [_safe_eval_node(arg) for arg in node.args]
        if node.keywords:
            raise ValueError("Keyword arguments are not supported")
        return func(*args)

    # Tuple / list for multi-arg functions like min(1, 2, 3)
    if isinstance(node, ast.Tuple) or isinstance(node, ast.List):
        return [_safe_eval_node(elt) for elt in node.elts]

    # Name references — only allow known constants
    if isinstance(node, ast.Name):
        name_constants: dict[str, Any] = {
            "pi": math.pi,
            "e": math.e,
            "inf": math.inf,
            "True": True,
            "False": False,
        }
        if node.id in name_constants:
            return name_constants[node.id]
        raise ValueError(
            f"Unknown name: {node.id!r}. Variables are not supported."
        )

    # Expression wrapper
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)

    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def safe_eval(expression: str) -> Any:
    """Safely evaluate a mathematical expression.

    Args:
        expression: A mathematical expression string.

    Returns:
        The computed result.

    Raises:
        ValueError: If the expression is invalid or uses unsupported features.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {e}") from e

    depth = _get_ast_depth(tree)
    if depth > MAX_AST_DEPTH:
        raise ValueError(f"Expression too deeply nested (depth {depth}, max {MAX_AST_DEPTH})")

    return _safe_eval_node(tree)


class EvalMathTool(AbstractTool):
    """Safely evaluate mathematical expressions.

    Uses AST parsing — never calls exec() or eval() on arbitrary code.
    Supports:
    - Arithmetic: +, -, *, /, //, %, **
    - Comparisons: ==, !=, <, <=, >, >=
    - Functions: abs, round, min, max, sqrt, int, float
    - Constants: pi, e, inf
    """

    def name(self) -> str:
        """Return the tool name."""
        return "eval_math"

    def version(self) -> str:
        """Return the tool version."""
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        """Return parameter definitions for the tool."""
        return [
            ToolParameter(
                name="expression",
                description="Mathematical expression to evaluate.",
                type="string",
                required=True,
            ),
        ]

    def info(self) -> ToolDescriptor:
        """Return metadata describing this tool."""
        return ToolDescriptor(
            name=self.name(),
            short_description="Safely evaluate mathematical expressions.",
            long_description=(
                "Evaluates mathematical expressions using AST parsing. "
                "Supports arithmetic, comparisons, and safe built-in functions "
                "(abs, round, min, max, sqrt). Never uses exec/eval."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def evaluate(self, expression: str) -> ToolResult:
        """Evaluate a mathematical expression.

        Args:
            expression: Mathematical expression string.

        Returns:
            ToolResult with the computed result in data["result"].
        """
        return self.safe_run(expression=expression)

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the math evaluation.

        Args:
            expression: The expression to evaluate.

        Returns:
            ToolResult with computed value or error.
        """
        expression: str = kwargs.get("expression", "")
        if not expression or not expression.strip():
            return ToolResult(success=False, error="Parameter 'expression' is required and must not be empty")
        if len(expression) > MAX_EXPRESSION_LENGTH:
            return ToolResult(
                success=False,
                error=f"Expression too long: {len(expression)} chars (limit: {MAX_EXPRESSION_LENGTH})",
            )

        try:
            result = safe_eval(expression)
        except ValueError as e:
            return ToolResult(success=False, error=str(e))
        except (TypeError, ArithmeticError, OverflowError) as e:
            return ToolResult(success=False, error=f"Evaluation error: {e}")

        return ToolResult(
            success=True,
            data={"result": result, "expression": expression},
        )
