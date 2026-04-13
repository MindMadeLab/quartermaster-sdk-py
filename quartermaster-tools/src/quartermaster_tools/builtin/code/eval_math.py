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

from quartermaster_tools.decorator import tool


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
    """Recursively evaluate an AST node in a safe manner."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

    if isinstance(node, ast.UnaryOp):
        op_func = _UNARY_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_func(_safe_eval_node(node.operand))

    if isinstance(node, ast.BinOp):
        op_func = _BINARY_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        if isinstance(node.op, ast.Pow):
            if isinstance(right, (int, float)) and abs(right) > 10_000:
                raise ValueError(f"Exponent too large: {right}")
        return op_func(left, right)

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

    if isinstance(node, ast.Tuple) or isinstance(node, ast.List):
        return [_safe_eval_node(elt) for elt in node.elts]

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


@tool()
def eval_math(expression: str) -> dict:
    """Safely evaluate mathematical expressions.

    Uses AST parsing -- never calls exec() or eval() on arbitrary code.
    Supports arithmetic (+, -, *, /, //, %, **), comparisons (==, !=, <, <=, >, >=),
    functions (abs, round, min, max, sqrt, int, float), and constants (pi, e, inf).

    Args:
        expression: Mathematical expression to evaluate.
    """
    if not expression or not expression.strip():
        raise ValueError("Parameter 'expression' is required and must not be empty")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError(
            f"Expression too long: {len(expression)} chars (limit: {MAX_EXPRESSION_LENGTH})"
        )

    result = safe_eval(expression)
    return {"result": result, "expression": expression}


# Backward-compatible alias
EvalMathTool = eval_math
