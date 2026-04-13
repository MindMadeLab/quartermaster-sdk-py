"""
data_filter: Filter, sort, and limit structured data.

Operates on lists of dicts (tabular data). Supports simple Python filter
expressions evaluated safely per row, sorting by key, and result limiting.
"""

from __future__ import annotations

import ast
import operator
from typing import Any

from quartermaster_tools.decorator import tool

# -- Lightweight AST-based safe evaluator for filter expressions ----------
# This is a self-contained version for quartermaster-tools (no cross-package
# dependency on quartermaster-nodes).  Only supports the subset needed for
# row-level filter expressions like ``row['age'] > 18``.

_FILTER_SAFE_FUNCS: dict[str, Any] = {
    "len": len, "int": int, "float": float, "str": str, "bool": bool,
    "abs": abs, "min": min, "max": max, "round": round, "sum": sum,
    "sorted": sorted, "any": any, "all": all,
    "isinstance": isinstance, "hasattr": hasattr,
    "True": True, "False": False, "None": None,
}

_FILTER_SAFE_METHODS: frozenset[str] = frozenset({
    "upper", "lower", "strip", "startswith", "endswith", "replace", "split",
    "find", "count", "isdigit", "isalpha", "get", "keys", "values", "items",
    "index", "copy",
})

_FILTER_BLOCKED: frozenset[str] = frozenset({
    "__import__", "eval", "exec", "compile", "open",
    "getattr", "setattr", "delattr", "globals", "locals", "vars", "dir",
    "type", "breakpoint", "exit", "quit", "input", "print", "__builtins__",
})

_BIN_OPS: dict[type, Any] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_CMP_OPS: dict[type, Any] = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne,
    ast.Lt: operator.lt, ast.LtE: operator.le,
    ast.Gt: operator.gt, ast.GtE: operator.ge,
    ast.Is: operator.is_, ast.IsNot: operator.is_not,
    ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b,
}
_UNARY_OPS: dict[type, Any] = {
    ast.UAdd: operator.pos, ast.USub: operator.neg, ast.Not: operator.not_,
}


class _FilterEval:
    """Minimal AST walker for filter expressions."""

    def __init__(self, ctx: dict[str, Any]) -> None:
        self._ctx = ctx

    def visit(self, node: ast.AST) -> Any:  # noqa: C901
        if isinstance(node, ast.Expression):
            return self.visit(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            n = node.id
            if n in _FILTER_BLOCKED:
                raise ValueError(f"Access to '{n}' is not allowed")
            if n in _FILTER_SAFE_FUNCS:
                return _FILTER_SAFE_FUNCS[n]
            if n in self._ctx:
                return self._ctx[n]
            raise ValueError(f"Unknown variable '{n}'")
        if isinstance(node, ast.Subscript):
            obj = self.visit(node.value)
            key = self.visit(node.slice)
            return obj[key]
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise ValueError(f"Private attribute '{node.attr}' is not allowed")
            obj = self.visit(node.value)
            val = getattr(obj, node.attr)
            if callable(val) and node.attr not in _FILTER_SAFE_METHODS:
                raise ValueError(f"Method '{node.attr}' is not allowed")
            return val
        if isinstance(node, ast.Call):
            func = self.visit(node.func)
            args = [self.visit(a) for a in node.args]
            kwargs = {kw.arg: self.visit(kw.value) for kw in node.keywords if kw.arg}
            return func(*args, **kwargs)
        if isinstance(node, ast.BinOp):
            op_fn = _BIN_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return op_fn(self.visit(node.left), self.visit(node.right))
        if isinstance(node, ast.UnaryOp):
            op_fn = _UNARY_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")
            return op_fn(self.visit(node.operand))
        if isinstance(node, ast.Compare):
            left = self.visit(node.left)
            for op, comp in zip(node.ops, node.comparators):
                op_fn = _CMP_OPS.get(type(op))
                if op_fn is None:
                    raise ValueError(f"Unsupported comparison: {type(op).__name__}")
                right = self.visit(comp)
                if not op_fn(left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                result: Any = True
                for v in node.values:
                    result = self.visit(v)
                    if not result:
                        return result
                return result
            if isinstance(node.op, ast.Or):
                result = False
                for v in node.values:
                    result = self.visit(v)
                    if result:
                        return result
                return result
        if isinstance(node, ast.IfExp):
            return self.visit(node.body) if self.visit(node.test) else self.visit(node.orelse)
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self.visit(e) for e in node.elts]
        raise ValueError(f"Unsupported expression: {type(node).__name__}")


def _safe_eval_filter(expression: str, context: dict[str, Any]) -> Any:
    """Safely evaluate a filter expression using AST walking.

    Never calls Python's ``eval()`` or ``exec()``.
    """
    if not expression or not expression.strip():
        raise ValueError("Expression is empty")
    if len(expression) > 10_000:
        raise ValueError("Expression too long")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression syntax: {exc}") from exc
    return _FilterEval(dict(context)).visit(tree)


def _validate_expression(expr: str) -> str | None:
    """Return an error message if *expr* uses blocked constructs, else None."""
    for name in _FILTER_BLOCKED:
        if name in expr:
            return f"Expression contains blocked name: {name!r}"
    if "__" in expr:
        return "Dunder attributes are not allowed in filter expressions"
    return None


def _filter_data(
    data: list[dict[str, Any]],
    filter_expression: str | None = None,
    sort_by: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Apply filter, sort, and limit to *data*.

    Args:
        data: List of dicts to process.
        filter_expression: Python expression with ``row`` variable in scope.
        sort_by: Dict key to sort results by.
        limit: Maximum number of results.

    Returns:
        Filtered, sorted, and limited list of dicts.

    Raises:
        ValueError: On invalid data, blocked expression, or evaluation error.
    """
    if not isinstance(data, list):
        raise ValueError("data must be a list")

    result = list(data)

    # Filter
    if filter_expression is not None:
        error = _validate_expression(filter_expression)
        if error:
            raise ValueError(error)

        filtered = []
        for row in result:
            try:
                if _safe_eval_filter(filter_expression, {"row": row}):
                    filtered.append(row)
            except Exception as exc:
                raise ValueError(
                    f"Error evaluating expression on row {row!r}: {exc}"
                ) from exc
        result = filtered

    # Sort
    if sort_by is not None:
        try:
            result.sort(key=lambda r: r.get(sort_by, ""))
        except TypeError:
            # Fall back to string comparison for mixed types
            result.sort(key=lambda r: str(r.get(sort_by, "")))

    # Limit
    if limit is not None:
        result = result[:int(limit)]

    return result


@tool()
def data_filter(data: list, filter_expression: str = None, sort_by: str = None, limit: int = None) -> dict:
    """Filter, sort, and limit structured data.

    Operates on a list of dicts. Supports filtering via a simple Python
    expression (evaluated with 'row' in scope), sorting by a key name,
    and limiting the number of results.

    Args:
        data: List of dicts to filter.
        filter_expression: Python expression evaluated per row (variable 'row').
        sort_by: Key name to sort by.
        limit: Maximum number of rows to return.
    """
    if data is None:
        return {"error": "Parameter 'data' is required"}

    if limit is not None:
        try:
            limit = int(limit)
        except (ValueError, TypeError):
            return {"error": "Parameter 'limit' must be a number"}

    try:
        rows = _filter_data(
            data,
            filter_expression=filter_expression,
            sort_by=sort_by,
            limit=limit,
        )
    except Exception as exc:
        return {"error": str(exc)}

    return {"rows": rows, "count": len(rows)}


# Backward-compatible alias
DataFilterTool = data_filter

# Public alias for the filter helper
filter_data = _filter_data
