"""
data_filter: Filter, sort, and limit structured data.

Operates on lists of dicts (tabular data). Supports simple Python filter
expressions evaluated safely per row, sorting by key, and result limiting.
"""

from __future__ import annotations

from typing import Any

from simpleeval import EvalWithCompoundTypes, FeatureNotAvailable, NameNotDefined

from quartermaster_tools.decorator import tool

# -- Safe filter evaluator backed by simpleeval ----------------------------

_FILTER_SAFE_FUNCS: dict[str, Any] = {
    "len": len,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "sum": sum,
    "sorted": sorted,
    "any": any,
    "all": all,
    "isinstance": isinstance,
    "hasattr": hasattr,
}


def _safe_eval_filter(expression: str, context: dict[str, Any]) -> Any:
    """Safely evaluate a filter expression using ``simpleeval``.

    Never calls Python's ``eval()`` or ``exec()``.
    """
    if not expression or not expression.strip():
        raise ValueError("Expression is empty")
    if len(expression) > 10_000:
        raise ValueError("Expression too long")

    evaluator = EvalWithCompoundTypes(names=context)
    evaluator.functions.update(_FILTER_SAFE_FUNCS)

    try:
        return evaluator.eval(expression)
    except (
        FeatureNotAvailable,
        NameNotDefined,
        TypeError,
        ValueError,
        AttributeError,
        KeyError,
        IndexError,
        ZeroDivisionError,
    ) as e:
        raise ValueError(str(e)) from e
    except Exception as e:
        raise ValueError(f"Evaluation failed: {e}") from e


def _validate_expression(expr: str) -> str | None:
    """Return an error message if *expr* uses blocked constructs, else None."""
    _blocked = {
        "__import__",
        "eval",
        "exec",
        "compile",
        "open",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "dir",
        "type",
        "breakpoint",
        "exit",
        "quit",
        "input",
        "print",
        "__builtins__",
    }
    for name in _blocked:
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
                raise ValueError(f"Error evaluating expression on row {row!r}: {exc}") from exc
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
        result = result[: int(limit)]

    return result


@tool()
def data_filter(
    data: list, filter_expression: str = None, sort_by: str = None, limit: int = None
) -> dict:
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
