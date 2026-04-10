"""
DataFilterTool: Filter, sort, and limit structured data.

Operates on lists of dicts (tabular data). Supports simple Python filter
expressions evaluated safely per row, sorting by key, and result limiting.
"""

from __future__ import annotations

from typing import Any

from quartermaster_tools.base import AbstractTool
from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolResult

# Builtins allowed in filter expressions for safety
_SAFE_BUILTINS: dict[str, Any] = {
    "len": len,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "True": True,
    "False": False,
    "None": None,
}

# Names that are never allowed in filter expressions
_BLOCKED_NAMES = frozenset({
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
})


def _validate_expression(expr: str) -> str | None:
    """Return an error message if *expr* contains blocked names, else None."""
    for name in _BLOCKED_NAMES:
        if name in expr:
            return f"Expression contains blocked name: {name!r}"
    if "__" in expr:
        return "Dunder attributes are not allowed in filter expressions"
    return None


class DataFilterTool(AbstractTool):
    """Filter, sort, and limit structured data (list of dicts).

    Supports:
    - ``filter_expression``: a Python expression evaluated per row with
      ``row`` in scope (e.g. ``row['age'] > 18``).
    - ``sort_by``: key name to sort by.
    - ``limit``: maximum number of rows to return.
    """

    def name(self) -> str:
        """Return the tool name."""
        return "data_filter"

    def version(self) -> str:
        """Return the tool version."""
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        """Return parameter definitions for the tool."""
        return [
            ToolParameter(
                name="data",
                description="List of dicts to filter.",
                type="array",
                required=True,
            ),
            ToolParameter(
                name="filter_expression",
                description="Python expression evaluated per row (variable 'row').",
                type="string",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="sort_by",
                description="Key name to sort by.",
                type="string",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="limit",
                description="Maximum number of rows to return.",
                type="number",
                required=False,
                default=None,
            ),
        ]

    def info(self) -> ToolDescriptor:
        """Return metadata describing this tool."""
        return ToolDescriptor(
            name=self.name(),
            short_description="Filter, sort, and limit structured data.",
            long_description=(
                "Operates on a list of dicts. Supports filtering via a simple "
                "Python expression (evaluated with 'row' in scope), sorting "
                "by a key name, and limiting the number of results."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def filter(
        self,
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
                    safe_globals: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
                    if eval(filter_expression, safe_globals, {"row": row}):  # noqa: S307
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

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the data filter tool.

        Args:
            data: List of dicts to process.
            filter_expression: Optional Python filter expression.
            sort_by: Optional key name to sort by.
            limit: Optional maximum number of rows.

        Returns:
            ToolResult with filtered rows in ``data["rows"]``.
        """
        data = kwargs.get("data")
        filter_expression: str | None = kwargs.get("filter_expression")
        sort_by: str | None = kwargs.get("sort_by")
        limit = kwargs.get("limit")

        if data is None:
            return ToolResult(success=False, error="Parameter 'data' is required")

        if limit is not None:
            try:
                limit = int(limit)
            except (ValueError, TypeError):
                return ToolResult(success=False, error="Parameter 'limit' must be a number")

        try:
            rows = self.filter(
                data,
                filter_expression=filter_expression,
                sort_by=sort_by,
                limit=limit,
            )
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

        return ToolResult(success=True, data={"rows": rows, "count": len(rows)})
