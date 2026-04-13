"""
FastMCP-style @tool() decorator for creating tools from plain functions.

Turns a decorated function into a FunctionTool (subclass of AbstractTool)
by extracting metadata from the function signature, type hints, and docstring.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import re
from typing import Any, Callable, get_type_hints

from quartermaster_tools.base import AbstractTool
from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolResult

# Python type → ToolParameter type string
_PYTHON_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

# Parameter names to skip (reserved for framework context injection)
_SKIP_PARAMS = frozenset({"self", "cls", "ctx", "context"})


def _python_type_to_str(py_type: Any) -> str:
    """Map a Python type annotation to a ToolParameter type string."""
    return _PYTHON_TYPE_MAP.get(py_type, "string")


def _parse_docstring_args(docstring: str | None) -> dict[str, str]:
    """Parse Google-style docstring Args section into {param_name: description}."""
    if not docstring:
        return {}

    descriptions: dict[str, str] = {}
    lines = docstring.split("\n")
    in_args = False
    current_param: str | None = None
    current_desc: list[str] = []
    args_indent: int | None = None

    for line in lines:
        stripped = line.strip()

        # Detect start of Args section
        if stripped in ("Args:", "Arguments:", "Parameters:"):
            in_args = True
            args_indent = None
            continue

        if not in_args:
            continue

        # Detect end of Args section (another section header or empty after content)
        if stripped and not stripped.startswith(" ") and stripped.endswith(":") and stripped != stripped.lstrip():
            pass  # continuation line
        if re.match(r"^[A-Z]\w*:\s*$", stripped) and stripped not in ("Args:", "Arguments:", "Parameters:"):
            # New section header like "Returns:", "Raises:", etc.
            if current_param:
                descriptions[current_param] = " ".join(current_desc).strip()
            break

        if not stripped:
            # Empty line might end the section if we had content
            if current_param:
                descriptions[current_param] = " ".join(current_desc).strip()
                current_param = None
                current_desc = []
            continue

        # Try to match a parameter line: "  param_name: description" or "  param_name (type): description"
        param_match = re.match(r"^\s+(\w+)(?:\s*\([^)]*\))?\s*:\s*(.*)$", line)
        if param_match:
            # Save previous param
            if current_param:
                descriptions[current_param] = " ".join(current_desc).strip()

            current_param = param_match.group(1)
            desc_text = param_match.group(2).strip()
            current_desc = [desc_text] if desc_text else []
        elif current_param and stripped:
            # Continuation line for current param
            current_desc.append(stripped)

    # Don't forget the last param
    if current_param:
        descriptions[current_param] = " ".join(current_desc).strip()

    return descriptions


def _extract_short_description(docstring: str | None) -> str:
    """Extract the first non-empty line of a docstring as short description."""
    if not docstring:
        return ""
    for line in docstring.strip().split("\n"):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


class FunctionTool(AbstractTool):
    """A tool that wraps a plain Python function.

    Created by the @tool() decorator. Implements AbstractTool by extracting
    metadata from the wrapped function's signature, type hints, and docstring.
    """

    def __init__(
        self,
        func: Callable[..., Any],
        tool_name: str,
        description: str,
        long_description: str,
        params: list[ToolParameter],
    ) -> None:
        self._func = func
        self._name = tool_name
        self._description = description
        self._long_description = long_description
        self._params = params
        # Preserve function metadata
        functools.update_wrapper(self, func)

    def name(self) -> str:
        return self._name

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return list(self._params)

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self._name,
            short_description=self._description,
            long_description=self._long_description,
            version=self.version(),
            parameters=list(self._params),
        )

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the wrapped function and return a ToolResult."""
        try:
            if asyncio.iscoroutinefunction(self._func):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    # We're inside a running event loop; create a new one in a thread
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        result = pool.submit(
                            asyncio.run, self._func(**kwargs)
                        ).result()
                else:
                    result = asyncio.run(self._func(**kwargs))
            else:
                result = self._func(**kwargs)

            # Wrap result appropriately
            if isinstance(result, ToolResult):
                return result
            if isinstance(result, dict):
                return ToolResult(success=True, data=result)
            return ToolResult(success=True, data={"result": result})

        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Make FunctionTool callable, delegating to the wrapped function."""
        return self._func(*args, **kwargs)

    def __repr__(self) -> str:
        return f"FunctionTool({self._name!r})"


def tool(
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable[..., Any]], FunctionTool]:
    """Decorator that converts a plain function into a FunctionTool.

    Usage:
        @tool()
        def my_func(x: str, y: int = 5) -> dict:
            '''Short description.

            Longer description here.

            Args:
                x: The x parameter.
                y: The y parameter.
            '''
            return {"result": x}

        @tool(name="custom_name", description="Override description")
        def another_func(a: float) -> dict:
            ...

    Args:
        name: Override the tool name (defaults to function __name__).
        description: Override the short description (defaults to docstring first line).

    Returns:
        A decorator that converts a function into a FunctionTool instance.
    """

    def decorator(func: Callable[..., Any]) -> FunctionTool:
        tool_name = name or func.__name__
        docstring = inspect.getdoc(func) or ""
        short_desc = description or _extract_short_description(docstring)
        long_desc = docstring

        # Get type hints (gracefully handle missing annotations)
        try:
            hints = get_type_hints(func)
        except Exception:
            hints = {}

        # Remove return type hint
        hints.pop("return", None)

        # Parse parameter descriptions from docstring
        param_descriptions = _parse_docstring_args(docstring)

        # Build ToolParameter list from signature
        sig = inspect.signature(func)
        params: list[ToolParameter] = []

        for param_name, param in sig.parameters.items():
            if param_name in _SKIP_PARAMS:
                continue

            py_type = hints.get(param_name, str)
            param_type_str = _python_type_to_str(py_type)
            param_desc = param_descriptions.get(param_name, "")
            has_default = param.default is not inspect.Parameter.empty
            default_value = param.default if has_default else None

            params.append(
                ToolParameter(
                    name=param_name,
                    description=param_desc,
                    type=param_type_str,
                    required=not has_default,
                    default=default_value,
                )
            )

        return FunctionTool(
            func=func,
            tool_name=tool_name,
            description=short_desc,
            long_description=long_desc,
            params=params,
        )

    return decorator
