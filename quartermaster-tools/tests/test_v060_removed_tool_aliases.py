"""Negative regression tests — v0.6.0 removed ~35 ``*Tool = snake_case``
backward-compat aliases across every builtin module.

The canonical API is always the snake_case function. These tests guard
against accidental reintroduction of the capitalised aliases.

If you need to add a new tool in the future: expose the snake_case
function. Do NOT create a ``FooBarTool = foo_bar`` alias.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import quartermaster_tools.builtin as _builtin_pkg

# The concrete aliases we dropped in v0.6.0. Kept as an explicit list so a
# failed test points at the exact alias that came back, not just "something
# capitalised got re-added".
_REMOVED_ALIASES: tuple[str, ...] = (
    # web_search
    "GoogleSearchTool",
    "DuckDuckGoSearchTool",
    "BraveSearchTool",
    "JsonApiTool",
    "WebScraperTool",
    # web_request
    "WebRequestTool",
    # data
    "ConvertFormatTool",
    "DataFilterTool",
    "ParseCSVTool",
    "ParseJSONTool",
    "ParseXMLTool",
    "ParseYAMLTool",
    # filesystem
    "ReadFileTool",
    "WriteFileTool",
    "DeleteFileTool",
    "CopyFileTool",
    "MoveFileTool",
    "GrepTool",
    "FindFilesTool",
    "CreateDirectoryTool",
    "ListDirectoryTool",
    "FileInfoTool",
    # database
    "SQLiteQueryTool",
    "SQLiteWriteTool",
    "SQLiteSchemaTool",
    # memory
    "GetVariableTool",
    "SetVariableTool",
    "ListVariablesTool",
    # code / math
    "EvalMathTool",
    "JavaScriptExecutorTool",
    "PythonExecutorTool",
    "ShellExecutorTool",
    # email
    "SendEmailTool",
    "ReadEmailTool",
    "SearchEmailTool",
)


@pytest.mark.parametrize("alias", _REMOVED_ALIASES)
def test_alias_not_importable_from_top_level(alias: str) -> None:
    """The capitalised aliases must not come back via
    ``from quartermaster_tools import FooBarTool``."""
    import quartermaster_tools

    assert not hasattr(quartermaster_tools, alias)


@pytest.mark.parametrize("alias", _REMOVED_ALIASES)
def test_alias_not_in_builtin_dunder_all(alias: str) -> None:
    """Walk every builtin submodule and assert the alias isn't re-exported."""
    for _finder, modname, _ispkg in pkgutil.walk_packages(
        _builtin_pkg.__path__, prefix=f"{_builtin_pkg.__name__}."
    ):
        try:
            mod = importlib.import_module(modname)
        except Exception:
            # Some built-ins require optional deps; skip if unimportable.
            continue
        assert not hasattr(mod, alias), (
            f"{modname} still exposes {alias!r} — v0.6.0 dropped it; "
            "use the snake_case function instead"
        )


def test_snake_case_functions_still_work() -> None:
    """Smoke-check that the canonical names are still the supported API —
    we only removed the aliases, never the underlying tools."""
    from quartermaster_tools.builtin.web_search import (
        brave_search,
        duckduckgo_search,
        google_search,
        json_api,
        web_scraper,
    )

    for fn in (brave_search, duckduckgo_search, google_search, json_api, web_scraper):
        assert callable(fn)
        assert hasattr(fn, "name"), (
            f"{fn.__name__} lost its @tool() decorator metadata — the "
            "alias removal should not touch the underlying function"
        )
