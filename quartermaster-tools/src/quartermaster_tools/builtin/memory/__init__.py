"""
Memory/variable tools for in-process key-value storage.

Provides tools for storing, retrieving, and listing variables
in a shared in-memory dictionary that persists across calls
within the same process.
"""

from quartermaster_tools.builtin.memory.tools import (
    set_variable,
    get_variable,
    list_variables,
    create_memory_tools,
    SetVariableTool,
    GetVariableTool,
    ListVariablesTool,
)

__all__ = [
    "set_variable",
    "get_variable",
    "list_variables",
    "create_memory_tools",
    # Backward-compatible aliases
    "SetVariableTool",
    "GetVariableTool",
    "ListVariablesTool",
]
