"""
SQLite database tools for querying, writing, and inspecting schemas.

Uses only the Python standard library ``sqlite3`` module.
All SQL execution uses parameterized queries to prevent injection.
"""

from quartermaster_tools.builtin.database.tools import (
    sqlite_query,
    sqlite_write,
    sqlite_schema,
    SQLiteQueryTool,
    SQLiteSchemaTool,
    SQLiteWriteTool,
)

__all__ = [
    "sqlite_query",
    "sqlite_write",
    "sqlite_schema",
    # Backward-compatible aliases
    "SQLiteQueryTool",
    "SQLiteSchemaTool",
    "SQLiteWriteTool",
]
