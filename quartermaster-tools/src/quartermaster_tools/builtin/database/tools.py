"""
SQLite database tools: query, write, and schema introspection.

All SQL execution uses parameterized queries (``?`` placeholders) to
prevent SQL injection.  Only the Python standard library ``sqlite3``
module is used.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Any

from quartermaster_tools.decorator import tool
from quartermaster_tools.types import ToolResult

# Statements that are considered write operations
_WRITE_PREFIXES = ("INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER", "REPLACE")


def _is_write_sql(sql: str) -> bool:
    """Return True if the SQL statement appears to be a write operation."""
    stripped = sql.strip().upper()
    return any(stripped.startswith(prefix) for prefix in _WRITE_PREFIXES)


def _connect(database: str) -> sqlite3.Connection:
    """Open a SQLite connection with row factory set to sqlite3.Row."""
    conn = sqlite3.connect(database)
    conn.row_factory = sqlite3.Row
    return conn


@tool()
def sqlite_query(
    database: str,
    sql: str,
    params: list = None,
    max_rows: int = 100,
) -> ToolResult:
    """Execute a read-only SQL query on a SQLite database.

    Runs a SELECT query against the given SQLite database file and returns the
    results as a list of dictionaries with column names as keys. Write statements
    are rejected.

    Args:
        database: Path to the SQLite database file.
        sql: The SQL SELECT query to execute.
        params: Optional list of parameters for the query placeholders.
        max_rows: Maximum number of rows to return.
    """
    if not database:
        return ToolResult(success=False, error="Parameter 'database' is required")
    if not sql:
        return ToolResult(success=False, error="Parameter 'sql' is required")

    if _is_write_sql(sql):
        return ToolResult(
            success=False,
            error="Write operations are not allowed in query tool. Use sqlite_write instead.",
        )

    if not os.path.isfile(database):
        return ToolResult(success=False, error=f"Database file not found: {database}")

    try:
        conn = _connect(database)
        try:
            cursor = conn.execute(sql, params or [])
            rows = cursor.fetchmany(max_rows)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            result_rows = [dict(zip(columns, row)) for row in rows]
            return ToolResult(
                success=True,
                data={"rows": result_rows, "columns": columns, "row_count": len(result_rows)},
            )
        finally:
            conn.close()
    except sqlite3.Error as e:
        return ToolResult(success=False, error=f"SQLite error: {e}")


@tool()
def sqlite_write(
    database: str,
    sql: str,
    params: list = None,
    confirm: bool = False,
) -> ToolResult:
    """Execute a write SQL statement on a SQLite database.

    Executes INSERT, UPDATE, DELETE, CREATE TABLE, or other write SQL against
    a SQLite file. Requires confirm=True as a safety guard.

    Args:
        database: Path to the SQLite database file.
        sql: The SQL statement to execute (INSERT, UPDATE, DELETE, CREATE, etc.).
        params: Optional list of parameters for the query placeholders.
        confirm: Must be True to execute write operations. Safety guard.
    """
    if not database:
        return ToolResult(success=False, error="Parameter 'database' is required")
    if not sql:
        return ToolResult(success=False, error="Parameter 'sql' is required")
    if not confirm:
        return ToolResult(
            success=False,
            error="Write operations require confirm=True as a safety guard.",
        )

    try:
        conn = _connect(database)
        try:
            cursor = conn.execute(sql, params or [])
            conn.commit()
            return ToolResult(
                success=True,
                data={"rows_affected": cursor.rowcount, "message": "Statement executed."},
            )
        finally:
            conn.close()
    except sqlite3.Error as e:
        return ToolResult(success=False, error=f"SQLite error: {e}")


@tool()
def sqlite_schema(
    database: str,
    table: str = None,
) -> ToolResult:
    """Introspect SQLite database schema.

    Returns the list of tables in a SQLite database, or the column definitions
    for a specific table.

    Args:
        database: Path to the SQLite database file.
        table: Optional table name to inspect. If omitted, lists all tables.
    """
    if not database:
        return ToolResult(success=False, error="Parameter 'database' is required")
    if not os.path.isfile(database):
        return ToolResult(success=False, error=f"Database file not found: {database}")

    try:
        conn = _connect(database)
        try:
            if table is None:
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                )
                tables = [row["name"] for row in cursor.fetchall()]
                return ToolResult(
                    success=True,
                    data={"tables": tables, "count": len(tables)},
                )
            else:
                if not table.isidentifier():
                    return ToolResult(
                        success=False,
                        error=f"Invalid table name: {table!r}",
                    )
                cursor = conn.execute(f"PRAGMA table_info({table})")
                columns = []
                for row in cursor.fetchall():
                    columns.append({
                        "cid": row["cid"],
                        "name": row["name"],
                        "type": row["type"],
                        "notnull": bool(row["notnull"]),
                        "default_value": row["dflt_value"],
                        "primary_key": bool(row["pk"]),
                    })
                if not columns:
                    return ToolResult(
                        success=False,
                        error=f"Table not found: {table!r}",
                    )
                return ToolResult(
                    success=True,
                    data={"table": table, "columns": columns, "column_count": len(columns)},
                )
        finally:
            conn.close()
    except sqlite3.Error as e:
        return ToolResult(success=False, error=f"SQLite error: {e}")


# Backward-compatible aliases
SQLiteQueryTool = sqlite_query
SQLiteWriteTool = sqlite_write
SQLiteSchemaTool = sqlite_schema
