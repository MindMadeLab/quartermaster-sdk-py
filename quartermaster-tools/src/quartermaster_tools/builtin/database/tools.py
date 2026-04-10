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

from quartermaster_tools.base import AbstractTool
from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolResult

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


class SQLiteQueryTool(AbstractTool):
    """Execute read-only SQL queries against a SQLite database file."""

    def name(self) -> str:
        return "sqlite_query"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="database",
                description="Path to the SQLite database file.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="sql",
                description="The SQL SELECT query to execute.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="params",
                description="Optional list of parameters for the query placeholders.",
                type="array",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="max_rows",
                description="Maximum number of rows to return.",
                type="number",
                required=False,
                default=100,
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Execute a read-only SQL query on a SQLite database.",
            long_description=(
                "Runs a SELECT query against the given SQLite database file "
                "and returns the results as a list of dictionaries with "
                "column names as keys.  Write statements are rejected."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute a read-only SQL query.

        Args:
            database: Path to the SQLite file.
            sql: The SQL query (must be a SELECT or similar read operation).
            params: Optional list of bind parameters.
            max_rows: Maximum rows to return (default 100).

        Returns:
            ToolResult with rows as list of dicts.
        """
        database: str = kwargs.get("database", "")
        sql: str = kwargs.get("sql", "")
        params: list[Any] | None = kwargs.get("params", None)
        max_rows: int = kwargs.get("max_rows", 100)

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


class SQLiteWriteTool(AbstractTool):
    """Execute write SQL statements against a SQLite database file."""

    def name(self) -> str:
        return "sqlite_write"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="database",
                description="Path to the SQLite database file.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="sql",
                description="The SQL statement to execute (INSERT, UPDATE, DELETE, CREATE, etc.).",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="params",
                description="Optional list of parameters for the query placeholders.",
                type="array",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="confirm",
                description="Must be True to execute write operations. Safety guard.",
                type="boolean",
                required=False,
                default=False,
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Execute a write SQL statement on a SQLite database.",
            long_description=(
                "Executes INSERT, UPDATE, DELETE, CREATE TABLE, or other "
                "write SQL against a SQLite file.  Requires confirm=True "
                "as a safety guard."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute a write SQL statement.

        Args:
            database: Path to the SQLite file.
            sql: The SQL statement to execute.
            params: Optional list of bind parameters.
            confirm: Must be True to proceed.

        Returns:
            ToolResult with rows_affected count.
        """
        database: str = kwargs.get("database", "")
        sql: str = kwargs.get("sql", "")
        params: list[Any] | None = kwargs.get("params", None)
        confirm: bool = kwargs.get("confirm", False)

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


class SQLiteSchemaTool(AbstractTool):
    """Introspect the schema of a SQLite database file."""

    def name(self) -> str:
        return "sqlite_schema"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="database",
                description="Path to the SQLite database file.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="table",
                description="Optional table name to inspect. If omitted, lists all tables.",
                type="string",
                required=False,
                default=None,
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Introspect SQLite database schema.",
            long_description=(
                "Returns the list of tables in a SQLite database, or the "
                "column definitions for a specific table."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        """Inspect SQLite schema.

        Args:
            database: Path to the SQLite file.
            table: If provided, return columns for this table. Otherwise list tables.

        Returns:
            ToolResult with schema information.
        """
        database: str = kwargs.get("database", "")
        table: str | None = kwargs.get("table", None)

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
                    # Use parameterized PRAGMA via string formatting only for
                    # the table name validation — PRAGMA doesn't support ? params.
                    # Validate table name to prevent injection.
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
