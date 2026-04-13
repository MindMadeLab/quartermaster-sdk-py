"""Tests for the SQLite database tools."""

from __future__ import annotations

import os
import sqlite3
import tempfile

import pytest

from quartermaster_tools.builtin.database.tools import (
    SQLiteQueryTool,
    SQLiteSchemaTool,
    SQLiteWriteTool,
)


@pytest.fixture()
def db_path(tmp_path):
    """Create a temporary SQLite database with a sample table."""
    path = str(tmp_path / "test.db")
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT)"
    )
    conn.execute("INSERT INTO users (name, email) VALUES ('Alice', 'alice@example.com')")
    conn.execute("INSERT INTO users (name, email) VALUES ('Bob', 'bob@example.com')")
    conn.execute("INSERT INTO users (name, email) VALUES ('Charlie', 'charlie@example.com')")
    conn.commit()
    conn.close()
    return path


@pytest.fixture()
def empty_db(tmp_path):
    """Create an empty temporary SQLite database."""
    path = str(tmp_path / "empty.db")
    conn = sqlite3.connect(path)
    conn.close()
    return path


# --- SQLiteQueryTool ---


class TestSQLiteQueryTool:
    def test_select_all(self, db_path):
        result = SQLiteQueryTool.run(database=db_path, sql="SELECT * FROM users")
        assert result.success is True
        assert result.data["row_count"] == 3
        assert result.data["columns"] == ["id", "name", "email"]

    def test_select_with_where(self, db_path):
        result = SQLiteQueryTool.run(
            database=db_path,
            sql="SELECT name FROM users WHERE id = ?",
            params=[1],
        )
        assert result.success is True
        assert result.data["rows"] == [{"name": "Alice"}]

    def test_select_with_params(self, db_path):
        result = SQLiteQueryTool.run(
            database=db_path,
            sql="SELECT * FROM users WHERE name = ?",
            params=["Bob"],
        )
        assert result.data["row_count"] == 1
        assert result.data["rows"][0]["name"] == "Bob"

    def test_max_rows_limit(self, db_path):
        result = SQLiteQueryTool.run(database=db_path, sql="SELECT * FROM users", max_rows=2)
        assert result.success is True
        assert result.data["row_count"] == 2

    def test_rejects_write_sql(self, db_path):
        result = SQLiteQueryTool.run(
            database=db_path,
            sql="INSERT INTO users (name, email) VALUES ('Eve', 'eve@test.com')",
        )
        assert result.success is False
        assert "write" in result.error.lower()

    def test_rejects_delete(self, db_path):
        result = SQLiteQueryTool.run(database=db_path, sql="DELETE FROM users WHERE id = 1")
        assert result.success is False

    def test_rejects_drop(self, db_path):
        result = SQLiteQueryTool.run(database=db_path, sql="DROP TABLE users")
        assert result.success is False

    def test_missing_database(self):
        result = SQLiteQueryTool.run(database="", sql="SELECT 1")
        assert result.success is False
        assert "database" in result.error.lower()

    def test_nonexistent_database_file(self):
        result = SQLiteQueryTool.run(database="/tmp/nonexistent_db_12345.db", sql="SELECT 1")
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_missing_sql(self, db_path):
        result = SQLiteQueryTool.run(database=db_path, sql="")
        assert result.success is False

    def test_invalid_sql(self, db_path):
        result = SQLiteQueryTool.run(database=db_path, sql="NOT VALID SQL")
        assert result.success is False
        assert "sqlite error" in result.error.lower()

    def test_sql_injection_prevented(self, db_path):
        """Parameterized queries prevent injection via params."""
        result = SQLiteQueryTool.run(
            database=db_path,
            sql="SELECT * FROM users WHERE name = ?",
            params=["'; DROP TABLE users; --"],
        )
        assert result.success is True
        assert result.data["row_count"] == 0
        # Table should still exist
        result2 = SQLiteQueryTool.run(database=db_path, sql="SELECT COUNT(*) as cnt FROM users")
        assert result2.success is True
        assert result2.data["rows"][0]["cnt"] == 3


# --- SQLiteWriteTool ---


class TestSQLiteWriteTool:
    def test_insert(self, db_path):
        result = SQLiteWriteTool.run(
            database=db_path,
            sql="INSERT INTO users (name, email) VALUES (?, ?)",
            params=["Dave", "dave@example.com"],
            confirm=True,
        )
        assert result.success is True
        assert result.data["rows_affected"] == 1

    def test_update(self, db_path):
        result = SQLiteWriteTool.run(
            database=db_path,
            sql="UPDATE users SET email = ? WHERE name = ?",
            params=["new@example.com", "Alice"],
            confirm=True,
        )
        assert result.success is True
        assert result.data["rows_affected"] == 1

    def test_delete(self, db_path):
        result = SQLiteWriteTool.run(
            database=db_path,
            sql="DELETE FROM users WHERE name = ?",
            params=["Charlie"],
            confirm=True,
        )
        assert result.success is True
        assert result.data["rows_affected"] == 1

    def test_create_table(self, db_path):
        result = SQLiteWriteTool.run(
            database=db_path,
            sql="CREATE TABLE logs (id INTEGER PRIMARY KEY, msg TEXT)",
            confirm=True,
        )
        assert result.success is True

    def test_requires_confirm(self, db_path):
        result = SQLiteWriteTool.run(
            database=db_path,
            sql="INSERT INTO users (name, email) VALUES (?, ?)",
            params=["Eve", "eve@test.com"],
        )
        assert result.success is False
        assert "confirm" in result.error.lower()

    def test_confirm_false_rejects(self, db_path):
        result = SQLiteWriteTool.run(
            database=db_path,
            sql="INSERT INTO users (name, email) VALUES (?, ?)",
            params=["Eve", "eve@test.com"],
            confirm=False,
        )
        assert result.success is False

    def test_missing_database(self):
        result = SQLiteWriteTool.run(database="", sql="SELECT 1", confirm=True)
        assert result.success is False

    def test_missing_sql(self, db_path):
        result = SQLiteWriteTool.run(database=db_path, sql="", confirm=True)
        assert result.success is False

    def test_invalid_sql(self, db_path):
        result = SQLiteWriteTool.run(database=db_path, sql="INVALID SQL", confirm=True)
        assert result.success is False

    def test_parameterized_insert_prevents_injection(self, db_path):
        malicious_name = "'; DROP TABLE users; --"
        result = SQLiteWriteTool.run(
            database=db_path,
            sql="INSERT INTO users (name, email) VALUES (?, ?)",
            params=[malicious_name, "evil@test.com"],
            confirm=True,
        )
        assert result.success is True
        # Verify table still intact with the malicious string stored as data
        r2 = SQLiteQueryTool.run(database=db_path, sql="SELECT * FROM users WHERE name = ?", params=[malicious_name])
        assert r2.data["row_count"] == 1


# --- SQLiteSchemaTool ---


class TestSQLiteSchemaTool:
    def test_list_tables(self, db_path):
        result = SQLiteSchemaTool.run(database=db_path)
        assert result.success is True
        assert "users" in result.data["tables"]

    def test_list_tables_empty_db(self, empty_db):
        result = SQLiteSchemaTool.run(database=empty_db)
        assert result.success is True
        assert result.data["tables"] == []

    def test_table_columns(self, db_path):
        result = SQLiteSchemaTool.run(database=db_path, table="users")
        assert result.success is True
        assert result.data["table"] == "users"
        col_names = [c["name"] for c in result.data["columns"]]
        assert "id" in col_names
        assert "name" in col_names
        assert "email" in col_names

    def test_table_column_types(self, db_path):
        result = SQLiteSchemaTool.run(database=db_path, table="users")
        cols = {c["name"]: c for c in result.data["columns"]}
        assert cols["id"]["type"] == "INTEGER"
        assert cols["id"]["primary_key"] is True
        assert cols["name"]["notnull"] is True

    def test_nonexistent_table(self, db_path):
        result = SQLiteSchemaTool.run(database=db_path, table="nonexistent")
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_invalid_table_name(self, db_path):
        result = SQLiteSchemaTool.run(database=db_path, table="'; DROP TABLE users;--")
        assert result.success is False
        assert "invalid" in result.error.lower()

    def test_missing_database(self):
        result = SQLiteSchemaTool.run(database="")
        assert result.success is False

    def test_nonexistent_database_file(self):
        result = SQLiteSchemaTool.run(database="/tmp/no_such_file_xyz.db")
        assert result.success is False

    def test_multiple_tables(self, db_path):
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER)")
        conn.commit()
        conn.close()

        result = SQLiteSchemaTool.run(database=db_path)
        assert "users" in result.data["tables"]
        assert "orders" in result.data["tables"]
        assert result.data["count"] == 2


# --- Tool metadata ---


class TestDatabaseToolMetadata:
    def test_query_tool_info(self):
        assert SQLiteQueryTool.name() == "sqlite_query"
        assert SQLiteQueryTool.version() == "1.0.0"
        info = SQLiteQueryTool.info()
        assert info.is_local is True

    def test_write_tool_info(self):
        assert SQLiteWriteTool.name() == "sqlite_write"

    def test_schema_tool_info(self):
        assert SQLiteSchemaTool.name() == "sqlite_schema"
