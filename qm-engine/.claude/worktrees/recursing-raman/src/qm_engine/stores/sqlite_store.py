"""SQLite execution store — persistent storage for local development.

Requires the `sqlite` optional dependency: pip install qm-engine[sqlite]
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from qm_engine.context.node_execution import NodeExecution, NodeStatus
from qm_engine.types import Message, MessageRole


class SQLiteStore:
    """SQLite-backed execution store for persistent local storage.

    Creates tables on first use. Thread-safe via SQLite's built-in locking.
    Suitable for local development, debugging, and single-process deployments.
    """

    def __init__(self, db_path: str | Path = "qm_engine.db") -> None:
        self._db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None
        self._ensure_tables()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def _ensure_tables(self) -> None:
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS node_executions (
                flow_id TEXT NOT NULL,
                node_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                started_at TEXT,
                finished_at TEXT,
                result TEXT,
                error TEXT,
                retry_count INTEGER DEFAULT 0,
                output_data TEXT DEFAULT '{}',
                PRIMARY KEY (flow_id, node_id)
            );

            CREATE TABLE IF NOT EXISTS flow_memory (
                flow_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (flow_id, key)
            );

            CREATE TABLE IF NOT EXISTS node_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                flow_id TEXT NOT NULL,
                node_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                name TEXT,
                tool_call_id TEXT,
                tool_calls TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_messages_flow_node
                ON node_messages(flow_id, node_id, position);
        """)
        conn.commit()

    def save_node_execution(self, flow_id: UUID, node_id: UUID, execution: NodeExecution) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO node_executions
               (flow_id, node_id, status, started_at, finished_at, result, error, retry_count, output_data)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(flow_id),
                str(node_id),
                execution.status.value,
                execution.started_at.isoformat() if execution.started_at else None,
                execution.finished_at.isoformat() if execution.finished_at else None,
                execution.result,
                execution.error,
                execution.retry_count,
                json.dumps(execution.output_data),
            ),
        )
        conn.commit()

    def get_node_execution(self, flow_id: UUID, node_id: UUID) -> NodeExecution | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM node_executions WHERE flow_id = ? AND node_id = ?",
            (str(flow_id), str(node_id)),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_execution(row)

    def get_all_node_executions(self, flow_id: UUID) -> dict[UUID, NodeExecution]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM node_executions WHERE flow_id = ?", (str(flow_id),)
        ).fetchall()
        return {UUID(row["node_id"]): self._row_to_execution(row) for row in rows}

    def save_memory(self, flow_id: UUID, key: str, value: Any) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO flow_memory (flow_id, key, value) VALUES (?, ?, ?)",
            (str(flow_id), key, json.dumps(value)),
        )
        conn.commit()

    def get_memory(self, flow_id: UUID, key: str) -> Any:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT value FROM flow_memory WHERE flow_id = ? AND key = ?",
            (str(flow_id), key),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["value"])

    def get_all_memory(self, flow_id: UUID) -> dict[str, Any]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT key, value FROM flow_memory WHERE flow_id = ?", (str(flow_id),)
        ).fetchall()
        return {row["key"]: json.loads(row["value"]) for row in rows}

    def delete_memory(self, flow_id: UUID, key: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "DELETE FROM flow_memory WHERE flow_id = ? AND key = ?",
            (str(flow_id), key),
        )
        conn.commit()

    def save_messages(self, flow_id: UUID, node_id: UUID, messages: list[Message]) -> None:
        conn = self._get_conn()
        # Clear existing messages for this node
        conn.execute(
            "DELETE FROM node_messages WHERE flow_id = ? AND node_id = ?",
            (str(flow_id), str(node_id)),
        )
        # Insert new messages
        for i, msg in enumerate(messages):
            conn.execute(
                """INSERT INTO node_messages
                   (flow_id, node_id, position, role, content, name, tool_call_id, tool_calls, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(flow_id),
                    str(node_id),
                    i,
                    msg.role.value,
                    msg.content,
                    msg.name,
                    msg.tool_call_id,
                    json.dumps(msg.tool_calls),
                    json.dumps(msg.metadata),
                ),
            )
        conn.commit()

    def get_messages(self, flow_id: UUID, node_id: UUID) -> list[Message]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM node_messages WHERE flow_id = ? AND node_id = ? ORDER BY position",
            (str(flow_id), str(node_id)),
        ).fetchall()
        return [self._row_to_message(row) for row in rows]

    def append_message(self, flow_id: UUID, node_id: UUID, message: Message) -> None:
        conn = self._get_conn()
        # Get the next position
        row = conn.execute(
            "SELECT COALESCE(MAX(position), -1) + 1 as next_pos FROM node_messages WHERE flow_id = ? AND node_id = ?",
            (str(flow_id), str(node_id)),
        ).fetchone()
        pos = row["next_pos"] if row else 0
        conn.execute(
            """INSERT INTO node_messages
               (flow_id, node_id, position, role, content, name, tool_call_id, tool_calls, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(flow_id),
                str(node_id),
                pos,
                message.role.value,
                message.content,
                message.name,
                message.tool_call_id,
                json.dumps(message.tool_calls),
                json.dumps(message.metadata),
            ),
        )
        conn.commit()

    def clear_flow(self, flow_id: UUID) -> None:
        conn = self._get_conn()
        fid = str(flow_id)
        conn.execute("DELETE FROM node_executions WHERE flow_id = ?", (fid,))
        conn.execute("DELETE FROM flow_memory WHERE flow_id = ?", (fid,))
        conn.execute("DELETE FROM node_messages WHERE flow_id = ?", (fid,))
        conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def _row_to_execution(self, row: sqlite3.Row) -> NodeExecution:
        return NodeExecution(
            node_id=UUID(row["node_id"]),
            status=NodeStatus(row["status"]),
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            finished_at=datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None,
            result=row["result"],
            error=row["error"],
            retry_count=row["retry_count"],
            output_data=json.loads(row["output_data"]) if row["output_data"] else {},
        )

    def _row_to_message(self, row: sqlite3.Row) -> Message:
        return Message(
            role=MessageRole(row["role"]),
            content=row["content"],
            name=row["name"],
            tool_call_id=row["tool_call_id"],
            tool_calls=json.loads(row["tool_calls"]) if row["tool_calls"] else [],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )
