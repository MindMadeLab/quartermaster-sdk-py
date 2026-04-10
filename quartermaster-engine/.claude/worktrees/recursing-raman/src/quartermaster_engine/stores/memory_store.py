"""In-memory execution store — default implementation backed by plain dicts."""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any
from uuid import UUID

from quartermaster_engine.context.node_execution import NodeExecution
from quartermaster_engine.types import Message


class InMemoryStore:
    """Dict-backed execution store. Fast but not persistent.

    Suitable for testing, short-lived flows, and single-process deployments.
    """

    def __init__(self) -> None:
        self._node_executions: dict[UUID, dict[UUID, NodeExecution]] = defaultdict(dict)
        self._memory: dict[UUID, dict[str, Any]] = defaultdict(dict)
        self._messages: dict[UUID, dict[UUID, list[Message]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def save_node_execution(self, flow_id: UUID, node_id: UUID, execution: NodeExecution) -> None:
        self._node_executions[flow_id][node_id] = execution

    def get_node_execution(self, flow_id: UUID, node_id: UUID) -> NodeExecution | None:
        return self._node_executions.get(flow_id, {}).get(node_id)

    def get_all_node_executions(self, flow_id: UUID) -> dict[UUID, NodeExecution]:
        return dict(self._node_executions.get(flow_id, {}))

    def save_memory(self, flow_id: UUID, key: str, value: Any) -> None:
        self._memory[flow_id][key] = copy.deepcopy(value)

    def get_memory(self, flow_id: UUID, key: str) -> Any:
        return copy.deepcopy(self._memory.get(flow_id, {}).get(key))

    def get_all_memory(self, flow_id: UUID) -> dict[str, Any]:
        return copy.deepcopy(dict(self._memory.get(flow_id, {})))

    def delete_memory(self, flow_id: UUID, key: str) -> None:
        mem = self._memory.get(flow_id, {})
        mem.pop(key, None)

    def save_messages(self, flow_id: UUID, node_id: UUID, messages: list[Message]) -> None:
        self._messages[flow_id][node_id] = list(messages)

    def get_messages(self, flow_id: UUID, node_id: UUID) -> list[Message]:
        return list(self._messages.get(flow_id, {}).get(node_id, []))

    def append_message(self, flow_id: UUID, node_id: UUID, message: Message) -> None:
        self._messages[flow_id][node_id].append(message)

    def clear_flow(self, flow_id: UUID) -> None:
        self._node_executions.pop(flow_id, None)
        self._memory.pop(flow_id, None)
        self._messages.pop(flow_id, None)
