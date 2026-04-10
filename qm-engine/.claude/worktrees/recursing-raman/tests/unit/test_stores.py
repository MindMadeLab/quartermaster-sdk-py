"""Tests for InMemoryStore and SQLiteStore."""

import tempfile
from pathlib import Path
from uuid import uuid4

from qm_engine.context.node_execution import NodeExecution, NodeStatus
from qm_engine.stores.memory_store import InMemoryStore
from qm_engine.stores.sqlite_store import SQLiteStore
from qm_engine.types import Message, MessageRole


class StoreTestMixin:
    """Common tests for all ExecutionStore implementations."""

    store: InMemoryStore | SQLiteStore

    def test_save_and_get_node_execution(self):
        flow_id, node_id = uuid4(), uuid4()
        execution = NodeExecution(node_id=node_id)
        execution.start()
        execution.finish(result="done")

        self.store.save_node_execution(flow_id, node_id, execution)
        retrieved = self.store.get_node_execution(flow_id, node_id)

        assert retrieved is not None
        assert retrieved.node_id == node_id
        assert retrieved.status == NodeStatus.FINISHED
        assert retrieved.result == "done"

    def test_get_nonexistent_execution(self):
        assert self.store.get_node_execution(uuid4(), uuid4()) is None

    def test_get_all_node_executions(self):
        flow_id = uuid4()
        n1, n2 = uuid4(), uuid4()

        e1 = NodeExecution(node_id=n1)
        e1.start()
        e1.finish(result="r1")

        e2 = NodeExecution(node_id=n2)
        e2.start()
        e2.fail("err")

        self.store.save_node_execution(flow_id, n1, e1)
        self.store.save_node_execution(flow_id, n2, e2)

        all_execs = self.store.get_all_node_executions(flow_id)
        assert len(all_execs) == 2
        assert all_execs[n1].status == NodeStatus.FINISHED
        assert all_execs[n2].status == NodeStatus.FAILED

    def test_save_and_get_memory(self):
        flow_id = uuid4()
        self.store.save_memory(flow_id, "key1", "value1")
        assert self.store.get_memory(flow_id, "key1") == "value1"

    def test_get_nonexistent_memory(self):
        assert self.store.get_memory(uuid4(), "missing") is None

    def test_get_all_memory(self):
        flow_id = uuid4()
        self.store.save_memory(flow_id, "a", 1)
        self.store.save_memory(flow_id, "b", 2)
        all_mem = self.store.get_all_memory(flow_id)
        assert all_mem == {"a": 1, "b": 2}

    def test_delete_memory(self):
        flow_id = uuid4()
        self.store.save_memory(flow_id, "key", "value")
        self.store.delete_memory(flow_id, "key")
        assert self.store.get_memory(flow_id, "key") is None

    def test_save_and_get_messages(self):
        flow_id, node_id = uuid4(), uuid4()
        messages = [
            Message(role=MessageRole.SYSTEM, content="Be helpful"),
            Message(role=MessageRole.USER, content="Hello"),
            Message(role=MessageRole.ASSISTANT, content="Hi there!"),
        ]
        self.store.save_messages(flow_id, node_id, messages)
        retrieved = self.store.get_messages(flow_id, node_id)

        assert len(retrieved) == 3
        assert retrieved[0].role == MessageRole.SYSTEM
        assert retrieved[1].content == "Hello"
        assert retrieved[2].role == MessageRole.ASSISTANT

    def test_get_empty_messages(self):
        assert self.store.get_messages(uuid4(), uuid4()) == []

    def test_append_message(self):
        flow_id, node_id = uuid4(), uuid4()
        self.store.save_messages(
            flow_id,
            node_id,
            [
                Message(role=MessageRole.USER, content="First"),
            ],
        )
        self.store.append_message(
            flow_id, node_id, Message(role=MessageRole.ASSISTANT, content="Second")
        )

        msgs = self.store.get_messages(flow_id, node_id)
        assert len(msgs) == 2
        assert msgs[1].content == "Second"

    def test_clear_flow(self):
        flow_id = uuid4()
        node_id = uuid4()

        execution = NodeExecution(node_id=node_id)
        execution.start()
        self.store.save_node_execution(flow_id, node_id, execution)
        self.store.save_memory(flow_id, "key", "value")
        self.store.save_messages(flow_id, node_id, [Message(role=MessageRole.USER, content="test")])

        self.store.clear_flow(flow_id)

        assert self.store.get_node_execution(flow_id, node_id) is None
        assert self.store.get_memory(flow_id, "key") is None
        assert self.store.get_messages(flow_id, node_id) == []

    def test_memory_overwrite(self):
        flow_id = uuid4()
        self.store.save_memory(flow_id, "key", "v1")
        self.store.save_memory(flow_id, "key", "v2")
        assert self.store.get_memory(flow_id, "key") == "v2"

    def test_execution_overwrite(self):
        flow_id, node_id = uuid4(), uuid4()

        e1 = NodeExecution(node_id=node_id)
        e1.start()
        self.store.save_node_execution(flow_id, node_id, e1)

        e2 = NodeExecution(node_id=node_id)
        e2.start()
        e2.finish(result="updated")
        self.store.save_node_execution(flow_id, node_id, e2)

        retrieved = self.store.get_node_execution(flow_id, node_id)
        assert retrieved is not None
        assert retrieved.result == "updated"

    def test_complex_memory_values(self):
        flow_id = uuid4()
        self.store.save_memory(flow_id, "list", [1, 2, 3])
        self.store.save_memory(flow_id, "dict", {"nested": {"deep": True}})
        self.store.save_memory(flow_id, "null", None)

        assert self.store.get_memory(flow_id, "list") == [1, 2, 3]
        assert self.store.get_memory(flow_id, "dict") == {"nested": {"deep": True}}
        assert self.store.get_memory(flow_id, "null") is None


class TestInMemoryStore(StoreTestMixin):
    def setup_method(self):
        self.store = InMemoryStore()

    def test_memory_isolation_via_deepcopy(self):
        """Modifying returned values should not affect stored values."""
        flow_id = uuid4()
        original = {"key": [1, 2, 3]}
        self.store.save_memory(flow_id, "data", original)

        # Modify the original — store should not be affected
        original["key"].append(4)
        stored = self.store.get_memory(flow_id, "data")
        assert stored == {"key": [1, 2, 3]}

        # Modify retrieved value — store should not be affected
        stored["key"].append(5)
        stored_again = self.store.get_memory(flow_id, "data")
        assert stored_again == {"key": [1, 2, 3]}


class TestSQLiteStore(StoreTestMixin):
    def setup_method(self):
        self._tmpfile = tempfile.NamedTemporaryFile(suffix=".db", delete=False)  # noqa: SIM115
        self.store = SQLiteStore(db_path=self._tmpfile.name)

    def teardown_method(self):
        self.store.close()
        Path(self._tmpfile.name).unlink(missing_ok=True)

    def test_persistence_across_connections(self):
        """Data should survive closing and reopening the store."""
        flow_id = uuid4()
        self.store.save_memory(flow_id, "persistent", "data")
        self.store.close()

        # Reopen
        store2 = SQLiteStore(db_path=self._tmpfile.name)
        assert store2.get_memory(flow_id, "persistent") == "data"
        store2.close()
