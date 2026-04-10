"""Integration tests for memory systems — flow memory, stores, and persistence."""

from __future__ import annotations

import tempfile
from pathlib import Path
from uuid import uuid4

from qm_engine.events import FlowEvent, NodeFinished
from qm_engine.memory.flow_memory import FlowMemory
from qm_engine.memory.persistent_memory import InMemoryPersistence
from qm_engine.nodes import SimpleNodeRegistry
from qm_engine.runner.flow_runner import FlowRunner
from qm_engine.stores.memory_store import InMemoryStore
from qm_engine.stores.sqlite_store import SQLiteStore
from qm_engine.types import (
    NodeType,
    TraverseOut,
)
from tests.conftest import (
    EchoExecutor,
    MemoryReadExecutor,
    MemoryWriteExecutor,
    make_edge,
    make_graph,
    make_node,
)


class TestInMemoryStoreOperations:
    """Direct store/retrieve/delete operations on InMemoryStore."""

    def test_save_and_retrieve_memory(self):
        store = InMemoryStore()
        flow_id = uuid4()
        store.save_memory(flow_id, "key1", "value1")
        assert store.get_memory(flow_id, "key1") == "value1"

    def test_get_nonexistent_returns_none(self):
        store = InMemoryStore()
        flow_id = uuid4()
        assert store.get_memory(flow_id, "missing") is None

    def test_delete_memory(self):
        store = InMemoryStore()
        flow_id = uuid4()
        store.save_memory(flow_id, "key1", "value1")
        store.delete_memory(flow_id, "key1")
        assert store.get_memory(flow_id, "key1") is None

    def test_get_all_memory(self):
        store = InMemoryStore()
        flow_id = uuid4()
        store.save_memory(flow_id, "a", 1)
        store.save_memory(flow_id, "b", 2)
        all_mem = store.get_all_memory(flow_id)
        assert all_mem == {"a": 1, "b": 2}

    def test_clear_flow_removes_all_state(self):
        store = InMemoryStore()
        flow_id = uuid4()
        store.save_memory(flow_id, "key", "val")
        store.clear_flow(flow_id)
        assert store.get_all_memory(flow_id) == {}


class TestFlowMemoryWrapper:
    """FlowMemory wrapper over ExecutionStore."""

    def test_set_and_get(self):
        store = InMemoryStore()
        flow_id = uuid4()
        mem = FlowMemory(flow_id, store)
        mem.set("greeting", "hello")
        assert mem.get("greeting") == "hello"

    def test_get_default(self):
        store = InMemoryStore()
        flow_id = uuid4()
        mem = FlowMemory(flow_id, store)
        assert mem.get("missing", "default") == "default"

    def test_delete(self):
        store = InMemoryStore()
        flow_id = uuid4()
        mem = FlowMemory(flow_id, store)
        mem.set("key", "val")
        mem.delete("key")
        assert mem.get("key") is None

    def test_list_keys(self):
        store = InMemoryStore()
        flow_id = uuid4()
        mem = FlowMemory(flow_id, store)
        mem.set("a", 1)
        mem.set("b", 2)
        keys = mem.list_keys()
        assert sorted(keys) == ["a", "b"]

    def test_clear(self):
        store = InMemoryStore()
        flow_id = uuid4()
        mem = FlowMemory(flow_id, store)
        mem.set("x", 1)
        mem.set("y", 2)
        mem.clear()
        assert mem.get_all() == {}


class TestMemoryPersistenceAcrossNodes:
    """Memory written by one node is readable by a subsequent node."""

    def test_memory_write_then_read_in_flow(self):
        start = make_node(NodeType.START, name="Start")
        writer = make_node(NodeType.INSTRUCTION, name="Writer")
        reader = make_node(NodeType.STATIC, name="Reader")
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, writer, reader, end],
            [
                make_edge(start, writer),
                make_edge(writer, reader),
                make_edge(reader, end),
            ],
            start,
        )

        registry = SimpleNodeRegistry()
        registry.register(
            NodeType.INSTRUCTION.value,
            MemoryWriteExecutor(key="session", value="active"),
        )
        registry.register(NodeType.STATIC.value, MemoryReadExecutor(key="session"))

        runner = FlowRunner(graph=graph, node_registry=registry)
        result = runner.run("memory persistence test")

        assert result.success
        assert "session=active" in result.final_output


class TestMemoryIsolation:
    """Different flow runs have isolated memory."""

    def test_separate_flows_have_independent_memory(self):
        store = InMemoryStore()
        flow1 = uuid4()
        flow2 = uuid4()

        store.save_memory(flow1, "key", "flow1_value")
        store.save_memory(flow2, "key", "flow2_value")

        assert store.get_memory(flow1, "key") == "flow1_value"
        assert store.get_memory(flow2, "key") == "flow2_value"

    def test_flow_runs_do_not_leak_memory(self):
        """Running two separate flows through the same runner shape should not share memory."""
        start = make_node(NodeType.START, name="Start")
        writer = make_node(NodeType.INSTRUCTION, name="Writer")
        reader = make_node(NodeType.STATIC, name="Reader")
        end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

        graph = make_graph(
            [start, writer, reader, end],
            [
                make_edge(start, writer),
                make_edge(writer, reader),
                make_edge(reader, end),
            ],
            start,
        )

        registry = SimpleNodeRegistry()
        registry.register(
            NodeType.INSTRUCTION.value,
            MemoryWriteExecutor(key="run_id", value="run1"),
        )
        registry.register(NodeType.STATIC.value, MemoryReadExecutor(key="run_id"))

        runner1 = FlowRunner(graph=graph, node_registry=registry)
        result1 = runner1.run("first run")

        # Create a second runner with different memory writer
        registry2 = SimpleNodeRegistry()
        registry2.register(
            NodeType.INSTRUCTION.value,
            MemoryWriteExecutor(key="run_id", value="run2"),
        )
        registry2.register(NodeType.STATIC.value, MemoryReadExecutor(key="run_id"))

        runner2 = FlowRunner(graph=graph, node_registry=registry2)
        result2 = runner2.run("second run")

        assert "run_id=run1" in result1.final_output
        assert "run_id=run2" in result2.final_output


class TestLargeMemoryPayloads:
    """Memory system handles large values without issues."""

    def test_large_string_value(self):
        store = InMemoryStore()
        flow_id = uuid4()
        large_value = "x" * 100_000
        store.save_memory(flow_id, "big", large_value)
        assert store.get_memory(flow_id, "big") == large_value

    def test_nested_dict_value(self):
        store = InMemoryStore()
        flow_id = uuid4()
        nested = {"level1": {"level2": {"level3": list(range(100))}}}
        store.save_memory(flow_id, "nested", nested)
        retrieved = store.get_memory(flow_id, "nested")
        assert retrieved == nested


class TestSQLiteStore:
    """SQLite-backed store for persistent storage."""

    def test_sqlite_save_and_retrieve_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = SQLiteStore(db_path=db_path)
            try:
                flow_id = uuid4()
                store.save_memory(flow_id, "key1", "value1")
                assert store.get_memory(flow_id, "key1") == "value1"
            finally:
                store.close()

    def test_sqlite_delete_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = SQLiteStore(db_path=db_path)
            try:
                flow_id = uuid4()
                store.save_memory(flow_id, "key1", "value1")
                store.delete_memory(flow_id, "key1")
                assert store.get_memory(flow_id, "key1") is None
            finally:
                store.close()

    def test_sqlite_get_all_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = SQLiteStore(db_path=db_path)
            try:
                flow_id = uuid4()
                store.save_memory(flow_id, "a", "1")
                store.save_memory(flow_id, "b", "2")
                all_mem = store.get_all_memory(flow_id)
                assert all_mem == {"a": "1", "b": "2"}
            finally:
                store.close()

    def test_sqlite_flow_isolation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = SQLiteStore(db_path=db_path)
            try:
                f1, f2 = uuid4(), uuid4()
                store.save_memory(f1, "key", "val1")
                store.save_memory(f2, "key", "val2")
                assert store.get_memory(f1, "key") == "val1"
                assert store.get_memory(f2, "key") == "val2"
            finally:
                store.close()

    def test_sqlite_clear_flow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = SQLiteStore(db_path=db_path)
            try:
                flow_id = uuid4()
                store.save_memory(flow_id, "key", "val")
                store.clear_flow(flow_id)
                assert store.get_all_memory(flow_id) == {}
            finally:
                store.close()

    def test_sqlite_as_flow_runner_store(self):
        """SQLiteStore works as the backing store for a FlowRunner."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = SQLiteStore(db_path=db_path)
            try:
                start = make_node(NodeType.START, name="Start")
                writer = make_node(NodeType.INSTRUCTION, name="Writer")
                reader = make_node(NodeType.STATIC, name="Reader")
                end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)

                graph = make_graph(
                    [start, writer, reader, end],
                    [
                        make_edge(start, writer),
                        make_edge(writer, reader),
                        make_edge(reader, end),
                    ],
                    start,
                )

                registry = SimpleNodeRegistry()
                registry.register(
                    NodeType.INSTRUCTION.value,
                    MemoryWriteExecutor(key="from_sqlite", value="works"),
                )
                registry.register(NodeType.STATIC.value, MemoryReadExecutor(key="from_sqlite"))

                runner = FlowRunner(graph=graph, node_registry=registry, store=store)
                result = runner.run("sqlite flow test")

                assert result.success
                assert "from_sqlite=works" in result.final_output
            finally:
                store.close()


class TestPersistentMemory:
    """InMemoryPersistence — cross-flow agent-level memory."""

    def test_write_and_read(self):
        mem = InMemoryPersistence()
        agent_id = uuid4()
        mem.write(agent_id, "fact", "The sky is blue")
        assert mem.read(agent_id, "fact") == "The sky is blue"

    def test_update_existing(self):
        mem = InMemoryPersistence()
        agent_id = uuid4()
        mem.write(agent_id, "fact", "old")
        mem.update(agent_id, "fact", "new")
        assert mem.read(agent_id, "fact") == "new"

    def test_delete(self):
        mem = InMemoryPersistence()
        agent_id = uuid4()
        mem.write(agent_id, "fact", "value")
        mem.delete(agent_id, "fact")
        assert mem.read(agent_id, "fact") is None

    def test_search(self):
        mem = InMemoryPersistence()
        agent_id = uuid4()
        mem.write(agent_id, "color", "blue")
        mem.write(agent_id, "shape", "circle")
        results = mem.search(agent_id, "blue")
        assert len(results) == 1
        assert results[0].value == "blue"

    def test_list_keys(self):
        mem = InMemoryPersistence()
        agent_id = uuid4()
        mem.write(agent_id, "a", "1")
        mem.write(agent_id, "b", "2")
        assert sorted(mem.list_keys(agent_id)) == ["a", "b"]
