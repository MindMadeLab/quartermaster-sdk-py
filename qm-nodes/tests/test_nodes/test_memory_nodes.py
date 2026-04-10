"""Tests for memory nodes."""

from tests.conftest import MockNodeContext, MockThought


class TestFlowMemoryNode:
    def test_info(self):
        from qm_nodes.nodes.memory.flow_memory import FlowMemoryNode

        info = FlowMemoryNode.info()
        assert "memory" in info.description.lower()

    def test_config_no_edges(self):
        from qm_nodes.nodes.memory.flow_memory import FlowMemoryNode

        config = FlowMemoryNode.flow_config()
        assert not config.accepts_incoming_edges
        assert not config.accepts_outgoing_edges


class TestReadMemoryNode:
    def test_calls_memory_reader(self):
        from qm_nodes.nodes.memory.read_memory import ReadMemoryNode

        read_calls = []

        def mock_reader(name, mtype, vars, ctx):
            read_calls.append((name, mtype, vars))
            return {"loaded_var": 42}

        ctx = MockNodeContext(
            node_metadata={
                "memory_name": "test_mem",
                "memory_type": "flow",
                "variable_names": ["loaded_var"],
                "_memory_reader": mock_reader,
            }
        )
        ReadMemoryNode.think(ctx)
        assert len(read_calls) == 1
        assert read_calls[0][0] == "test_mem"
        assert ctx.handle.last_metadata_update == {"loaded_var": 42}


class TestWriteMemoryNode:
    def test_calls_memory_writer(self):
        from qm_nodes.nodes.memory.write_memory import WriteMemoryNode

        write_calls = []

        def mock_writer(name, mtype, data, ctx):
            write_calls.append((name, mtype, data))

        ctx = MockNodeContext(
            node_metadata={
                "memory_name": "test_mem",
                "memory_type": "flow",
                "variables": [{"name": "x", "expression": "a + b"}],
                "_memory_writer": mock_writer,
            },
            thought=MockThought(metadata={"a": 3, "b": 7}),
        )
        WriteMemoryNode.think(ctx)
        assert len(write_calls) == 1
        assert write_calls[0][2] == {"x": 10}


class TestUpdateMemoryNode:
    def test_calls_memory_updater(self):
        from qm_nodes.nodes.memory.update_memory import UpdateMemoryNode

        update_calls = []

        def mock_updater(name, mtype, data, ctx):
            update_calls.append((name, data))

        ctx = MockNodeContext(
            node_metadata={
                "memory_name": "counter",
                "variables": [{"name": "count", "expression": "count + 1"}],
                "_memory_updater": mock_updater,
            },
            thought=MockThought(metadata={"count": 5}),
        )
        UpdateMemoryNode.think(ctx)
        assert len(update_calls) == 1
        assert update_calls[0][1] == {"count": 6}


class TestUserMemoryNode:
    def test_info(self):
        from qm_nodes.nodes.memory.user_memory import UserMemoryNode

        info = UserMemoryNode.info()
        assert "memory" in info.description.lower()
