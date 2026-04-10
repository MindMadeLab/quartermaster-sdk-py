"""Tests for FlowMemoryNode (FlowMemory1) — flow-scoped persistent memory."""

import pytest

from tests.conftest import MockNodeContext
from quartermaster_nodes.nodes.memory.flow_memory import FlowMemoryNode
from quartermaster_nodes.enums import (
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)


class TestFlowMemoryInfo:
    """Tests for FlowMemoryNode class metadata."""

    def test_name(self):
        assert FlowMemoryNode.name() == "FlowMemory"

    def test_version(self):
        assert FlowMemoryNode.version() == "1.0"

    def test_info_description(self):
        info = FlowMemoryNode.info()
        assert "memory" in info.description.lower()

    def test_info_metadata_keys(self):
        info = FlowMemoryNode.info()
        assert "memory_name" in info.metadata
        assert "initial_data" in info.metadata

    def test_default_memory_name(self):
        assert FlowMemoryNode.metadata_memory_name_default == "default"

    def test_default_initial_data(self):
        assert FlowMemoryNode.metadata_initial_data_default == []

    def test_info_version_matches(self):
        info = FlowMemoryNode.info()
        assert info.version == "1.0"

    def test_info_has_instructions(self):
        info = FlowMemoryNode.info()
        assert info.instructions


class TestFlowMemoryFlowConfig:
    """Tests for FlowMemoryNode.flow_config()."""

    def test_traverse_in(self):
        config = FlowMemoryNode.flow_config()
        assert config.traverse_in == AvailableTraversingIn.AwaitFirst

    def test_traverse_out(self):
        config = FlowMemoryNode.flow_config()
        assert config.traverse_out == AvailableTraversingOut.SpawnAll

    def test_thought_type_skip(self):
        config = FlowMemoryNode.flow_config()
        assert config.thought_type == AvailableThoughtTypes.SkipThought1

    def test_message_type_variable(self):
        config = FlowMemoryNode.flow_config()
        assert config.message_type == AvailableMessageTypes.Variable

    def test_does_not_accept_incoming_edges(self):
        config = FlowMemoryNode.flow_config()
        assert config.accepts_incoming_edges is False

    def test_does_not_accept_outgoing_edges(self):
        config = FlowMemoryNode.flow_config()
        assert config.accepts_outgoing_edges is False

    def test_config_validation_passes(self):
        config = FlowMemoryNode.flow_config()
        config.validate()

    def test_config_serialization(self):
        config = FlowMemoryNode.flow_config()
        d = config.asdict()
        assert d["thought_type"] == "SkipThought1"
        assert d["accepts_incoming_edges"] is False
        assert d["accepts_outgoing_edges"] is False


class TestFlowMemoryThink:
    """Tests for FlowMemoryNode.think() execution."""

    def test_think_does_nothing(self):
        ctx = MockNodeContext()
        result = FlowMemoryNode.think(ctx)
        assert result is None

    def test_think_does_not_modify_handle(self):
        ctx = MockNodeContext()
        FlowMemoryNode.think(ctx)
        assert ctx.handle.texts == []
        assert ctx.handle.metadata_updates == []

    def test_think_with_empty_metadata(self):
        ctx = MockNodeContext(node_metadata={})
        FlowMemoryNode.think(ctx)  # Should not raise

    def test_think_with_custom_metadata(self):
        ctx = MockNodeContext(node_metadata={
            "memory_name": "custom_mem",
            "initial_data": [{"key": "val"}],
        })
        FlowMemoryNode.think(ctx)  # Should not raise; logic handled by StartNode

    def test_think_with_none_thought(self):
        ctx = MockNodeContext(thought=None, thought_id=None)
        FlowMemoryNode.think(ctx)  # Should not raise


class TestFlowMemoryMetadataAccess:
    """Tests for metadata retrieval via get_metadata_key_value."""

    def test_get_memory_name_default(self):
        ctx = MockNodeContext()
        val = FlowMemoryNode.get_metadata_key_value(
            ctx, "memory_name", "default"
        )
        assert val == "default"

    def test_get_memory_name_custom(self):
        ctx = MockNodeContext(node_metadata={"memory_name": "my_memory"})
        val = FlowMemoryNode.get_metadata_key_value(
            ctx, "memory_name", "default"
        )
        assert val == "my_memory"

    def test_get_initial_data_default(self):
        ctx = MockNodeContext()
        val = FlowMemoryNode.get_metadata_key_value(
            ctx, "initial_data", []
        )
        assert val == []

    def test_get_initial_data_custom(self):
        data = [{"name": "counter", "value": 0}]
        ctx = MockNodeContext(node_metadata={"initial_data": data})
        val = FlowMemoryNode.get_metadata_key_value(
            ctx, "initial_data", []
        )
        assert val == data

    def test_store_metadata(self):
        ctx = MockNodeContext(node_metadata={})
        FlowMemoryNode.store_metadata_key_value(ctx, "test_key", "test_val")
        assert ctx.node_metadata["test_key"] == "test_val"
