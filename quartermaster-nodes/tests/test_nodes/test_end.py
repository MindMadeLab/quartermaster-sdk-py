"""Tests for EndNodeV1 — flow termination."""

import pytest

from tests.conftest import MockNodeContext
from quartermaster_nodes.nodes.control_flow.end import EndNodeV1
from quartermaster_nodes.enums import (
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)


class TestEndNodeInfo:
    """Tests for EndNodeV1 class metadata."""

    def test_name(self):
        assert EndNodeV1.name() == "EndNode"

    def test_version(self):
        assert EndNodeV1.version() == "1.0"

    def test_info_description(self):
        info = EndNodeV1.info()
        assert info.description
        assert "end" in info.description.lower() or "spawn" in info.description.lower()

    def test_info_empty_metadata(self):
        info = EndNodeV1.info()
        assert info.metadata == {}

    def test_info_version_matches(self):
        info = EndNodeV1.info()
        assert info.version == "1.0"

    def test_info_has_instructions(self):
        info = EndNodeV1.info()
        assert info.instructions


class TestEndNodeFlowConfig:
    """Tests for EndNodeV1.flow_config()."""

    def test_traverse_in(self):
        config = EndNodeV1.flow_config()
        assert config.traverse_in == AvailableTraversingIn.AwaitFirst

    def test_traverse_out_spawn_start(self):
        config = EndNodeV1.flow_config()
        assert config.traverse_out == AvailableTraversingOut.SpawnStart

    def test_thought_type_skip(self):
        config = EndNodeV1.flow_config()
        assert config.thought_type == AvailableThoughtTypes.SkipThought1

    def test_message_type_variable(self):
        config = EndNodeV1.flow_config()
        assert config.message_type == AvailableMessageTypes.Variable

    def test_does_not_accept_outgoing_edges(self):
        config = EndNodeV1.flow_config()
        assert config.accepts_outgoing_edges is False

    def test_accepts_incoming_edges(self):
        config = EndNodeV1.flow_config()
        assert config.accepts_incoming_edges is True

    def test_config_validation_passes(self):
        config = EndNodeV1.flow_config()
        config.validate()

    def test_config_serialization(self):
        config = EndNodeV1.flow_config()
        d = config.asdict()
        assert d["traverse_out"] == "SpawnStart"
        assert d["accepts_outgoing_edges"] is False
        assert d["accepts_incoming_edges"] is True


class TestEndNodeThink:
    """Tests for EndNodeV1.think() execution."""

    def test_think_does_nothing(self):
        ctx = MockNodeContext()
        result = EndNodeV1.think(ctx)
        assert result is None

    def test_think_with_empty_metadata(self):
        ctx = MockNodeContext(node_metadata={})
        EndNodeV1.think(ctx)  # Should not raise

    def test_think_does_not_modify_handle(self):
        ctx = MockNodeContext()
        EndNodeV1.think(ctx)
        assert ctx.handle.texts == []
        assert ctx.handle.metadata_updates == []

    def test_think_with_none_thought(self):
        ctx = MockNodeContext(thought=None, thought_id=None)
        EndNodeV1.think(ctx)  # Should not raise
