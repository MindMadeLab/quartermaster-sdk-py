"""Tests for StartNodeV1 — flow entry point."""

import pytest

from tests.conftest import MockNodeContext
from qm_nodes.nodes.control_flow.start import StartNodeV1
from qm_nodes.enums import (
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)


class TestStartNodeInfo:
    """Tests for StartNodeV1 class metadata."""

    def test_name(self):
        assert StartNodeV1.name() == "StartNode"

    def test_version(self):
        assert StartNodeV1.version() == "1.0"

    def test_info_description(self):
        info = StartNodeV1.info()
        assert info.description
        assert "entry" in info.description.lower() or "start" in info.description.lower()

    def test_info_empty_metadata(self):
        info = StartNodeV1.info()
        assert info.metadata == {}

    def test_info_version_matches(self):
        info = StartNodeV1.info()
        assert info.version == "1.0"


class TestStartNodeFlowConfig:
    """Tests for StartNodeV1.flow_config()."""

    def test_traverse_in(self):
        config = StartNodeV1.flow_config()
        assert config.traverse_in == AvailableTraversingIn.AwaitFirst

    def test_traverse_out(self):
        config = StartNodeV1.flow_config()
        assert config.traverse_out == AvailableTraversingOut.SpawnAll

    def test_thought_type_skip(self):
        config = StartNodeV1.flow_config()
        assert config.thought_type == AvailableThoughtTypes.SkipThought1

    def test_message_type_variable(self):
        config = StartNodeV1.flow_config()
        assert config.message_type == AvailableMessageTypes.Variable

    def test_does_not_accept_incoming_edges(self):
        config = StartNodeV1.flow_config()
        assert config.accepts_incoming_edges is False

    def test_accepts_outgoing_edges(self):
        config = StartNodeV1.flow_config()
        assert config.accepts_outgoing_edges is True

    def test_config_validation_passes(self):
        config = StartNodeV1.flow_config()
        config.validate()


class TestStartNodeThink:
    """Tests for StartNodeV1.think() execution."""

    def test_think_without_initializer(self):
        ctx = MockNodeContext()
        StartNodeV1.think(ctx)  # Should not raise

    def test_think_with_memory_initializer(self):
        initialized = []
        ctx = MockNodeContext(
            node_metadata={
                "_memory_initializer": lambda c: initialized.append(c)
            }
        )
        StartNodeV1.think(ctx)
        assert len(initialized) == 1
        assert initialized[0] is ctx

    def test_think_initializer_called_once(self):
        call_count = 0

        def counting_initializer(c):
            nonlocal call_count
            call_count += 1

        ctx = MockNodeContext(
            node_metadata={"_memory_initializer": counting_initializer}
        )
        StartNodeV1.think(ctx)
        assert call_count == 1

    def test_think_with_none_initializer(self):
        ctx = MockNodeContext(node_metadata={"_memory_initializer": None})
        StartNodeV1.think(ctx)  # Should not raise

    def test_think_with_empty_metadata(self):
        ctx = MockNodeContext(node_metadata={})
        StartNodeV1.think(ctx)  # Should not raise

    def test_think_initializer_receives_context(self):
        received_ctx = []
        ctx = MockNodeContext(
            node_metadata={
                "_memory_initializer": lambda c: received_ctx.append(c)
            }
        )
        StartNodeV1.think(ctx)
        assert received_ctx[0] is ctx

    def test_think_initializer_error_propagates(self):
        def failing_initializer(c):
            raise RuntimeError("Init failed")

        ctx = MockNodeContext(
            node_metadata={"_memory_initializer": failing_initializer}
        )
        with pytest.raises(RuntimeError, match="Init failed"):
            StartNodeV1.think(ctx)
