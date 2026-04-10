"""Tests for StaticNode1 (Static1) — static text content output."""

import pytest

from tests.conftest import MockNodeContext, MockThought, MockHandle
from qm_nodes.nodes.data.static import StaticNode1
from qm_nodes.enums import (
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)


class TestStaticNodeInfo:
    """Tests for StaticNode1 class metadata."""

    def test_name(self):
        assert StaticNode1.name() == "StaticAssistant"

    def test_version(self):
        assert StaticNode1.version() == "1.0"

    def test_info_description(self):
        info = StaticNode1.info()
        assert info.description
        assert "static" in info.description.lower()

    def test_info_metadata_keys(self):
        info = StaticNode1.info()
        assert "static_text" in info.metadata

    def test_default_static_text(self):
        assert StaticNode1.metadata_static_text_default == "This is a default static text."

    def test_info_version_matches(self):
        info = StaticNode1.info()
        assert info.version == "1.0"


class TestStaticNodeFlowConfig:
    """Tests for StaticNode1.flow_config()."""

    def test_traverse_in(self):
        config = StaticNode1.flow_config()
        assert config.traverse_in == AvailableTraversingIn.AwaitFirst

    def test_traverse_out(self):
        config = StaticNode1.flow_config()
        assert config.traverse_out == AvailableTraversingOut.SpawnAll

    def test_thought_type_new(self):
        config = StaticNode1.flow_config()
        assert config.thought_type == AvailableThoughtTypes.NewThought1

    def test_message_type_automatic(self):
        config = StaticNode1.flow_config()
        assert config.message_type == AvailableMessageTypes.Automatic

    def test_available_thought_types(self):
        config = StaticNode1.flow_config()
        assert AvailableThoughtTypes.EditSameOrAddNew1 in config.available_thought_types
        assert AvailableThoughtTypes.UsePreviousThought1 in config.available_thought_types
        assert AvailableThoughtTypes.NewHiddenThought1 in config.available_thought_types
        assert AvailableThoughtTypes.NewCollapsedThought1 in config.available_thought_types

    def test_available_message_types(self):
        config = StaticNode1.flow_config()
        assert AvailableMessageTypes.Assistant in config.available_message_types
        assert AvailableMessageTypes.User in config.available_message_types

    def test_accepts_edges(self):
        config = StaticNode1.flow_config()
        assert config.accepts_incoming_edges is True
        assert config.accepts_outgoing_edges is True

    def test_config_validation_passes(self):
        config = StaticNode1.flow_config()
        config.validate()


class TestStaticNodeThink:
    """Tests for StaticNode1.think() execution."""

    def test_outputs_default_text(self):
        ctx = MockNodeContext()
        StaticNode1.think(ctx)
        assert ctx.handle.last_text == "This is a default static text."

    def test_outputs_custom_text(self):
        ctx = MockNodeContext(node_metadata={"static_text": "Hello, World!"})
        StaticNode1.think(ctx)
        assert ctx.handle.last_text == "Hello, World!"

    def test_outputs_empty_string(self):
        ctx = MockNodeContext(node_metadata={"static_text": ""})
        StaticNode1.think(ctx)
        assert ctx.handle.last_text == ""

    def test_outputs_multiline_text(self):
        text = "Line 1\nLine 2\nLine 3"
        ctx = MockNodeContext(node_metadata={"static_text": text})
        StaticNode1.think(ctx)
        assert ctx.handle.last_text == text

    def test_outputs_text_with_special_characters(self):
        text = "Price: $100 & <special> \"chars\""
        ctx = MockNodeContext(node_metadata={"static_text": text})
        StaticNode1.think(ctx)
        assert ctx.handle.last_text == text

    def test_appends_to_handle(self):
        ctx = MockNodeContext(node_metadata={"static_text": "test"})
        StaticNode1.think(ctx)
        assert len(ctx.handle.texts) == 1
        assert ctx.handle.all_text == "test"


class TestStaticNodeErrors:
    """Tests for StaticNode1 error handling."""

    def test_raises_without_thought(self):
        ctx = MockNodeContext(thought=None)
        with pytest.raises(ValueError, match="Memory ID cannot be None"):
            StaticNode1.think(ctx)

    def test_raises_with_none_handle(self):
        ctx = MockNodeContext(handle=None)
        with pytest.raises(AssertionError, match="handle not set"):
            StaticNode1.think(ctx)


class TestStaticNodeMetadataAccess:
    """Tests for metadata retrieval."""

    def test_get_static_text_default(self):
        ctx = MockNodeContext()
        val = StaticNode1.get_metadata_key_value(
            ctx, "static_text", StaticNode1.metadata_static_text_default
        )
        assert val == "This is a default static text."

    def test_get_static_text_custom(self):
        ctx = MockNodeContext(node_metadata={"static_text": "Custom"})
        val = StaticNode1.get_metadata_key_value(
            ctx, "static_text", StaticNode1.metadata_static_text_default
        )
        assert val == "Custom"
