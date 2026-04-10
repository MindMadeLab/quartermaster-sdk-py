"""Tests for configuration models."""

import pytest
from uuid import uuid4

from quartermaster_nodes.config import AssistantInfo, FlowNodeConf, FlowRunConfig
from quartermaster_nodes.enums import (
    AvailableErrorHandlingStrategies,
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)


class TestAssistantInfo:
    def test_defaults(self):
        info = AssistantInfo()
        assert info.version == ""
        assert info.description == ""
        assert info.instructions == ""

    def test_assignment(self):
        info = AssistantInfo()
        info.version = "1.0"
        info.description = "Test"
        info.metadata = {"key": "val"}
        assert info.version == "1.0"
        assert info.metadata["key"] == "val"


class TestFlowNodeConf:
    def test_basic_creation(self):
        conf = FlowNodeConf(
            traverse_in=AvailableTraversingIn.AwaitFirst,
            traverse_out=AvailableTraversingOut.SpawnAll,
            thought_type=AvailableThoughtTypes.NewThought1,
            message_type=AvailableMessageTypes.Assistant,
        )
        assert conf.traverse_in == AvailableTraversingIn.AwaitFirst
        assert conf.traverse_out == AvailableTraversingOut.SpawnAll

    def test_auto_adds_defaults_to_available(self):
        conf = FlowNodeConf(
            traverse_in=AvailableTraversingIn.AwaitFirst,
            traverse_out=AvailableTraversingOut.SpawnAll,
            thought_type=AvailableThoughtTypes.NewThought1,
            message_type=AvailableMessageTypes.Assistant,
        )
        assert AvailableThoughtTypes.NewThought1 in conf.available_thought_types
        assert AvailableTraversingIn.AwaitFirst in conf.available_traversing_in

    def test_custom_available_types(self):
        conf = FlowNodeConf(
            traverse_in=AvailableTraversingIn.AwaitFirst,
            traverse_out=AvailableTraversingOut.SpawnAll,
            thought_type=AvailableThoughtTypes.NewThought1,
            message_type=AvailableMessageTypes.Assistant,
            available_thought_types={AvailableThoughtTypes.EditSameOrAddNew1},
        )
        assert AvailableThoughtTypes.NewThought1 in conf.available_thought_types
        assert AvailableThoughtTypes.EditSameOrAddNew1 in conf.available_thought_types

    def test_default_error_handling(self):
        conf = FlowNodeConf(
            traverse_in=AvailableTraversingIn.AwaitFirst,
            traverse_out=AvailableTraversingOut.SpawnAll,
            thought_type=AvailableThoughtTypes.NewThought1,
            message_type=AvailableMessageTypes.Assistant,
        )
        assert AvailableErrorHandlingStrategies.Stop in conf.available_error_handling_strategies
        assert AvailableErrorHandlingStrategies.Continue in conf.available_error_handling_strategies

    def test_asdict(self):
        conf = FlowNodeConf(
            traverse_in=AvailableTraversingIn.AwaitFirst,
            traverse_out=AvailableTraversingOut.SpawnAll,
            thought_type=AvailableThoughtTypes.NewThought1,
            message_type=AvailableMessageTypes.Assistant,
        )
        d = conf.asdict()
        assert d["traverse_in"] == "AwaitFirst"
        assert d["traverse_out"] == "SpawnAll"
        assert d["thought_type"] == "NewThought1"

    def test_from_dict(self):
        d = {
            "traverse_in": "AwaitFirst",
            "traverse_out": "SpawnAll",
            "thought_type": "NewThought1",
            "message_type": "Assistant",
        }
        conf = FlowNodeConf.from_dict(d)
        assert conf.traverse_in == AvailableTraversingIn.AwaitFirst

    def test_validation(self):
        conf = FlowNodeConf(
            traverse_in=AvailableTraversingIn.AwaitFirst,
            traverse_out=AvailableTraversingOut.SpawnAll,
            thought_type=AvailableThoughtTypes.NewThought1,
            message_type=AvailableMessageTypes.Assistant,
        )
        conf.validate()  # Should not raise

    def test_is_valid_methods(self):
        conf = FlowNodeConf(
            traverse_in=AvailableTraversingIn.AwaitFirst,
            traverse_out=AvailableTraversingOut.SpawnAll,
            thought_type=AvailableThoughtTypes.NewThought1,
            message_type=AvailableMessageTypes.Assistant,
        )
        assert conf.is_valid_thought_type("NewThought1")
        assert not conf.is_valid_thought_type("SkipThought1")

    def test_accepts_edges_defaults(self):
        conf = FlowNodeConf(
            traverse_in=AvailableTraversingIn.AwaitFirst,
            traverse_out=AvailableTraversingOut.SpawnAll,
            thought_type=AvailableThoughtTypes.NewThought1,
            message_type=AvailableMessageTypes.Assistant,
        )
        assert conf.accepts_incoming_edges
        assert conf.accepts_outgoing_edges

    def test_no_incoming_edges(self):
        conf = FlowNodeConf(
            traverse_in=AvailableTraversingIn.AwaitFirst,
            traverse_out=AvailableTraversingOut.SpawnAll,
            thought_type=AvailableThoughtTypes.SkipThought1,
            message_type=AvailableMessageTypes.Variable,
            accepts_incoming_edges=False,
        )
        assert not conf.accepts_incoming_edges


class TestFlowRunConfig:
    def test_creation(self):
        uid = uuid4()
        config = FlowRunConfig(
            assistant_node_id=uid,
            user_id=uid,
            chat_id=uid,
        )
        assert config.assistant_node_id == uid
        assert config.scheduled is False

    def test_copy(self):
        uid = uuid4()
        config = FlowRunConfig(assistant_node_id=uid, user_id=uid, chat_id=uid)
        copy = config.copy()
        assert copy.assistant_node_id == config.assistant_node_id
        assert copy is not config

    def test_serialization_roundtrip(self):
        uid1, uid2, uid3 = uuid4(), uuid4(), uuid4()
        config = FlowRunConfig(
            assistant_node_id=uid1,
            user_id=uid2,
            chat_id=uid3,
            scheduled=True,
        )
        d = config.asdict()
        restored = FlowRunConfig.from_dict(d)
        assert restored.assistant_node_id == uid1
        assert restored.user_id == uid2
        assert restored.chat_id == uid3
        assert restored.scheduled is True

    def test_none_uuid_serialization(self):
        uid = uuid4()
        config = FlowRunConfig(
            assistant_node_id=uid,
            user_id=uid,
            chat_id=uid,
            environment_id=None,
        )
        d = config.asdict()
        assert d["environment_id"] is None

    def test_str(self):
        uid = uuid4()
        config = FlowRunConfig(assistant_node_id=uid, user_id=uid, chat_id=uid)
        s = str(config)
        assert "assistant node id" in s
