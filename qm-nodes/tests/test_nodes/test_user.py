"""Tests for UserNode1 (User1) — pause flow and await user input."""

from uuid import uuid4

import pytest

from tests.conftest import MockNodeContext
from qm_nodes.nodes.user_interaction.user import UserNode1
from qm_nodes.enums import (
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)
from qm_nodes.exceptions import MissingMemoryIDException, ProcessStopException


class TestUserNodeInfo:
    """Tests for UserNode1 class metadata."""

    def test_name(self):
        assert UserNode1.name() == "UserAssistant1"

    def test_version(self):
        assert UserNode1.version() == "1.0.0"

    def test_info_description(self):
        info = UserNode1.info()
        assert info.description
        assert "user" in info.description.lower() or "pause" in info.description.lower()

    def test_info_metadata_keys(self):
        info = UserNode1.info()
        assert "text_snippets" in info.metadata

    def test_default_text_snippets(self):
        assert UserNode1.metadata_text_snippets_default == []


class TestUserNodeFlowConfig:
    """Tests for UserNode1.flow_config()."""

    def test_traverse_in(self):
        config = UserNode1.flow_config()
        assert config.traverse_in == AvailableTraversingIn.AwaitFirst

    def test_traverse_out(self):
        config = UserNode1.flow_config()
        assert config.traverse_out == AvailableTraversingOut.SpawnAll

    def test_thought_type_new(self):
        config = UserNode1.flow_config()
        assert config.thought_type == AvailableThoughtTypes.NewThought1

    def test_message_type_user(self):
        config = UserNode1.flow_config()
        assert config.message_type == AvailableMessageTypes.User

    def test_available_thought_types(self):
        config = UserNode1.flow_config()
        assert AvailableThoughtTypes.EditSameOrAddNew1 in config.available_thought_types

    def test_available_traversing_out(self):
        config = UserNode1.flow_config()
        assert AvailableTraversingOut.SpawnNone in config.available_traversing_out

    def test_config_validation_passes(self):
        config = UserNode1.flow_config()
        config.validate()


class TestUserNodeThink:
    """Tests for UserNode1.think() execution."""

    def test_raises_process_stop(self):
        ctx = MockNodeContext()
        with pytest.raises(ProcessStopException, match="Awaiting user input"):
            UserNode1.think(ctx)

    def test_raises_missing_memory_before_stop(self):
        ctx = MockNodeContext(thought_id=None)
        with pytest.raises(MissingMemoryIDException):
            UserNode1.think(ctx)

    def test_missing_memory_takes_priority(self):
        """MissingMemoryIDException should be raised before ProcessStopException."""
        ctx = MockNodeContext(thought_id=None)
        with pytest.raises(MissingMemoryIDException):
            UserNode1.think(ctx)

    def test_calls_status_updater(self):
        updates = []
        flow_node_id = uuid4()
        ctx = MockNodeContext(
            node_metadata={
                "_status_updater": lambda fid, status: updates.append((fid, status))
            },
            flow_node_id=flow_node_id,
        )

        with pytest.raises(ProcessStopException):
            UserNode1.think(ctx)

        assert len(updates) == 1
        assert updates[0][0] == flow_node_id
        assert updates[0][1] == "AwaitingUserResponse"

    def test_status_updater_receives_flow_node_id(self):
        received_ids = []
        fid = uuid4()
        ctx = MockNodeContext(
            node_metadata={
                "_status_updater": lambda fid, status: received_ids.append(fid)
            },
            flow_node_id=fid,
        )

        with pytest.raises(ProcessStopException):
            UserNode1.think(ctx)

        assert received_ids[0] == fid

    def test_no_status_updater(self):
        """Without a status updater, should still raise ProcessStopException."""
        ctx = MockNodeContext()
        with pytest.raises(ProcessStopException):
            UserNode1.think(ctx)

    def test_status_updater_none(self):
        ctx = MockNodeContext(node_metadata={"_status_updater": None})
        with pytest.raises(ProcessStopException):
            UserNode1.think(ctx)

    def test_does_not_modify_handle(self):
        ctx = MockNodeContext()
        with pytest.raises(ProcessStopException):
            UserNode1.think(ctx)
        assert ctx.handle.texts == []
        assert ctx.handle.metadata_updates == []
