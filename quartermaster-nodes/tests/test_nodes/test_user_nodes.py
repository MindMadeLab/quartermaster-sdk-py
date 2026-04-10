"""Tests for user interaction nodes."""

import pytest
from uuid import uuid4

from tests.conftest import MockNodeContext
from quartermaster_nodes.exceptions import MissingMemoryIDException, ProcessStopException


class TestUserNode:
    def test_raises_process_stop(self):
        from quartermaster_nodes.nodes.user_interaction.user import UserNode1

        ctx = MockNodeContext()
        with pytest.raises(ProcessStopException):
            UserNode1.think(ctx)

    def test_raises_without_thought_id(self):
        from quartermaster_nodes.nodes.user_interaction.user import UserNode1

        ctx = MockNodeContext(thought_id=None)
        with pytest.raises(MissingMemoryIDException):
            UserNode1.think(ctx)

    def test_calls_status_updater(self):
        from quartermaster_nodes.nodes.user_interaction.user import UserNode1

        updates = []
        ctx = MockNodeContext(
            node_metadata={
                "_status_updater": lambda fid, status: updates.append((fid, status))
            }
        )

        with pytest.raises(ProcessStopException):
            UserNode1.think(ctx)

        assert len(updates) == 1
        assert updates[0][1] == "AwaitingUserResponse"


class TestUserDecision:
    def test_raises_process_stop(self):
        from quartermaster_nodes.nodes.user_interaction.user_decision import UserDecisionV1

        ctx = MockNodeContext()
        with pytest.raises(ProcessStopException):
            UserDecisionV1.think(ctx)

    def test_raises_without_thought_id(self):
        from quartermaster_nodes.nodes.user_interaction.user_decision import UserDecisionV1

        ctx = MockNodeContext(thought_id=None)
        with pytest.raises(MissingMemoryIDException):
            UserDecisionV1.think(ctx)


class TestUserForm:
    def test_raises_process_stop(self):
        from quartermaster_nodes.nodes.user_interaction.user_form import UserFormV1

        ctx = MockNodeContext()
        with pytest.raises(ProcessStopException):
            UserFormV1.think(ctx)

    def test_has_parameters_metadata(self):
        from quartermaster_nodes.nodes.user_interaction.user_form import UserFormV1

        info = UserFormV1.info()
        assert "parameters" in info.metadata


class TestUserProgramForm:
    def test_raises_process_stop(self):
        from quartermaster_nodes.nodes.user_interaction.user_program_form import UserProgramFormV1

        ctx = MockNodeContext()
        with pytest.raises(ProcessStopException):
            UserProgramFormV1.think(ctx)
