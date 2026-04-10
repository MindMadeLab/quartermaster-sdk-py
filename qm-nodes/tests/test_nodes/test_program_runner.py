"""Tests for ProgramRunner1 — execute a registered tool/program."""

import pytest

from tests.conftest import MockNodeContext, MockHandle
from qm_nodes.nodes.data.program_runner import ProgramRunner1
from qm_nodes.enums import (
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)


class TestProgramRunnerInfo:
    """Tests for ProgramRunner1 class metadata."""

    def test_name(self):
        assert ProgramRunner1.name() == "ProgramRunner1"

    def test_version(self):
        assert ProgramRunner1.version() == "1.0"

    def test_info_description(self):
        info = ProgramRunner1.info()
        assert info.description
        assert "tool" in info.description.lower() or "program" in info.description.lower()

    def test_info_metadata_keys(self):
        info = ProgramRunner1.info()
        assert "program_version_id" in info.metadata
        assert "parameters" in info.metadata

    def test_default_program_id(self):
        assert ProgramRunner1.metadata_program_version_id_default is None

    def test_default_parameters(self):
        assert ProgramRunner1.metadata_parameters_default == {}

    def test_info_version_matches(self):
        info = ProgramRunner1.info()
        assert info.version == "1.0"


class TestProgramRunnerFlowConfig:
    """Tests for ProgramRunner1.flow_config()."""

    def test_traverse_in(self):
        config = ProgramRunner1.flow_config()
        assert config.traverse_in == AvailableTraversingIn.AwaitFirst

    def test_traverse_out(self):
        config = ProgramRunner1.flow_config()
        assert config.traverse_out == AvailableTraversingOut.SpawnAll

    def test_thought_type_new(self):
        config = ProgramRunner1.flow_config()
        assert config.thought_type == AvailableThoughtTypes.NewThought1

    def test_message_type_tool(self):
        config = ProgramRunner1.flow_config()
        assert config.message_type == AvailableMessageTypes.Tool

    def test_available_thought_types(self):
        config = ProgramRunner1.flow_config()
        assert AvailableThoughtTypes.EditSameOrAddNew1 in config.available_thought_types
        assert AvailableThoughtTypes.UsePreviousThought1 in config.available_thought_types
        assert AvailableThoughtTypes.NewHiddenThought1 in config.available_thought_types

    def test_config_validation_passes(self):
        config = ProgramRunner1.flow_config()
        config.validate()


class TestProgramRunnerThink:
    """Tests for ProgramRunner1.think() execution."""

    def test_calls_executor_with_correct_args(self):
        called_with = []

        def mock_executor(pid, params, ctx):
            called_with.append((pid, params))
            return None

        ctx = MockNodeContext(
            node_metadata={
                "program_version_id": "prog-42",
                "parameters": {"input": "data"},
                "_program_executor": mock_executor,
            }
        )

        ProgramRunner1.think(ctx)

        assert len(called_with) == 1
        assert called_with[0][0] == "prog-42"
        assert called_with[0][1] == {"input": "data"}

    def test_appends_result_to_handle(self):
        def mock_executor(pid, params, ctx):
            return "Execution result"

        ctx = MockNodeContext(
            node_metadata={
                "program_version_id": "prog-1",
                "parameters": {},
                "_program_executor": mock_executor,
            }
        )

        ProgramRunner1.think(ctx)
        assert ctx.handle.last_text == "Execution result"

    def test_no_append_when_result_is_none(self):
        def mock_executor(pid, params, ctx):
            return None

        ctx = MockNodeContext(
            node_metadata={
                "program_version_id": "prog-1",
                "parameters": {},
                "_program_executor": mock_executor,
            }
        )

        ProgramRunner1.think(ctx)
        assert ctx.handle.texts == []

    def test_no_executor(self):
        """Without an executor, think() should do nothing."""
        ctx = MockNodeContext(
            node_metadata={
                "program_version_id": "prog-1",
                "parameters": {},
            }
        )
        ProgramRunner1.think(ctx)
        assert ctx.handle.texts == []

    def test_no_program_id(self):
        """Without a program ID, think() should do nothing."""
        called = []

        def mock_executor(pid, params, ctx):
            called.append(True)
            return "result"

        ctx = MockNodeContext(
            node_metadata={
                "parameters": {},
                "_program_executor": mock_executor,
            }
        )
        ProgramRunner1.think(ctx)
        assert called == []

    def test_executor_with_none_handle(self):
        """If handle is None and result is returned, no append should happen."""
        def mock_executor(pid, params, ctx):
            return "result"

        ctx = MockNodeContext(
            node_metadata={
                "program_version_id": "prog-1",
                "parameters": {},
                "_program_executor": mock_executor,
            },
            handle=None,
        )
        ProgramRunner1.think(ctx)
        # Should not raise; the code guards with `ctx.handle is not None`

    def test_result_converted_to_string(self):
        def mock_executor(pid, params, ctx):
            return 42

        ctx = MockNodeContext(
            node_metadata={
                "program_version_id": "prog-1",
                "parameters": {},
                "_program_executor": mock_executor,
            }
        )

        ProgramRunner1.think(ctx)
        assert ctx.handle.last_text == "42"

    def test_result_dict_converted_to_string(self):
        def mock_executor(pid, params, ctx):
            return {"key": "value"}

        ctx = MockNodeContext(
            node_metadata={
                "program_version_id": "prog-1",
                "parameters": {},
                "_program_executor": mock_executor,
            }
        )

        ProgramRunner1.think(ctx)
        assert ctx.handle.last_text == "{'key': 'value'}"

    def test_default_parameters_when_not_set(self):
        called_params = []

        def mock_executor(pid, params, ctx):
            called_params.append(params)
            return None

        ctx = MockNodeContext(
            node_metadata={
                "program_version_id": "prog-1",
                "_program_executor": mock_executor,
            }
        )

        ProgramRunner1.think(ctx)
        assert called_params[0] == {}

    def test_executor_receives_context(self):
        received_ctx = []

        def mock_executor(pid, params, ctx):
            received_ctx.append(ctx)
            return None

        ctx = MockNodeContext(
            node_metadata={
                "program_version_id": "prog-1",
                "parameters": {},
                "_program_executor": mock_executor,
            }
        )

        ProgramRunner1.think(ctx)
        assert received_ctx[0] is ctx

    def test_executor_error_propagates(self):
        def failing_executor(pid, params, ctx):
            raise RuntimeError("Tool execution failed")

        ctx = MockNodeContext(
            node_metadata={
                "program_version_id": "prog-1",
                "parameters": {},
                "_program_executor": failing_executor,
            }
        )

        with pytest.raises(RuntimeError, match="Tool execution failed"):
            ProgramRunner1.think(ctx)
