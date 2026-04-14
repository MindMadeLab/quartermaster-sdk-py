"""Tests for LLM nodes."""

import pytest

from tests.conftest import MockNodeContext, MockThought
from quartermaster_nodes.enums import AvailableThoughtTypes, AvailableTraversingOut


class TestInstructionNode:
    def test_info(self):
        from quartermaster_nodes.nodes.llm.instruction import InstructionNodeV1

        info = InstructionNodeV1.info()
        assert "instruction" in info.description.lower() or "response" in info.description.lower()
        assert "llm_system_instruction" in info.metadata
        assert "llm_model" in info.metadata

    def test_flow_config(self):
        from quartermaster_nodes.nodes.llm.instruction import InstructionNodeV1

        config = InstructionNodeV1.flow_config()
        assert config.thought_type == AvailableThoughtTypes.NewThought1

    def test_name_and_version(self):
        from quartermaster_nodes.nodes.llm.instruction import InstructionNodeV1

        assert InstructionNodeV1.name() == "InstructionNode"
        assert InstructionNodeV1.version() == "1.0"


class TestDecisionNode:
    def test_info(self):
        from quartermaster_nodes.nodes.llm.decision import Decision1

        info = Decision1.info()
        assert "decide" in info.description.lower() or "path" in info.description.lower()

    def test_flow_config(self):
        from quartermaster_nodes.nodes.llm.decision import Decision1

        config = Decision1.flow_config()
        assert config.traverse_out == AvailableTraversingOut.SpawnPickedNode

    def test_create_decision_tool(self):
        from quartermaster_nodes.nodes.llm.decision import Decision1

        tool = Decision1.create_decision_tool()
        assert tool.name == "pick_path"
        assert len(tool.parameters) == 1
        assert tool.parameters[0].is_required

        tool_dict = tool.to_dict()
        assert tool_dict["type"] == "function"
        assert tool_dict["function"]["name"] == "pick_path"

    def test_stream_disabled(self):
        from quartermaster_nodes.nodes.llm.decision import Decision1

        info = Decision1.info()
        assert info.metadata.get("llm_stream") is False


class TestAgentNode:
    def test_info(self):
        from quartermaster_nodes.nodes.llm.agent import AgentNodeV1

        info = AgentNodeV1.info()
        assert "agent" in info.description.lower() or "autonomous" in info.description.lower()
        assert "max_iterations" in info.metadata

    def test_flow_config(self):
        from quartermaster_nodes.nodes.llm.agent import AgentNodeV1
        from quartermaster_nodes.enums import AvailableErrorHandlingStrategies

        config = AgentNodeV1.flow_config()
        assert config.error_handling_strategy == AvailableErrorHandlingStrategies.Retry


class TestSummarizeNode:
    def test_info(self):
        from quartermaster_nodes.nodes.llm.summarize import Summarize1

        info = Summarize1.info()
        assert "summarize" in info.description.lower() or "conversation" in info.description.lower()


class TestMergeNode:
    def test_info(self):
        from quartermaster_nodes.nodes.llm.merge import Merge1

        info = Merge1.info()
        assert "merge" in info.description.lower()

    def test_flow_config_await_all(self):
        from quartermaster_nodes.nodes.llm.merge import Merge1
        from quartermaster_nodes.enums import AvailableTraversingIn

        config = Merge1.flow_config()
        assert config.traverse_in == AvailableTraversingIn.AwaitAll

    def test_prepare_message(self):
        from quartermaster_nodes.nodes.llm.merge import Merge1

        child1 = MockThought(text="First branch result")
        child2 = MockThought(text="Second branch result")
        thought = MockThought(
            text="",
            _child_thoughts=[child1, child2],
        )

        ctx = MockNodeContext(
            node_metadata={
                "prefix_message": "Combine:",
                "suffix_message": "End.",
            },
            thought=thought,
        )

        msg = Merge1.prepare_message(ctx)
        assert "Combine:" in msg
        assert "First branch result" in msg
        assert "Second branch result" in msg
        assert "End." in msg


class TestInstructionImageVision:
    def test_vision_enabled_by_default(self):
        from quartermaster_nodes.nodes.llm.instruction_image_vision import InstructionImageVision1

        assert InstructionImageVision1.metadata_vision_default_value is True
        info = InstructionImageVision1.info()
        assert info.metadata.get("llm_vision") is True


class TestInstructionParameters:
    def test_stream_disabled(self):
        from quartermaster_nodes.nodes.llm.instruction_parameters import InstructionParameters1

        assert InstructionParameters1.metadata_stream_default_value is False


class TestInstructionProgram:
    def test_has_program_ids_metadata(self):
        from quartermaster_nodes.nodes.llm.instruction_program import InstructionProgram1

        info = InstructionProgram1.info()
        assert "program_version_ids" in info.metadata


class TestInstructionProgramParameters:
    def test_stream_disabled(self):
        from quartermaster_nodes.nodes.llm.instruction_program_parameters import (
            InstructionProgramParameters1,
        )

        assert InstructionProgramParameters1.metadata_stream_default_value is False
