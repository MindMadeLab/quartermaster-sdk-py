"""Tests for InstructionNodeV1 — the core LLM node."""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from tests.conftest import MockNodeContext, MockThought, MockHandle
from quartermaster_nodes.nodes.llm.instruction import InstructionNodeV1
from quartermaster_nodes.enums import (
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)


class TestInstructionNodeInfo:
    """Tests for InstructionNodeV1.info() class metadata."""

    def test_name(self):
        assert InstructionNodeV1.name() == "InstructionNode"

    def test_version(self):
        assert InstructionNodeV1.version() == "1.0"

    def test_not_deprecated(self):
        assert InstructionNodeV1.deprecated() is False

    def test_info_has_description(self):
        info = InstructionNodeV1.info()
        assert info.description
        assert isinstance(info.description, str)

    def test_info_has_instructions(self):
        info = InstructionNodeV1.info()
        assert info.instructions
        assert isinstance(info.instructions, str)

    def test_info_version_matches(self):
        info = InstructionNodeV1.info()
        assert info.version == "1.0"

    def test_info_metadata_keys(self):
        info = InstructionNodeV1.info()
        expected_keys = {
            "llm_system_instruction",
            "llm_model",
            "llm_provider",
            "llm_temperature",
            "llm_max_input_tokens",
            "llm_max_output_tokens",
            "llm_max_messages",
            "llm_stream",
            "llm_thinking_level",
        }
        assert expected_keys == set(info.metadata.keys())


class TestInstructionNodeFlowConfig:
    """Tests for InstructionNodeV1.flow_config()."""

    def test_traverse_in(self):
        config = InstructionNodeV1.flow_config()
        assert config.traverse_in == AvailableTraversingIn.AwaitFirst

    def test_traverse_out(self):
        config = InstructionNodeV1.flow_config()
        assert config.traverse_out == AvailableTraversingOut.SpawnAll

    def test_thought_type(self):
        config = InstructionNodeV1.flow_config()
        assert config.thought_type == AvailableThoughtTypes.NewThought1

    def test_message_type(self):
        config = InstructionNodeV1.flow_config()
        assert config.message_type == AvailableMessageTypes.Assistant

    def test_accepts_edges(self):
        config = InstructionNodeV1.flow_config()
        assert config.accepts_incoming_edges is True
        assert config.accepts_outgoing_edges is True

    def test_available_thought_types(self):
        config = InstructionNodeV1.flow_config()
        assert AvailableThoughtTypes.EditSameOrAddNew1 in config.available_thought_types
        assert AvailableThoughtTypes.UsePreviousThought1 in config.available_thought_types
        assert AvailableThoughtTypes.NewCollapsedThought1 in config.available_thought_types

    def test_available_message_types(self):
        config = InstructionNodeV1.flow_config()
        assert AvailableMessageTypes.Automatic in config.available_message_types
        assert AvailableMessageTypes.User in config.available_message_types

    def test_config_validation_passes(self):
        config = InstructionNodeV1.flow_config()
        config.validate()  # Should not raise

    def test_config_serialization_roundtrip(self):
        config = InstructionNodeV1.flow_config()
        d = config.asdict()
        assert d["traverse_in"] == "AwaitFirst"
        assert d["traverse_out"] == "SpawnAll"
        assert d["thought_type"] == "NewThought1"


class TestInstructionNodeLLMConfig:
    """Tests for LLM configuration from metadata."""

    def test_default_llm_config(self):
        ctx = MockNodeContext()
        llm_config = InstructionNodeV1.llm_config(ctx)
        assert llm_config.model == "gpt-4o-mini"
        assert llm_config.provider == "openai"
        assert llm_config.temperature == 0.5
        assert llm_config.stream is True

    def test_custom_llm_config(self):
        ctx = MockNodeContext(node_metadata={
            "llm_model": "claude-3-opus",
            "llm_provider": "anthropic",
            "llm_temperature": 0.9,
            "llm_stream": False,
            "llm_max_input_tokens": 8192,
            "llm_max_output_tokens": 4096,
        })
        llm_config = InstructionNodeV1.llm_config(ctx)
        assert llm_config.model == "claude-3-opus"
        assert llm_config.provider == "anthropic"
        assert llm_config.temperature == 0.9
        assert llm_config.stream is False
        assert llm_config.max_input_tokens == 8192
        assert llm_config.max_output_tokens == 4096

    def test_temperature_string_conversion(self):
        ctx = MockNodeContext(node_metadata={"llm_temperature": "0.7"})
        llm_config = InstructionNodeV1.llm_config(ctx)
        assert llm_config.temperature == 0.7
        assert isinstance(llm_config.temperature, float)

    def test_thinking_level_off(self):
        ctx = MockNodeContext(node_metadata={"llm_thinking_level": "off"})
        llm_config = InstructionNodeV1.llm_config(ctx)
        assert llm_config.thinking_enabled is False
        assert llm_config.thinking_budget is None

    def test_thinking_level_high(self):
        ctx = MockNodeContext(node_metadata={"llm_thinking_level": "high"})
        llm_config = InstructionNodeV1.llm_config(ctx)
        assert llm_config.thinking_enabled is True
        assert llm_config.thinking_budget == 16384

    def test_system_instruction_default(self):
        ctx = MockNodeContext()
        llm_config = InstructionNodeV1.llm_config(ctx)
        assert "helpful" in llm_config.system_message.lower()

    def test_custom_system_instruction(self):
        ctx = MockNodeContext(node_metadata={
            "llm_system_instruction": "You are a pirate."
        })
        llm_config = InstructionNodeV1.llm_config(ctx)
        assert llm_config.system_message == "You are a pirate."


class TestInstructionNodeThink:
    """Tests for InstructionNodeV1.think() execution."""

    @patch("quartermaster_nodes.nodes.llm.instruction.Chain")
    def test_think_builds_chain(self, mock_chain_cls):
        mock_chain = MagicMock()
        mock_chain.add_handler.return_value = mock_chain
        mock_chain_cls.return_value = mock_chain

        ctx = MockNodeContext(
            node_metadata={
                "_transformer": MagicMock(),
                "_client": MagicMock(),
            },
            thought_id=uuid4(),
        )

        InstructionNodeV1.think(ctx)

        mock_chain_cls.assert_called_once()
        assert mock_chain.add_handler.call_count == 6
        mock_chain.run.assert_called_once()

    @patch("quartermaster_nodes.nodes.llm.instruction.Chain")
    def test_think_passes_correct_initial_data(self, mock_chain_cls):
        mock_chain = MagicMock()
        mock_chain.add_handler.return_value = mock_chain
        mock_chain_cls.return_value = mock_chain

        thought_id = uuid4()
        flow_node_id = uuid4()
        ctx = MockNodeContext(
            node_metadata={"_transformer": None, "_client": None},
            thought_id=thought_id,
            flow_node_id=flow_node_id,
        )

        InstructionNodeV1.think(ctx)

        call_args = mock_chain.run.call_args[0][0]
        assert call_args["memory_id"] == thought_id
        assert call_args["flow_node_id"] == flow_node_id
        assert call_args["ctx"] is ctx

    @patch("quartermaster_nodes.nodes.llm.instruction.Chain")
    def test_think_with_no_transformer_or_client(self, mock_chain_cls):
        mock_chain = MagicMock()
        mock_chain.add_handler.return_value = mock_chain
        mock_chain_cls.return_value = mock_chain

        ctx = MockNodeContext()

        InstructionNodeV1.think(ctx)
        mock_chain.run.assert_called_once()


class TestInstructionNodeContextManagerConfig:
    """Tests for context manager configuration."""

    def test_default_context_config(self):
        ctx = MockNodeContext()
        llm_config = InstructionNodeV1.llm_config(ctx)
        cm_config = InstructionNodeV1.context_manager_config(ctx, llm_config)

        assert cm_config.tool_clearing_trigger is None
        assert cm_config.tool_clearing_keep is None
        assert cm_config.exclude_tools == []
        assert cm_config.max_tool_result_tokens is None

    def test_custom_context_config(self):
        ctx = MockNodeContext(node_metadata={
            "context_tool_clearing_trigger": 10,
            "context_tool_clearing_keep": 3,
            "context_exclude_tools": ["tool_a"],
            "context_max_tool_result_tokens": 1000,
        })
        llm_config = InstructionNodeV1.llm_config(ctx)
        cm_config = InstructionNodeV1.context_manager_config(ctx, llm_config)

        assert cm_config.tool_clearing_trigger == 10
        assert cm_config.tool_clearing_keep == 3
        assert cm_config.exclude_tools == ["tool_a"]
        assert cm_config.max_tool_result_tokens == 1000
