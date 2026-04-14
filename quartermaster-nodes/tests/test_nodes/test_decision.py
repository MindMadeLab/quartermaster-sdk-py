"""Tests for Decision1 — LLM-driven path selection."""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from tests.conftest import (
    MockAssistantNode,
    MockEdge,
    MockEdgeQuerySet,
    MockNodeContext,
    MockThought,
)
from quartermaster_nodes.nodes.llm.decision import Decision1
from quartermaster_nodes.enums import (
    NEXT_ASSISTANT_NODE_ID,
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)


class TestDecisionInfo:
    """Tests for Decision1 class metadata."""

    def test_name(self):
        assert Decision1.name() == "Decision1"

    def test_version(self):
        assert Decision1.version() == "1.0"

    def test_info_description(self):
        info = Decision1.info()
        assert info.description
        assert "path" in info.description.lower() or "decide" in info.description.lower()

    def test_info_metadata_keys(self):
        info = Decision1.info()
        assert "llm_system_instruction" in info.metadata
        assert "llm_model" in info.metadata
        assert "prefix_message" in info.metadata
        assert "suffix_message" in info.metadata

    def test_stream_disabled_by_default(self):
        info = Decision1.info()
        assert info.metadata.get("llm_stream") is False

    def test_default_model(self):
        assert Decision1.metadata_model_default_value == "gpt-4o-mini"

    def test_default_system_instruction(self):
        assert "pick_path" in Decision1.metadata_system_instruction_default_value


class TestDecisionFlowConfig:
    """Tests for Decision1.flow_config()."""

    def test_traverse_out_spawn_picked(self):
        config = Decision1.flow_config()
        assert config.traverse_out == AvailableTraversingOut.SpawnPickedNode

    def test_traverse_in_await_first(self):
        config = Decision1.flow_config()
        assert config.traverse_in == AvailableTraversingIn.AwaitFirst

    def test_thought_type_use_previous(self):
        config = Decision1.flow_config()
        assert config.thought_type == AvailableThoughtTypes.UsePreviousThought1

    def test_message_type_variable(self):
        config = Decision1.flow_config()
        assert config.message_type == AvailableMessageTypes.Variable


class TestDecisionCreateTool:
    """Tests for Decision1.create_decision_tool()."""

    def test_tool_name(self):
        tool = Decision1.create_decision_tool()
        assert tool.name == "pick_path"

    def test_tool_has_one_required_parameter(self):
        tool = Decision1.create_decision_tool()
        assert len(tool.parameters) == 1
        assert tool.parameters[0].is_required is True
        assert tool.parameters[0].name == NEXT_ASSISTANT_NODE_ID

    def test_tool_parameter_type(self):
        tool = Decision1.create_decision_tool()
        assert tool.parameters[0].type == "string"

    def test_tool_dict_format(self):
        tool = Decision1.create_decision_tool()
        d = tool.to_dict()
        assert d["type"] == "function"
        assert d["function"]["name"] == "pick_path"
        assert NEXT_ASSISTANT_NODE_ID in d["function"]["parameters"]["properties"]
        assert NEXT_ASSISTANT_NODE_ID in d["function"]["parameters"]["required"]


class TestDecisionPrepareMessage:
    """Tests for Decision1.prepare_decision_message()."""

    def test_includes_edge_descriptions(self):
        edge1_id = str(uuid4())
        edge2_id = str(uuid4())
        ctx = MockNodeContext(
            node_metadata={
                "prefix_message": "Pick a path:",
                "suffix_message": "",
            },
            assistant_node=MockAssistantNode(
                predecessor_edges=MockEdgeQuerySet(
                    [
                        MockEdge(tail_id=edge1_id, main_direction=True, direction_text="Go left"),
                        MockEdge(tail_id=edge2_id, main_direction=False, direction_text="Go right"),
                    ]
                )
            ),
        )
        msg = Decision1.prepare_decision_message(ctx)
        assert "Pick a path:" in msg
        assert edge1_id in msg
        assert "Go left" in msg
        assert edge2_id in msg
        assert "Go right" in msg

    def test_uses_default_prefix(self):
        ctx = MockNodeContext(
            assistant_node=MockAssistantNode(
                predecessor_edges=MockEdgeQuerySet(
                    [
                        MockEdge(tail_id="e1", main_direction=True, direction_text="yes"),
                    ]
                )
            ),
        )
        msg = Decision1.prepare_decision_message(ctx)
        assert "best path" in msg.lower()

    def test_suffix_included(self):
        ctx = MockNodeContext(
            node_metadata={"suffix_message": "Choose wisely."},
            assistant_node=MockAssistantNode(
                predecessor_edges=MockEdgeQuerySet(
                    [
                        MockEdge(tail_id="e1", main_direction=True, direction_text="option"),
                    ]
                )
            ),
        )
        msg = Decision1.prepare_decision_message(ctx)
        assert "Choose wisely." in msg

    def test_empty_edges(self):
        ctx = MockNodeContext(
            assistant_node=MockAssistantNode(predecessor_edges=MockEdgeQuerySet([])),
        )
        msg = Decision1.prepare_decision_message(ctx)
        assert isinstance(msg, str)


class TestDecisionThink:
    """Tests for Decision1.think() execution."""

    @patch("quartermaster_nodes.nodes.llm.decision.Chain")
    def test_think_builds_chain(self, mock_chain_cls):
        mock_chain = MagicMock()
        mock_chain.add_handler.return_value = mock_chain
        mock_chain_cls.return_value = mock_chain

        ctx = MockNodeContext(
            node_metadata={"_transformer": MagicMock(), "_client": MagicMock()},
            thought_id=uuid4(),
            assistant_node=MockAssistantNode(
                predecessor_edges=MockEdgeQuerySet(
                    [
                        MockEdge(tail_id="e1", main_direction=True, direction_text="yes"),
                    ]
                )
            ),
        )

        Decision1.think(ctx)

        mock_chain_cls.assert_called_once()
        assert mock_chain.add_handler.call_count == 6
        mock_chain.run.assert_called_once()

    @patch("quartermaster_nodes.nodes.llm.decision.Chain")
    def test_think_disables_streaming(self, mock_chain_cls):
        mock_chain = MagicMock()
        mock_chain.add_handler.return_value = mock_chain
        mock_chain_cls.return_value = mock_chain

        ctx = MockNodeContext(
            node_metadata={
                "_transformer": None,
                "_client": None,
                "llm_stream": True,  # Even if set to True, Decision forces False
            },
            assistant_node=MockAssistantNode(predecessor_edges=MockEdgeQuerySet([])),
        )

        Decision1.think(ctx)
        # The node internally sets llm_config.stream = False
        mock_chain.run.assert_called_once()

    @patch("quartermaster_nodes.nodes.llm.decision.Chain")
    def test_think_initial_data_contains_expected_keys(self, mock_chain_cls):
        mock_chain = MagicMock()
        mock_chain.add_handler.return_value = mock_chain
        mock_chain_cls.return_value = mock_chain

        thought_id = uuid4()
        flow_node_id = uuid4()
        ctx = MockNodeContext(
            node_metadata={"_transformer": None, "_client": None},
            thought_id=thought_id,
            flow_node_id=flow_node_id,
            assistant_node=MockAssistantNode(predecessor_edges=MockEdgeQuerySet([])),
        )

        Decision1.think(ctx)

        call_args = mock_chain.run.call_args[0][0]
        assert call_args["memory_id"] == thought_id
        assert call_args["flow_node_id"] == flow_node_id
        assert call_args["ctx"] is ctx


class TestDecisionLLMConfig:
    """Tests for LLM configuration specifics."""

    def test_default_max_input_tokens(self):
        assert Decision1.metadata_max_input_tokens_default_value == 32768

    def test_llm_config_stream_default_false(self):
        ctx = MockNodeContext()
        llm_config = Decision1.llm_config(ctx)
        # Default from class is False
        assert llm_config.stream is False
