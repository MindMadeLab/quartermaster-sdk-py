"""Tests for Merge1 — combine parallel conversation branches."""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from tests.conftest import MockNodeContext, MockThought
from quartermaster_nodes.nodes.llm.merge import Merge1
from quartermaster_nodes.enums import (
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)


class TestMergeInfo:
    """Tests for Merge1 class metadata."""

    def test_name(self):
        assert Merge1.name() == "Merge1"

    def test_version(self):
        assert Merge1.version() == "1.0"

    def test_info_description(self):
        info = Merge1.info()
        assert "merge" in info.description.lower()

    def test_info_metadata_keys(self):
        info = Merge1.info()
        assert "prefix_message" in info.metadata
        assert "suffix_message" in info.metadata
        assert "llm_system_instruction" in info.metadata
        assert "llm_model" in info.metadata
        assert "llm_provider" in info.metadata

    def test_default_model(self):
        assert Merge1.metadata_model_default_value == "gpt-4o-mini"

    def test_default_prefix(self):
        assert "compress" in Merge1.metadata_prefix_message_default_value.lower()


class TestMergeFlowConfig:
    """Tests for Merge1.flow_config()."""

    def test_await_all(self):
        config = Merge1.flow_config()
        assert config.traverse_in == AvailableTraversingIn.AwaitAll

    def test_spawn_all(self):
        config = Merge1.flow_config()
        assert config.traverse_out == AvailableTraversingOut.SpawnAll

    def test_new_thought(self):
        config = Merge1.flow_config()
        assert config.thought_type == AvailableThoughtTypes.NewThought1

    def test_assistant_message_type(self):
        config = Merge1.flow_config()
        assert config.message_type == AvailableMessageTypes.Assistant

    def test_available_thought_types(self):
        config = Merge1.flow_config()
        assert AvailableThoughtTypes.NewHiddenThought1 in config.available_thought_types
        assert AvailableThoughtTypes.NewCollapsedThought1 in config.available_thought_types

    def test_config_validation_passes(self):
        config = Merge1.flow_config()
        config.validate()


class TestMergePrepareMessage:
    """Tests for Merge1.prepare_message()."""

    def test_combines_child_thoughts(self):
        child1 = MockThought(text="Branch A result")
        child2 = MockThought(text="Branch B result")
        thought = MockThought(text="", _child_thoughts=[child1, child2])

        ctx = MockNodeContext(
            node_metadata={
                "prefix_message": "Combine:",
                "suffix_message": "Done.",
            },
            thought=thought,
        )

        msg = Merge1.prepare_message(ctx)
        assert "Combine:" in msg
        assert "Branch A result" in msg
        assert "Branch B result" in msg
        assert "Done." in msg

    def test_uses_default_prefix_and_suffix(self):
        child = MockThought(text="Some text")
        thought = MockThought(text="", _child_thoughts=[child])

        ctx = MockNodeContext(thought=thought)

        msg = Merge1.prepare_message(ctx)
        assert "Compress" in msg or "compress" in msg
        assert "Some text" in msg

    def test_empty_child_thoughts(self):
        thought = MockThought(text="", _child_thoughts=[])
        ctx = MockNodeContext(thought=thought)

        msg = Merge1.prepare_message(ctx)
        assert isinstance(msg, str)

    def test_single_child_thought(self):
        child = MockThought(text="Only one branch")
        thought = MockThought(text="", _child_thoughts=[child])

        ctx = MockNodeContext(
            node_metadata={"prefix_message": "Merge:", "suffix_message": ""},
            thought=thought,
        )

        msg = Merge1.prepare_message(ctx)
        assert "Only one branch" in msg

    def test_many_child_thoughts(self):
        children = [MockThought(text=f"Branch {i}") for i in range(5)]
        thought = MockThought(text="", _child_thoughts=children)

        ctx = MockNodeContext(
            node_metadata={"prefix_message": "P", "suffix_message": "S"},
            thought=thought,
        )

        msg = Merge1.prepare_message(ctx)
        for i in range(5):
            assert f"Branch {i}" in msg


class TestMergeThink:
    """Tests for Merge1.think() execution."""

    def test_raises_without_thought(self):
        ctx = MockNodeContext(thought=None)
        with pytest.raises(ValueError, match="Memory ID is required"):
            Merge1.think(ctx)

    @patch("quartermaster_nodes.nodes.llm.merge.Chain")
    def test_think_builds_chain(self, mock_chain_cls):
        mock_chain = MagicMock()
        mock_chain.add_handler.return_value = mock_chain
        mock_chain_cls.return_value = mock_chain

        thought = MockThought(text="", _child_thoughts=[MockThought(text="child")])
        ctx = MockNodeContext(
            node_metadata={"_transformer": MagicMock(), "_client": MagicMock()},
            thought=thought,
            thought_id=uuid4(),
        )

        Merge1.think(ctx)

        mock_chain_cls.assert_called_once()
        assert mock_chain.add_handler.call_count == 5
        mock_chain.run.assert_called_once()

    @patch("quartermaster_nodes.nodes.llm.merge.Chain")
    def test_think_initial_data_keys(self, mock_chain_cls):
        mock_chain = MagicMock()
        mock_chain.add_handler.return_value = mock_chain
        mock_chain_cls.return_value = mock_chain

        thought_id = uuid4()
        flow_node_id = uuid4()
        thought = MockThought(text="", _child_thoughts=[])
        ctx = MockNodeContext(
            node_metadata={"_transformer": None, "_client": None},
            thought=thought,
            thought_id=thought_id,
            flow_node_id=flow_node_id,
        )

        Merge1.think(ctx)

        call_args = mock_chain.run.call_args[0][0]
        assert call_args["memory_id"] == thought_id
        assert call_args["to_memory_id"] == thought_id
        assert call_args["flow_node_id"] == flow_node_id
        assert call_args["ctx"] is ctx


class TestMergeLLMConfig:
    """Tests for Merge LLM configuration."""

    def test_default_system_instruction(self):
        assert "combine" in Merge1.metadata_system_instruction_default_value.lower()

    def test_custom_system_instruction(self):
        ctx = MockNodeContext(node_metadata={
            "llm_system_instruction": "You are a merger."
        })
        llm_config = Merge1.llm_config(ctx)
        assert llm_config.system_message == "You are a merger."
