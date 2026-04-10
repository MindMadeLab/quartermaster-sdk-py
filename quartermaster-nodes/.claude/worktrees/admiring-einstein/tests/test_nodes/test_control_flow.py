"""Tests for control flow nodes."""

import pytest
from uuid import uuid4

from tests.conftest import MockNodeContext, MockEdge, MockEdgeQuerySet, MockAssistantNode
from quartermaster_nodes.enums import NEXT_ASSISTANT_NODE_ID, AvailableTraversingOut, AvailableThoughtTypes
from quartermaster_nodes.exceptions import ProcessStopException


class TestStartNode:
    def test_info(self):
        from quartermaster_nodes.nodes.control_flow.start import StartNodeV1

        info = StartNodeV1.info()
        assert info.description
        assert not StartNodeV1.flow_config().accepts_incoming_edges

    def test_think_without_initializer(self):
        from quartermaster_nodes.nodes.control_flow.start import StartNodeV1

        ctx = MockNodeContext()
        StartNodeV1.think(ctx)  # Should not raise

    def test_think_with_initializer(self):
        from quartermaster_nodes.nodes.control_flow.start import StartNodeV1

        initialized = []
        ctx = MockNodeContext(
            node_metadata={"_memory_initializer": lambda c: initialized.append(True)}
        )
        StartNodeV1.think(ctx)
        assert len(initialized) == 1


class TestEndNode:
    def test_info(self):
        from quartermaster_nodes.nodes.control_flow.end import EndNodeV1

        info = EndNodeV1.info()
        assert info.description
        assert EndNodeV1.flow_config().traverse_out == AvailableTraversingOut.SpawnStart
        assert not EndNodeV1.flow_config().accepts_outgoing_edges

    def test_think_does_nothing(self):
        from quartermaster_nodes.nodes.control_flow.end import EndNodeV1

        ctx = MockNodeContext()
        EndNodeV1.think(ctx)  # Should not raise


class TestIfNode:
    def test_picks_true_branch(self):
        from quartermaster_nodes.nodes.control_flow.if_node import IfNode

        true_id = str(uuid4())
        false_id = str(uuid4())

        ctx = MockNodeContext(
            node_metadata={
                "if_expression": "x > 5",
                "_expression_evaluator": None,
            },
            thought=pytest.importorskip("tests.conftest").MockThought(metadata={"x": 10}),
            assistant_node=MockAssistantNode(
                predecessor_edges=MockEdgeQuerySet([
                    MockEdge(tail_id=true_id, main_direction=True, direction_text="true"),
                    MockEdge(tail_id=false_id, main_direction=False, direction_text="false"),
                ])
            ),
        )

        IfNode.think(ctx)
        assert ctx.handle.last_metadata_update[NEXT_ASSISTANT_NODE_ID] == true_id

    def test_picks_false_branch(self):
        from quartermaster_nodes.nodes.control_flow.if_node import IfNode

        true_id = str(uuid4())
        false_id = str(uuid4())

        ctx = MockNodeContext(
            node_metadata={"if_expression": "x > 5"},
            thought=pytest.importorskip("tests.conftest").MockThought(metadata={"x": 2}),
            assistant_node=MockAssistantNode(
                predecessor_edges=MockEdgeQuerySet([
                    MockEdge(tail_id=true_id, main_direction=True, direction_text="true"),
                    MockEdge(tail_id=false_id, main_direction=False, direction_text="false"),
                ])
            ),
        )

        IfNode.think(ctx)
        assert ctx.handle.last_metadata_update[NEXT_ASSISTANT_NODE_ID] == false_id

    def test_raises_without_thought(self):
        from quartermaster_nodes.nodes.control_flow.if_node import IfNode

        ctx = MockNodeContext(thought=None)
        with pytest.raises(ValueError, match="Memory ID cannot be None"):
            IfNode.think(ctx)

    def test_raises_without_edges(self):
        from quartermaster_nodes.nodes.control_flow.if_node import IfNode
        from tests.conftest import MockThought

        ctx = MockNodeContext(
            node_metadata={"if_expression": "True"},
            thought=MockThought(),
            assistant_node=MockAssistantNode(predecessor_edges=MockEdgeQuerySet([])),
        )
        with pytest.raises(ValueError, match="must have one edge"):
            IfNode.think(ctx)


class TestSwitchNode:
    def test_picks_matching_case(self):
        from quartermaster_nodes.nodes.control_flow.switch import SwitchNode1
        from tests.conftest import MockThought

        edge1 = str(uuid4())
        edge2 = str(uuid4())

        ctx = MockNodeContext(
            node_metadata={
                "cases": [
                    {"edge_id": edge1, "expression": "x == 1"},
                    {"edge_id": edge2, "expression": "x == 2"},
                ],
            },
            thought=MockThought(metadata={"x": 2}),
            assistant_node=MockAssistantNode(
                predecessor_edges=MockEdgeQuerySet([
                    MockEdge(tail_id=edge1, main_direction=True, direction_text="case 1"),
                    MockEdge(tail_id=edge2, main_direction=False, direction_text="case 2"),
                ])
            ),
        )

        SwitchNode1.think(ctx)
        assert ctx.handle.last_metadata_update[NEXT_ASSISTANT_NODE_ID] == edge2

    def test_raises_without_default_when_no_match(self):
        from quartermaster_nodes.nodes.control_flow.switch import SwitchNode1
        from tests.conftest import MockThought

        ctx = MockNodeContext(
            node_metadata={"cases": [{"edge_id": "x", "expression": "False"}]},
            thought=MockThought(),
            assistant_node=MockAssistantNode(predecessor_edges=MockEdgeQuerySet([])),
        )

        with pytest.raises(ValueError, match="no case matched"):
            SwitchNode1.think(ctx)


class TestBreakNode:
    def test_info_and_config(self):
        from quartermaster_nodes.nodes.control_flow.break_node import BreakNode1

        info = BreakNode1.info()
        assert "boundary" in info.description.lower()

        config = BreakNode1.flow_config()
        assert config.thought_type == AvailableThoughtTypes.SkipThought1

    def test_think_does_nothing(self):
        from quartermaster_nodes.nodes.control_flow.break_node import BreakNode1

        ctx = MockNodeContext()
        BreakNode1.think(ctx)  # No-op


class TestSubAssistant:
    def test_calls_sub_flow_runner(self):
        from quartermaster_nodes.nodes.control_flow.sub_assistant import SubAssistant1

        called_with = []
        ctx = MockNodeContext(
            node_metadata={
                "sub_assistant_id": "sub-123",
                "_sub_flow_runner": lambda sid, c: called_with.append(sid),
            }
        )
        SubAssistant1.think(ctx)
        assert called_with == ["sub-123"]
