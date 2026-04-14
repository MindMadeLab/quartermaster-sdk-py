"""Tests for IfNode (If1) — binary conditional branching."""

from uuid import uuid4

import pytest

from tests.conftest import (
    MockAssistantNode,
    MockEdge,
    MockEdgeQuerySet,
    MockExpressionEvaluator,
    MockNodeContext,
    MockThought,
)
from quartermaster_nodes.nodes.control_flow.if_node import IfNode
from quartermaster_nodes.enums import (
    NEXT_ASSISTANT_NODE_ID,
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)
from quartermaster_nodes.protocols import ExpressionResult


class TestIfNodeInfo:
    """Tests for IfNode class metadata."""

    def test_name(self):
        assert IfNode.name() == "IfNode1"

    def test_version(self):
        assert IfNode.version() == "1.0"

    def test_info_description(self):
        info = IfNode.info()
        assert "expression" in info.description.lower() or "branch" in info.description.lower()

    def test_info_metadata_keys(self):
        info = IfNode.info()
        assert "if_expression" in info.metadata

    def test_default_expression_empty(self):
        assert IfNode.metadata_if_expression_default == ""


class TestIfNodeFlowConfig:
    """Tests for IfNode.flow_config()."""

    def test_traverse_out_spawn_picked(self):
        config = IfNode.flow_config()
        assert config.traverse_out == AvailableTraversingOut.SpawnPickedNode

    def test_traverse_in(self):
        config = IfNode.flow_config()
        assert config.traverse_in == AvailableTraversingIn.AwaitFirst

    def test_thought_type_use_previous(self):
        config = IfNode.flow_config()
        assert config.thought_type == AvailableThoughtTypes.UsePreviousThought1

    def test_message_type_variable(self):
        config = IfNode.flow_config()
        assert config.message_type == AvailableMessageTypes.Variable

    def test_available_thought_types(self):
        config = IfNode.flow_config()
        assert AvailableThoughtTypes.EditSameOrAddNew1 in config.available_thought_types
        assert AvailableThoughtTypes.NewHiddenThought1 in config.available_thought_types

    def test_config_validation_passes(self):
        config = IfNode.flow_config()
        config.validate()


def _make_if_context(
    expression: str,
    metadata: dict | None = None,
    evaluator=None,
    true_id: str | None = None,
    false_id: str | None = None,
) -> MockNodeContext:
    """Helper to build a context for IfNode tests."""
    true_id = true_id or str(uuid4())
    false_id = false_id or str(uuid4())

    node_metadata = {"if_expression": expression}
    if evaluator is not None:
        node_metadata["_expression_evaluator"] = evaluator
    node_metadata.update(metadata or {})

    return MockNodeContext(
        node_metadata=node_metadata,
        thought=MockThought(metadata=metadata or {}),
        assistant_node=MockAssistantNode(
            predecessor_edges=MockEdgeQuerySet(
                [
                    MockEdge(tail_id=true_id, main_direction=True, direction_text="true"),
                    MockEdge(tail_id=false_id, main_direction=False, direction_text="false"),
                ]
            )
        ),
    )


class TestIfNodeThink:
    """Tests for IfNode.think() execution."""

    def test_picks_true_branch(self):
        true_id = str(uuid4())
        false_id = str(uuid4())
        ctx = MockNodeContext(
            node_metadata={"if_expression": "x > 5"},
            thought=MockThought(metadata={"x": 10}),
            assistant_node=MockAssistantNode(
                predecessor_edges=MockEdgeQuerySet(
                    [
                        MockEdge(tail_id=true_id, main_direction=True, direction_text="true"),
                        MockEdge(tail_id=false_id, main_direction=False, direction_text="false"),
                    ]
                )
            ),
        )

        IfNode.think(ctx)
        assert ctx.handle.last_metadata_update[NEXT_ASSISTANT_NODE_ID] == true_id

    def test_picks_false_branch(self):
        true_id = str(uuid4())
        false_id = str(uuid4())
        ctx = MockNodeContext(
            node_metadata={"if_expression": "x > 5"},
            thought=MockThought(metadata={"x": 2}),
            assistant_node=MockAssistantNode(
                predecessor_edges=MockEdgeQuerySet(
                    [
                        MockEdge(tail_id=true_id, main_direction=True, direction_text="true"),
                        MockEdge(tail_id=false_id, main_direction=False, direction_text="false"),
                    ]
                )
            ),
        )

        IfNode.think(ctx)
        assert ctx.handle.last_metadata_update[NEXT_ASSISTANT_NODE_ID] == false_id

    def test_equality_expression(self):
        true_id = "t"
        false_id = "f"
        ctx = MockNodeContext(
            node_metadata={"if_expression": "status == 'active'"},
            thought=MockThought(metadata={"status": "active"}),
            assistant_node=MockAssistantNode(
                predecessor_edges=MockEdgeQuerySet(
                    [
                        MockEdge(tail_id=true_id, main_direction=True, direction_text="true"),
                        MockEdge(tail_id=false_id, main_direction=False, direction_text="false"),
                    ]
                )
            ),
        )

        IfNode.think(ctx)
        assert ctx.handle.last_metadata_update[NEXT_ASSISTANT_NODE_ID] == true_id

    def test_boolean_variable(self):
        true_id = "t"
        false_id = "f"
        ctx = MockNodeContext(
            node_metadata={"if_expression": "flag"},
            thought=MockThought(metadata={"flag": False}),
            assistant_node=MockAssistantNode(
                predecessor_edges=MockEdgeQuerySet(
                    [
                        MockEdge(tail_id=true_id, main_direction=True, direction_text="true"),
                        MockEdge(tail_id=false_id, main_direction=False, direction_text="false"),
                    ]
                )
            ),
        )

        IfNode.think(ctx)
        assert ctx.handle.last_metadata_update[NEXT_ASSISTANT_NODE_ID] == false_id

    def test_with_expression_evaluator(self):
        true_id = "t"
        false_id = "f"
        evaluator = MockExpressionEvaluator(results={"custom_expr": True})
        ctx = MockNodeContext(
            node_metadata={
                "if_expression": "custom_expr",
                "_expression_evaluator": evaluator,
            },
            thought=MockThought(metadata={}),
            assistant_node=MockAssistantNode(
                predecessor_edges=MockEdgeQuerySet(
                    [
                        MockEdge(tail_id=true_id, main_direction=True, direction_text="true"),
                        MockEdge(tail_id=false_id, main_direction=False, direction_text="false"),
                    ]
                )
            ),
        )

        IfNode.think(ctx)
        assert ctx.handle.last_metadata_update[NEXT_ASSISTANT_NODE_ID] == true_id

    def test_evaluator_returns_false(self):
        true_id = "t"
        false_id = "f"
        evaluator = MockExpressionEvaluator(results={"check": False})
        ctx = MockNodeContext(
            node_metadata={
                "if_expression": "check",
                "_expression_evaluator": evaluator,
            },
            thought=MockThought(metadata={}),
            assistant_node=MockAssistantNode(
                predecessor_edges=MockEdgeQuerySet(
                    [
                        MockEdge(tail_id=true_id, main_direction=True, direction_text="true"),
                        MockEdge(tail_id=false_id, main_direction=False, direction_text="false"),
                    ]
                )
            ),
        )

        IfNode.think(ctx)
        assert ctx.handle.last_metadata_update[NEXT_ASSISTANT_NODE_ID] == false_id


class TestIfNodeErrors:
    """Tests for IfNode error handling."""

    def test_raises_without_thought(self):
        ctx = MockNodeContext(thought=None)
        with pytest.raises(ValueError, match="Memory ID cannot be None"):
            IfNode.think(ctx)

    def test_raises_without_edges(self):
        ctx = MockNodeContext(
            node_metadata={"if_expression": "True"},
            thought=MockThought(),
            assistant_node=MockAssistantNode(predecessor_edges=MockEdgeQuerySet([])),
        )
        with pytest.raises(ValueError, match="must have one edge"):
            IfNode.think(ctx)

    def test_raises_with_only_true_edge(self):
        ctx = MockNodeContext(
            node_metadata={"if_expression": "True"},
            thought=MockThought(),
            assistant_node=MockAssistantNode(
                predecessor_edges=MockEdgeQuerySet(
                    [
                        MockEdge(tail_id="t", main_direction=True, direction_text="true"),
                    ]
                )
            ),
        )
        with pytest.raises(ValueError, match="must have one edge"):
            IfNode.think(ctx)

    def test_raises_with_only_false_edge(self):
        ctx = MockNodeContext(
            node_metadata={"if_expression": "True"},
            thought=MockThought(),
            assistant_node=MockAssistantNode(
                predecessor_edges=MockEdgeQuerySet(
                    [
                        MockEdge(tail_id="f", main_direction=False, direction_text="false"),
                    ]
                )
            ),
        )
        with pytest.raises(ValueError, match="must have one edge"):
            IfNode.think(ctx)

    def test_raises_on_handle_none(self):
        ctx = MockNodeContext(
            node_metadata={"if_expression": "True"},
            thought=MockThought(),
            handle=None,
            assistant_node=MockAssistantNode(
                predecessor_edges=MockEdgeQuerySet(
                    [
                        MockEdge(tail_id="t", main_direction=True, direction_text="true"),
                        MockEdge(tail_id="f", main_direction=False, direction_text="false"),
                    ]
                )
            ),
        )
        with pytest.raises(AssertionError, match="handle not set"):
            IfNode.think(ctx)
