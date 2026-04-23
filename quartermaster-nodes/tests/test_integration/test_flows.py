"""Integration tests for multi-node flows."""

import pytest
from uuid import uuid4

from tests.conftest import (
    MockNodeContext,
    MockThought,
    MockHandle,
    MockEdge,
    MockEdgeQuerySet,
    MockAssistantNode,
)
from quartermaster_nodes.enums import NEXT_ASSISTANT_NODE_ID
from quartermaster_nodes.exceptions import ProcessStopException


class TestSimpleFlow:
    """Test: Start -> Static -> End"""

    def test_start_static_end(self):
        from quartermaster_nodes.nodes.control_flow.start import StartNodeV1
        from quartermaster_nodes.nodes.data.static import StaticNode1
        from quartermaster_nodes.nodes.control_flow.end import EndNodeV1

        # Start node
        ctx_start = MockNodeContext()
        StartNodeV1.think(ctx_start)  # Should complete without error

        # Static node
        ctx_static = MockNodeContext(node_metadata={"static_text": "Hello from static!"})
        StaticNode1.think(ctx_static)
        assert ctx_static.handle.last_text == "Hello from static!"

        # End node
        ctx_end = MockNodeContext()
        EndNodeV1.think(ctx_end)  # Should complete without error


class TestDecisionFlow:
    """Test: Start -> [decision logic] -> Path A or Path B"""

    def test_if_node_branching(self):
        from quartermaster_nodes.nodes.control_flow.if_node import IfNode
        from quartermaster_nodes.nodes.data.static import StaticNode1

        path_a = str(uuid4())
        path_b = str(uuid4())

        # If node with condition score > 80
        ctx_if = MockNodeContext(
            node_metadata={"if_expression": "score > 80"},
            thought=MockThought(metadata={"score": 95}),
            assistant_node=MockAssistantNode(
                predecessor_edges=MockEdgeQuerySet(
                    [
                        MockEdge(tail_id=path_a, main_direction=True, direction_text="pass"),
                        MockEdge(tail_id=path_b, main_direction=False, direction_text="fail"),
                    ]
                )
            ),
        )
        IfNode.think(ctx_if)

        picked = ctx_if.handle.last_metadata_update[NEXT_ASSISTANT_NODE_ID]
        assert picked == path_a  # Score 95 > 80, should go to path A

        # Execute the selected path
        ctx_static = MockNodeContext(node_metadata={"static_text": "You passed!"})
        StaticNode1.think(ctx_static)
        assert ctx_static.handle.last_text == "You passed!"


class TestVariableFlow:
    """Test: Var -> Text -> TextToVariable"""

    def test_variable_to_text_pipeline(self):
        from quartermaster_nodes.nodes.data.var import VarNode
        from quartermaster_nodes.nodes.data.text import TextNode
        from quartermaster_nodes.nodes.data.text_to_variable import TextToVariableNode

        # Step 1: Compute a variable
        ctx_var = MockNodeContext(
            node_metadata={"name": "greeting", "expression": "'Hello, ' + user_name"},
            thought=MockThought(metadata={"user_name": "Alice"}),
        )
        VarNode.think(ctx_var)
        assert ctx_var.handle.last_metadata_update == {"greeting": "Hello, Alice"}

        # Step 2: Render a template using the computed variable
        # Simulate passing the variable forward
        ctx_text = MockNodeContext(
            node_metadata={"text": "Message: {{ greeting }}"},
            thought=MockThought(metadata={"greeting": "Hello, Alice"}),
        )
        TextNode.think(ctx_text)
        assert ctx_text.handle.last_text == "Message: Hello, Alice"

        # Step 3: Capture the text as a variable
        ctx_ttv = MockNodeContext(
            node_metadata={"variable_name": "final_output"},
            thought=MockThought(text="Message: Hello, Alice"),
        )
        TextToVariableNode.think(ctx_ttv)
        assert ctx_ttv.handle.last_metadata_update == {"final_output": "Message: Hello, Alice"}


class TestUserInteractionFlow:
    """Test flow that pauses for user input."""

    def test_pause_and_resume(self):
        from quartermaster_nodes.nodes.data.static import StaticNode1
        from quartermaster_nodes.nodes.user_interaction.user import UserNode1

        # Static node runs normally
        ctx_static = MockNodeContext(node_metadata={"static_text": "Please provide your name:"})
        StaticNode1.think(ctx_static)
        assert ctx_static.handle.last_text == "Please provide your name:"

        # User node pauses the flow
        ctx_user = MockNodeContext()
        with pytest.raises(ProcessStopException):
            UserNode1.think(ctx_user)


class TestSwitchFlow:
    """Test multi-way branching with Switch node."""

    def test_switch_routing(self):
        from quartermaster_nodes.nodes.control_flow.switch import SwitchNode1

        edge_low = str(uuid4())
        edge_med = str(uuid4())
        edge_high = str(uuid4())

        cases = [
            {"edge_id": edge_low, "expression": "priority == 'low'"},
            {"edge_id": edge_med, "expression": "priority == 'medium'"},
            {"edge_id": edge_high, "expression": "priority == 'high'"},
        ]

        ctx = MockNodeContext(
            node_metadata={"cases": cases},
            thought=MockThought(metadata={"priority": "medium"}),
            assistant_node=MockAssistantNode(
                predecessor_edges=MockEdgeQuerySet(
                    [
                        MockEdge(tail_id=edge_low, main_direction=False, direction_text="low"),
                        MockEdge(tail_id=edge_med, main_direction=False, direction_text="medium"),
                        MockEdge(tail_id=edge_high, main_direction=True, direction_text="high"),
                    ]
                )
            ),
        )

        SwitchNode1.think(ctx)
        assert ctx.handle.last_metadata_update[NEXT_ASSISTANT_NODE_ID] == edge_med


class TestRegistryDiscovery:
    """Test that all nodes can be discovered and registered."""

    def test_discover_all_nodes(self):
        from quartermaster_nodes.registry import NodeCatalog

        registry = NodeCatalog()
        count = registry.discover("quartermaster_nodes.nodes")
        assert count >= 30  # We have ~38 node types

        # Verify some key nodes are registered
        catalog = registry.catalog_json()
        names = {entry["name"] for entry in catalog}
        assert "InstructionNode" in names
        assert "Decision1" in names
        assert "StartNode" in names
        assert "EndNode" in names
        assert "StaticAssistant" in names

    def test_all_nodes_have_valid_config(self):
        from quartermaster_nodes.registry import NodeCatalog

        registry = NodeCatalog()
        registry.discover("quartermaster_nodes.nodes")

        for entry in registry.catalog_json():
            node_cls = registry.get(entry["name"], entry["version"])
            config = node_cls.flow_config()
            config.validate()  # Should not raise

    def test_all_nodes_have_info(self):
        from quartermaster_nodes.registry import NodeCatalog

        registry = NodeCatalog()
        registry.discover("quartermaster_nodes.nodes")

        for entry in registry.catalog_json():
            node_cls = registry.get(entry["name"], entry["version"])
            info = node_cls.info()
            assert info.version
            assert info.description
