"""Tests for data manipulation nodes."""

import pytest

from tests.conftest import MockNodeContext, MockThought


class TestStaticNode:
    def test_outputs_default_text(self):
        from quartermaster_nodes.nodes.data.static import StaticNode1

        ctx = MockNodeContext()
        StaticNode1.think(ctx)
        assert ctx.handle.last_text == "This is a default static text."

    def test_outputs_custom_text(self):
        from quartermaster_nodes.nodes.data.static import StaticNode1

        ctx = MockNodeContext(node_metadata={"static_text": "Custom text here"})
        StaticNode1.think(ctx)
        assert ctx.handle.last_text == "Custom text here"

    def test_raises_without_thought(self):
        from quartermaster_nodes.nodes.data.static import StaticNode1

        ctx = MockNodeContext(thought=None)
        with pytest.raises(ValueError):
            StaticNode1.think(ctx)


class TestTextNode:
    def test_renders_template(self):
        from quartermaster_nodes.nodes.data.text import TextNode

        ctx = MockNodeContext(
            node_metadata={"text": "Hello {{ name }}!"},
            thought=MockThought(metadata={"name": "World"}),
        )
        TextNode.think(ctx)
        assert ctx.handle.last_text == "Hello World!"

    def test_renders_default_text(self):
        from quartermaster_nodes.nodes.data.text import TextNode

        ctx = MockNodeContext(thought=MockThought())
        TextNode.think(ctx)
        assert ctx.handle.last_text == "This is a default text."

    def test_renders_with_multiple_variables(self):
        from quartermaster_nodes.nodes.data.text import TextNode

        ctx = MockNodeContext(
            node_metadata={"text": "{{ a }} + {{ b }} = {{ c }}"},
            thought=MockThought(metadata={"a": 1, "b": 2, "c": 3}),
        )
        TextNode.think(ctx)
        assert ctx.handle.last_text == "1 + 2 = 3"

    def test_raises_without_thought(self):
        from quartermaster_nodes.nodes.data.text import TextNode

        ctx = MockNodeContext(thought=None)
        with pytest.raises(ValueError):
            TextNode.think(ctx)


class TestVarNode:
    def test_evaluates_expression(self):
        from quartermaster_nodes.nodes.data.var import VarNode

        ctx = MockNodeContext(
            node_metadata={"name": "result", "expression": "x + y"},
            thought=MockThought(metadata={"x": 3, "y": 4}),
        )
        VarNode.think(ctx)
        assert ctx.handle.last_metadata_update == {"result": 7}

    def test_string_expression(self):
        from quartermaster_nodes.nodes.data.var import VarNode

        ctx = MockNodeContext(
            node_metadata={"name": "greeting", "expression": "'hello ' + name"},
            thought=MockThought(metadata={"name": "world"}),
        )
        VarNode.think(ctx)
        assert ctx.handle.last_metadata_update == {"greeting": "hello world"}

    def test_raises_without_thought(self):
        from quartermaster_nodes.nodes.data.var import VarNode
        from quartermaster_nodes.exceptions import MissingMemoryIDException

        ctx = MockNodeContext(thought=None)
        with pytest.raises(MissingMemoryIDException):
            VarNode.think(ctx)


class TestTextToVariableNode:
    def test_stores_text_as_variable(self):
        from quartermaster_nodes.nodes.data.text_to_variable import TextToVariableNode

        ctx = MockNodeContext(
            node_metadata={"variable_name": "my_var"},
            thought=MockThought(text="Some LLM output"),
        )
        TextToVariableNode.think(ctx)
        assert ctx.handle.last_metadata_update == {"my_var": "Some LLM output"}

    def test_default_variable_name(self):
        from quartermaster_nodes.nodes.data.text_to_variable import TextToVariableNode

        ctx = MockNodeContext(thought=MockThought(text="output"))
        TextToVariableNode.think(ctx)
        assert "custom_variable" in ctx.handle.last_metadata_update


class TestCodeNode:
    def test_info_and_config(self):
        from quartermaster_nodes.nodes.data.code import CodeNode

        info = CodeNode.info()
        assert "code" in info.description.lower()
        config = CodeNode.flow_config()
        assert not config.accepts_incoming_edges
        assert not config.accepts_outgoing_edges


class TestProgramRunner:
    def test_calls_executor(self):
        from quartermaster_nodes.nodes.data.program_runner import ProgramRunner1

        executed = []
        ctx = MockNodeContext(
            node_metadata={
                "program_version_id": "prog-1",
                "parameters": {"key": "val"},
                "_program_executor": lambda pid, params, c: executed.append(pid),
            }
        )
        ProgramRunner1.think(ctx)
        assert executed == ["prog-1"]


class TestStaticMerge:
    def test_outputs_text(self):
        from quartermaster_nodes.nodes.data.static_merge import StaticMerge1

        ctx = MockNodeContext(node_metadata={"static_text": "merged content"})
        StaticMerge1.think(ctx)
        assert ctx.handle.last_text == "merged content"


class TestStaticDecision:
    def test_picks_path(self):
        from quartermaster_nodes.nodes.data.static_decision import StaticDecision1
        from tests.conftest import MockEdge, MockEdgeQuerySet, MockAssistantNode
        from quartermaster_nodes.enums import NEXT_ASSISTANT_NODE_ID

        true_id = "true-path"
        false_id = "false-path"

        ctx = MockNodeContext(
            node_metadata={"expression": "x > 0"},
            thought=MockThought(metadata={"x": 5}),
            assistant_node=MockAssistantNode(
                predecessor_edges=MockEdgeQuerySet([
                    MockEdge(tail_id=true_id, main_direction=True, direction_text="yes"),
                    MockEdge(tail_id=false_id, main_direction=False, direction_text="no"),
                ])
            ),
        )
        StaticDecision1.think(ctx)
        assert ctx.handle.last_metadata_update[NEXT_ASSISTANT_NODE_ID] == true_id
