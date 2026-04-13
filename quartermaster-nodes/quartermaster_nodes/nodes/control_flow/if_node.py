"""If node — binary conditional branching based on expression evaluation."""

from quartermaster_nodes.base import AbstractAssistantNode
from quartermaster_nodes.config import AssistantInfo, FlowNodeConf
from quartermaster_nodes.enums import (
    NEXT_ASSISTANT_NODE_ID,
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)


class IfNode(AbstractAssistantNode):
    """Binary conditional branching based on Python expression evaluation.

    Evaluates an expression and picks the true or false branch.

    Use Case:
        - Create conditional flows based on dynamic python expressions
        - Branch the flow based on specific criteria or conditions
    """

    metadata_if_expression_key = "if_expression"
    metadata_if_expression_default = ""

    @classmethod
    def name(cls) -> str:
        return "IfNode1"

    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = "Evaluate an expression and branch true/false"
        info.instructions = "Binary conditional branching based on Python expression"
        info.metadata = {
            cls.metadata_if_expression_key: cls.metadata_if_expression_default,
        }
        return info

    @classmethod
    def flow_config(cls) -> FlowNodeConf:
        return FlowNodeConf(
            traverse_in=AvailableTraversingIn.AwaitFirst,
            traverse_out=AvailableTraversingOut.SpawnPickedNode,
            thought_type=AvailableThoughtTypes.UsePreviousThought1,
            message_type=AvailableMessageTypes.Variable,
            available_thought_types={
                AvailableThoughtTypes.EditSameOrAddNew1,
                AvailableThoughtTypes.UsePreviousThought1,
                AvailableThoughtTypes.NewHiddenThought1,
                AvailableThoughtTypes.NewCollapsedThought1,
            },
            available_message_types={
                AvailableMessageTypes.Assistant,
                AvailableMessageTypes.User,
            },
        )

    @classmethod
    def think(cls, ctx) -> None:
        if ctx.thought is None:
            raise ValueError("Memory ID cannot be None")

        if_expression = cls.get_metadata_key_value(
            ctx, cls.metadata_if_expression_key, cls.metadata_if_expression_default
        )

        thought = ctx.thought
        metadata = thought.metadata

        edges = ctx.assistant_node.predecessor_edges.all()
        true_node = next((e.tail_id for e in edges if e.main_direction), None)
        false_node = next((e.tail_id for e in edges if not e.main_direction), None)

        if not true_node or not false_node:
            raise ValueError(
                "IfNode must have one edge with main_direction=True and one with main_direction=False"
            )

        # Evaluate expression using injected evaluator or simple eval
        evaluator = ctx.node_metadata.get("_expression_evaluator")
        if evaluator is not None:
            result = evaluator.eval_expression(ctx.flow_node_id, if_expression, metadata)
            eval_result = result.result
        else:
            eval_result = eval(if_expression, {"__builtins__": {}}, metadata)

        picked_node = true_node if eval_result else false_node

        assert ctx.handle is not None, "handle not set"
        ctx.handle.update_metadata({NEXT_ASSISTANT_NODE_ID: picked_node})
