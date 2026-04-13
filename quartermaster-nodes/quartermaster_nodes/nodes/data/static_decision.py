"""Static decision — rule-based decision without LLM."""

from quartermaster_nodes.base import AbstractAssistantNode
from quartermaster_nodes.config import AssistantInfo, FlowNodeConf
from quartermaster_nodes.enums import (
    NEXT_ASSISTANT_NODE_ID,
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)


class StaticDecision1(AbstractAssistantNode):
    """Rule-based decision without LLM.

    Picks a path based on a static expression evaluation.

    Use Case:
        - Deterministic path selection without LLM overhead
        - Simple rule-based routing
    """

    metadata_expression_key = "expression"
    metadata_expression_default = ""

    @classmethod
    def name(cls) -> str:
        return "StaticDecision1"

    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = "Rule-based decision without LLM"
        info.instructions = "Picks a path based on static expression evaluation"
        info.metadata = {
            cls.metadata_expression_key: cls.metadata_expression_default,
        }
        return info

    @classmethod
    def flow_config(cls) -> FlowNodeConf:
        return FlowNodeConf(
            traverse_in=AvailableTraversingIn.AwaitFirst,
            traverse_out=AvailableTraversingOut.SpawnPickedNode,
            thought_type=AvailableThoughtTypes.UsePreviousThought1,
            message_type=AvailableMessageTypes.Variable,
        )

    @classmethod
    def think(cls, ctx) -> None:
        if ctx.thought is None:
            raise ValueError("Memory ID cannot be None")

        expression = cls.get_metadata_key_value(
            ctx, cls.metadata_expression_key, cls.metadata_expression_default
        )
        metadata = ctx.thought.metadata

        edges = ctx.assistant_node.predecessor_edges.all()
        true_node = next((e.tail_id for e in edges if e.main_direction), None)
        false_node = next((e.tail_id for e in edges if not e.main_direction), None)

        evaluator = ctx.node_metadata.get("_expression_evaluator")
        if evaluator is not None:
            result = evaluator.eval_expression(ctx.flow_node_id, expression, metadata)
            eval_result = result.result
        else:
            from quartermaster_nodes.safe_eval import safe_eval
            eval_result = safe_eval(expression, metadata)

        picked_node = true_node if eval_result else false_node

        if picked_node is not None:
            assert ctx.handle is not None, "handle not set"
            ctx.handle.update_metadata({NEXT_ASSISTANT_NODE_ID: picked_node})
