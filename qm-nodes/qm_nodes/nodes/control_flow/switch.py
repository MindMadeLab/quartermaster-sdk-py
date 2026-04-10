"""Switch node — multi-way branching based on case expressions."""

from qm_nodes.base import AbstractAssistantNode
from qm_nodes.config import AssistantInfo, FlowNodeConf
from qm_nodes.enums import (
    NEXT_ASSISTANT_NODE_ID,
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)


class SwitchNode1(AbstractAssistantNode):
    """Multi-way branching based on multiple case expressions.

    Evaluates cases in order; the first truthy result wins.
    Falls back to default edge if no case matches.

    Use Case:
        - Multi-way branching based on dynamic Python expressions
        - When you need more than two branches (unlike IfNode)
    """

    metadata_cases_key = "cases"
    metadata_cases_default = []
    metadata_default_edge_key = "default_edge_id"
    metadata_default_edge_default = None

    @classmethod
    def name(cls) -> str:
        return "Switch1"

    @classmethod
    def version(cls) -> str:
        return "1.0"

    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = "Evaluate multiple case expressions and pick the first match"
        info.instructions = "Multi-way branching with fallback to default"
        info.metadata = {
            cls.metadata_cases_key: cls.metadata_cases_default,
            cls.metadata_default_edge_key: cls.metadata_default_edge_default,
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

        cases = cls.get_metadata_key_value(ctx, cls.metadata_cases_key, cls.metadata_cases_default)
        metadata = ctx.thought.metadata

        default_edge_id = cls.get_metadata_key_value(
            ctx, cls.metadata_default_edge_key, cls.metadata_default_edge_default
        )

        edges = ctx.assistant_node.predecessor_edges.all()
        edge_map = {str(e.tail_id): e.tail_id for e in edges}

        default_node = edge_map.get(str(default_edge_id)) if default_edge_id else None

        evaluator = ctx.node_metadata.get("_expression_evaluator")
        picked_node = None

        for case in cases:
            edge_id = str(case.get("edge_id", ""))
            expression = case.get("expression", "")

            if not expression or edge_id not in edge_map:
                continue

            try:
                if evaluator is not None:
                    result = evaluator.eval_expression(ctx.flow_node_id, expression, metadata)
                    eval_result = result.result
                else:
                    eval_result = eval(expression, {"__builtins__": {}}, metadata)

                if eval_result:
                    picked_node = edge_map[edge_id]
                    break
            except Exception:
                continue

        if picked_node is None:
            picked_node = default_node

        if picked_node is None:
            raise ValueError("SwitchNode: no case matched and no default edge is configured")

        assert ctx.handle is not None, "handle not set"
        ctx.handle.update_metadata({NEXT_ASSISTANT_NODE_ID: picked_node})
