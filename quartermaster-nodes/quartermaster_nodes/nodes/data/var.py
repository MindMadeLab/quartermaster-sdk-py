"""Var node — evaluate expression and store as variable."""

from quartermaster_nodes.base import AbstractAssistantNode
from quartermaster_nodes.config import AssistantInfo, FlowNodeConf
from quartermaster_nodes.enums import (
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)
from quartermaster_nodes.exceptions import MissingMemoryIDException


class VarNode(AbstractAssistantNode):
    """Evaluate a Python expression and store the result as a variable.

    Use Case:
        - Compute derived values from existing variables
        - Transform data between nodes
    """

    metadata_var_name_key = "name"
    metadata_var_name_default = ""
    metadata_var_expression_key = "expression"
    metadata_var_expression_default = ""

    @classmethod
    def name(cls) -> str:
        return "VarNode1"

    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = "Evaluate expression and store as variable"
        info.instructions = "Computes a Python expression and stores the result in metadata"
        info.metadata = {
            cls.metadata_var_name_key: cls.metadata_var_name_default,
            cls.metadata_var_expression_key: cls.metadata_var_expression_default,
        }
        return info

    @classmethod
    def flow_config(cls) -> FlowNodeConf:
        return FlowNodeConf(
            traverse_in=AvailableTraversingIn.AwaitFirst,
            traverse_out=AvailableTraversingOut.SpawnAll,
            thought_type=AvailableThoughtTypes.UsePreviousThought1,
            message_type=AvailableMessageTypes.Variable,
            available_thought_types={
                AvailableThoughtTypes.EditSameOrAddNew1,
                AvailableThoughtTypes.NewThought1,
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
            raise MissingMemoryIDException()

        var_name = cls.get_metadata_key_value(
            ctx, cls.metadata_var_name_key, cls.metadata_var_name_default
        )
        expression = cls.get_metadata_key_value(
            ctx, cls.metadata_var_expression_key, cls.metadata_var_expression_default
        )

        metadata = ctx.thought.metadata

        evaluator = ctx.node_metadata.get("_expression_evaluator")
        if evaluator is not None:
            result = evaluator.eval_expression(ctx.flow_node_id, expression, metadata)
            eval_result = result.result
        else:
            from quartermaster_nodes.safe_eval import safe_eval

            eval_result = safe_eval(expression, metadata)

        assert ctx.handle is not None, "handle not set"
        ctx.handle.update_metadata({var_name: eval_result})
