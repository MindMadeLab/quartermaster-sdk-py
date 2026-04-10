"""WriteMemory node — write to persistent memory."""

from quartermaster_nodes.base import AbstractAssistantNode
from quartermaster_nodes.config import AssistantInfo, FlowNodeConf
from quartermaster_nodes.enums import (
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)


class WriteMemoryNode(AbstractAssistantNode):
    """Write variables to persistent memory.

    Use Case:
        - Persist computed values for later use
        - Store results across flow executions
    """

    metadata_memory_name_key = "memory_name"
    metadata_memory_name_default = "default"
    metadata_memory_type_key = "memory_type"
    metadata_memory_type_default = "flow"
    metadata_variables_key = "variables"
    metadata_variables_default = []

    @classmethod
    def name(cls) -> str:
        return "WriteMemory1"

    @classmethod
    def version(cls) -> str:
        return "1.0"

    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = "Write variables to persistent memory"
        info.instructions = "Persists data from thought metadata to memory"
        info.metadata = {
            cls.metadata_memory_name_key: cls.metadata_memory_name_default,
            cls.metadata_memory_type_key: cls.metadata_memory_type_default,
            cls.metadata_variables_key: cls.metadata_variables_default,
        }
        return info

    @classmethod
    def flow_config(cls) -> FlowNodeConf:
        return FlowNodeConf(
            traverse_in=AvailableTraversingIn.AwaitFirst,
            traverse_out=AvailableTraversingOut.SpawnAll,
            thought_type=AvailableThoughtTypes.UsePreviousThought1,
            message_type=AvailableMessageTypes.Variable,
        )

    @classmethod
    def think(cls, ctx) -> None:
        memory_writer = ctx.node_metadata.get("_memory_writer")
        memory_name = cls.get_metadata_key_value(
            ctx, cls.metadata_memory_name_key, cls.metadata_memory_name_default
        )
        memory_type = cls.get_metadata_key_value(
            ctx, cls.metadata_memory_type_key, cls.metadata_memory_type_default
        )
        variables = cls.get_metadata_key_value(
            ctx, cls.metadata_variables_key, cls.metadata_variables_default
        )

        if memory_writer is not None and ctx.thought is not None:
            data = {}
            for var in variables:
                var_name = var.get("name", "")
                expression = var.get("expression", "")
                if var_name:
                    evaluator = ctx.node_metadata.get("_expression_evaluator")
                    if evaluator is not None:
                        result = evaluator.eval_expression(
                            ctx.flow_node_id, expression, ctx.thought.metadata
                        )
                        data[var_name] = result.result
                    elif expression:
                        data[var_name] = eval(
                            expression, {"__builtins__": {}}, ctx.thought.metadata
                        )
            memory_writer(memory_name, memory_type, data, ctx)
