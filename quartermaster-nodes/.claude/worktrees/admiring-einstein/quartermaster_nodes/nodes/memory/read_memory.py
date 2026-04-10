"""ReadMemory node — read from persistent memory."""

from quartermaster_nodes.base import AbstractAssistantNode
from quartermaster_nodes.config import AssistantInfo, FlowNodeConf
from quartermaster_nodes.enums import (
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)


class ReadMemoryNode(AbstractAssistantNode):
    """Read variables from persistent memory into thought metadata.

    Use Case:
        - Load previously stored data for use in the current flow
        - Access flow-scoped or user-scoped memory
    """

    metadata_memory_name_key = "memory_name"
    metadata_memory_name_default = "default"
    metadata_memory_type_key = "memory_type"
    metadata_memory_type_default = "flow"
    metadata_variable_names_key = "variable_names"
    metadata_variable_names_default = []

    @classmethod
    def name(cls) -> str:
        return "ReadMemory1"

    @classmethod
    def version(cls) -> str:
        return "1.0"

    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = "Read variables from persistent memory"
        info.instructions = "Loads stored data into thought metadata"
        info.metadata = {
            cls.metadata_memory_name_key: cls.metadata_memory_name_default,
            cls.metadata_memory_type_key: cls.metadata_memory_type_default,
            cls.metadata_variable_names_key: cls.metadata_variable_names_default,
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
        memory_reader = ctx.node_metadata.get("_memory_reader")
        memory_name = cls.get_metadata_key_value(
            ctx, cls.metadata_memory_name_key, cls.metadata_memory_name_default
        )
        memory_type = cls.get_metadata_key_value(
            ctx, cls.metadata_memory_type_key, cls.metadata_memory_type_default
        )
        variable_names = cls.get_metadata_key_value(
            ctx, cls.metadata_variable_names_key, cls.metadata_variable_names_default
        )

        if memory_reader is not None:
            variables = memory_reader(memory_name, memory_type, variable_names, ctx)
            if variables and ctx.handle is not None:
                ctx.handle.update_metadata(variables)
