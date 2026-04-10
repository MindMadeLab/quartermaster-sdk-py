"""UseFile node — attach file to context."""

from qm_nodes.base import AbstractAssistantNode
from qm_nodes.config import AssistantInfo, FlowNodeConf
from qm_nodes.enums import (
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)


class UseFileNode(AbstractAssistantNode):
    """Attach a file to the execution context.

    Use Case:
        - Provide file content as context for LLM nodes
        - Load documents for processing in the flow
    """

    metadata_file_id_key = "file_id"
    metadata_file_id_default = None

    @classmethod
    def name(cls) -> str:
        return "UseFile1"

    @classmethod
    def version(cls) -> str:
        return "1.0.0"

    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = "Attach file to context"
        info.instructions = "Loads a file into the execution context"
        info.metadata = {
            cls.metadata_file_id_key: cls.metadata_file_id_default,
        }
        return info

    @classmethod
    def flow_config(cls) -> FlowNodeConf:
        return FlowNodeConf(
            traverse_in=AvailableTraversingIn.AwaitFirst,
            traverse_out=AvailableTraversingOut.SpawnAll,
            thought_type=AvailableThoughtTypes.SkipThought1,
            message_type=AvailableMessageTypes.Variable,
        )

    @classmethod
    def think(cls, ctx) -> None:
        file_loader = ctx.node_metadata.get("_file_loader")
        file_id = cls.get_metadata_key_value(
            ctx, cls.metadata_file_id_key, cls.metadata_file_id_default
        )
        if file_loader is not None and file_id is not None:
            file_loader(file_id, ctx)
