"""ViewMetadata node — debug: inspect node metadata."""

from quartermaster_nodes.base import AbstractAssistantNode
from quartermaster_nodes.config import AssistantInfo, FlowNodeConf
from quartermaster_nodes.enums import (
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)


class ViewMetadataNode(AbstractAssistantNode):
    """Debug node to inspect and display node metadata.

    Use Case:
        - Debug flows by inspecting current metadata state
        - Verify variable values during development
    """

    @classmethod
    def name(cls) -> str:
        return "ViewMetadata1"

    @classmethod
    def version(cls) -> str:
        return "1.0.0"

    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = "Debug: inspect node metadata"
        info.instructions = "Displays current metadata for debugging"
        info.metadata = {}
        return info

    @classmethod
    def flow_config(cls) -> FlowNodeConf:
        return FlowNodeConf(
            traverse_in=AvailableTraversingIn.AwaitFirst,
            traverse_out=AvailableTraversingOut.SpawnAll,
            thought_type=AvailableThoughtTypes.NewThought1,
            message_type=AvailableMessageTypes.Automatic,
        )

    @classmethod
    def think(cls, ctx) -> None:
        if ctx.thought is not None and ctx.handle is not None:
            import json
            metadata_str = json.dumps(ctx.thought.metadata, indent=2, default=str)
            ctx.handle.append_text(f"Metadata:\n{metadata_str}")
