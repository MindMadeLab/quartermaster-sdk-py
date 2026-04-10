"""Static merge — merge with static content."""

from qm_nodes.base import AbstractAssistantNode
from qm_nodes.config import AssistantInfo, FlowNodeConf
from qm_nodes.enums import (
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)


class StaticMerge1(AbstractAssistantNode):
    """Merge multiple branches with static text content.

    Like Merge1 but uses static text instead of LLM to combine branches.

    Use Case:
        - Combine branches without LLM overhead
        - Append static content after merging paths
    """

    metadata_static_text_key = "static_text"
    metadata_static_text_default = ""

    @classmethod
    def name(cls) -> str:
        return "StaticMerge1"

    @classmethod
    def version(cls) -> str:
        return "1.0"

    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = "Merge branches with static text"
        info.instructions = "Combines parallel branches using static text"
        info.metadata = {
            cls.metadata_static_text_key: cls.metadata_static_text_default,
        }
        return info

    @classmethod
    def flow_config(cls) -> FlowNodeConf:
        return FlowNodeConf(
            traverse_in=AvailableTraversingIn.AwaitAll,
            traverse_out=AvailableTraversingOut.SpawnAll,
            thought_type=AvailableThoughtTypes.NewThought1,
            message_type=AvailableMessageTypes.Automatic,
        )

    @classmethod
    def think(cls, ctx) -> None:
        value = cls.get_metadata_key_value(
            ctx, cls.metadata_static_text_key, cls.metadata_static_text_default
        )
        if ctx.thought is None:
            raise ValueError("Memory ID cannot be None")
        assert ctx.handle is not None, "handle not set"
        ctx.handle.append_text(value)
