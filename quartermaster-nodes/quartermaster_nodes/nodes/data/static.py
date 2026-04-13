"""Static node — pass-through static text content."""

from quartermaster_nodes.base import AbstractAssistantNode
from quartermaster_nodes.config import AssistantInfo, FlowNodeConf
from quartermaster_nodes.enums import (
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)


class StaticNode1(AbstractAssistantNode):
    """Output static text content without LLM processing.

    Use Case:
        - Display fixed messages or instructions
        - Provide static context to subsequent nodes
    """

    metadata_static_text_key = "static_text"
    metadata_static_text_default = "This is a default static text."

    @classmethod
    def name(cls) -> str:
        return "StaticAssistant"

    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = "Output static text content"
        info.instructions = "Stores and outputs static text without LLM processing"
        info.metadata = {
            cls.metadata_static_text_key: cls.metadata_static_text_default,
        }
        return info

    @classmethod
    def flow_config(cls) -> FlowNodeConf:
        return FlowNodeConf(
            traverse_in=AvailableTraversingIn.AwaitFirst,
            traverse_out=AvailableTraversingOut.SpawnAll,
            thought_type=AvailableThoughtTypes.NewThought1,
            message_type=AvailableMessageTypes.Automatic,
            available_thought_types={
                AvailableThoughtTypes.EditSameOrAddNew1,
                AvailableThoughtTypes.UsePreviousThought1,
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
        value = cls.get_metadata_key_value(
            ctx, cls.metadata_static_text_key, cls.metadata_static_text_default
        )

        if ctx.thought is None:
            raise ValueError("Memory ID cannot be None")

        assert ctx.handle is not None, "handle not set"
        ctx.handle.append_text(value)
