"""TextToVariable node — extract thought text into a variable."""

from qm_nodes.base import AbstractAssistantNode
from qm_nodes.config import AssistantInfo, FlowNodeConf
from qm_nodes.enums import (
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)


class TextToVariableNode(AbstractAssistantNode):
    """Read thought text and store it in metadata under a custom variable name.

    Use Case:
        - Capture LLM output as a variable for downstream processing
        - Convert thought text into a named variable
    """

    metadata_variable_name_key = "variable_name"
    metadata_variable_name_default = "custom_variable"

    @classmethod
    def name(cls) -> str:
        return "TextToVariableNode1"

    @classmethod
    def version(cls) -> str:
        return "1.0"

    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = "Extract thought text into a named variable"
        info.instructions = "Reads thought text and stores it in metadata"
        info.metadata = {
            cls.metadata_variable_name_key: cls.metadata_variable_name_default,
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
        variable_name = cls.get_metadata_key_value(
            ctx, cls.metadata_variable_name_key, cls.metadata_variable_name_default
        )

        if ctx.thought is None:
            raise ValueError("Memory ID cannot be None")

        assert ctx.handle is not None, "handle not set"
        ctx.handle.update_metadata({variable_name: ctx.thought.text})
