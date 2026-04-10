"""Break node — message collection boundary."""

from qm_nodes.base import AbstractAssistantNode
from qm_nodes.config import AssistantInfo, FlowNodeConf
from qm_nodes.enums import (
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)


class BreakNode1(AbstractAssistantNode):
    """Message collection boundary that stops backward message traversal.

    When the LLM builds context by traversing the flow backwards,
    it stops at Break nodes. Configurable break_targets allow
    selective clearing of tools or thinking messages.

    Use Case:
        - Control the amount of context fed to the LLM
        - Create logical conversation segments
        - Optimize token usage by truncating message collection
        - Selectively clear tool or thinking messages
    """

    metadata_break_targets_key = "break_targets"
    metadata_break_targets_default = []

    @classmethod
    def name(cls) -> str:
        return "Break1"

    @classmethod
    def version(cls) -> str:
        return "1.0.0"

    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = (
            "Message collection boundary that stops LLM context building"
        )
        info.instructions = (
            "Place at strategic points to limit message history collection. "
            "break_targets: [] = full break, ['tools'] = clear tools, "
            "['thinking'] = clear thinking"
        )
        info.metadata = {
            cls.metadata_break_targets_key: cls.metadata_break_targets_default,
        }
        return info

    @classmethod
    def flow_config(cls) -> FlowNodeConf:
        return FlowNodeConf(
            thought_type=AvailableThoughtTypes.SkipThought1,
            traverse_in=AvailableTraversingIn.AwaitFirst,
            traverse_out=AvailableTraversingOut.SpawnAll,
            message_type=AvailableMessageTypes.System,
        )

    @classmethod
    def think(cls, ctx) -> None:
        pass  # Break node acts as signal during message collection, no runtime logic
