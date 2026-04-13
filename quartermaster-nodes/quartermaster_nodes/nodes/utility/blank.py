"""Blank node — no-op placeholder."""

from quartermaster_nodes.base import AbstractAssistantNode
from quartermaster_nodes.config import AssistantInfo, FlowNodeConf
from quartermaster_nodes.enums import (
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)


class BlankNode(AbstractAssistantNode):
    """No-op placeholder node.

    Use Case:
        - Placeholder in the agent graph
        - Merging connections before Merge nodes
        - Simplifying graph structure for readability
    """

    @classmethod
    def name(cls) -> str:
        return "Blank1"

    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = "No-op placeholder node"
        info.instructions = "Passes through without processing"
        info.metadata = {}
        return info

    @classmethod
    def flow_config(cls) -> FlowNodeConf:
        return FlowNodeConf(
            thought_type=AvailableThoughtTypes.SkipThought1,
            traverse_in=AvailableTraversingIn.AwaitFirst,
            traverse_out=AvailableTraversingOut.SpawnAll,
            message_type=AvailableMessageTypes.Variable,
            available_traversing_in={AvailableTraversingIn.AwaitAll},
            available_thought_types={
                AvailableThoughtTypes.NewThought1,
                AvailableThoughtTypes.NewHiddenThought1,
                AvailableThoughtTypes.NewCollapsedThought1,
                AvailableThoughtTypes.UsePreviousThought1,
            },
        )

    @classmethod
    def think(cls, ctx) -> None:
        pass  # No-op
