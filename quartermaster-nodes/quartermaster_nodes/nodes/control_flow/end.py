"""End node — flow termination."""

from quartermaster_nodes.base import AbstractAssistantNode
from quartermaster_nodes.config import AssistantInfo, FlowNodeConf
from quartermaster_nodes.enums import (
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)


class EndNodeV1(AbstractAssistantNode):
    """Flow termination node. Spawns Start nodes of connected agents.

    Use Case:
        - End the current flow and start a new one in a connected agent
        - Create clear separation between different flows
    """

    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = "End flow and spawn connected agent Start nodes"
        info.instructions = "Place at the end of every flow"
        info.metadata = {}
        return info

    @classmethod
    def name(cls) -> str:
        return "EndNode"

    @classmethod
    def flow_config(cls) -> FlowNodeConf:
        return FlowNodeConf(
            traverse_in=AvailableTraversingIn.AwaitFirst,
            traverse_out=AvailableTraversingOut.SpawnStart,
            thought_type=AvailableThoughtTypes.SkipThought1,
            message_type=AvailableMessageTypes.Variable,
            accepts_outgoing_edges=False,
        )

    @classmethod
    def think(cls, ctx) -> None:
        pass  # End node has no logic — traversal config handles spawning
