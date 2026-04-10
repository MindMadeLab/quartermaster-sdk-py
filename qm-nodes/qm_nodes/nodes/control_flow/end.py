"""End node — flow termination."""

from qm_nodes.base import AbstractAssistantNode
from qm_nodes.config import AssistantInfo, FlowNodeConf
from qm_nodes.enums import (
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
    def version(cls) -> str:
        return "1.0.0"

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
