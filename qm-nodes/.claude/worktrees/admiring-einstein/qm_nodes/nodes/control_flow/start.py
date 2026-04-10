"""Start node — flow entry point."""

from qm_nodes.base import AbstractAssistantNode
from qm_nodes.config import AssistantInfo, FlowNodeConf
from qm_nodes.enums import (
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)


class StartNodeV1(AbstractAssistantNode):
    """Flow entry point. Initializes flow and memory nodes.

    Use Case:
        - Entry point for every flow
        - Initializes flow-scoped and user-scoped memory
    """

    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = "Flow entry point that initializes memory nodes"
        info.instructions = "Place at the start of every flow"
        info.metadata = {}
        return info

    @classmethod
    def name(cls) -> str:
        return "StartNode"

    @classmethod
    def version(cls) -> str:
        return "1.0"

    @classmethod
    def flow_config(cls) -> FlowNodeConf:
        return FlowNodeConf(
            traverse_in=AvailableTraversingIn.AwaitFirst,
            traverse_out=AvailableTraversingOut.SpawnAll,
            thought_type=AvailableThoughtTypes.SkipThought1,
            message_type=AvailableMessageTypes.Variable,
            accepts_incoming_edges=False,
        )

    @classmethod
    def think(cls, ctx) -> None:
        memory_initializer = ctx.node_metadata.get("_memory_initializer")
        if memory_initializer is not None:
            memory_initializer(ctx)
