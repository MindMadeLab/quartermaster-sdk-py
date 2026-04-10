"""UnselectEnvironment node — deactivate execution environment."""

from qm_nodes.base import AbstractAssistantNode
from qm_nodes.config import AssistantInfo, FlowNodeConf
from qm_nodes.enums import (
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)


class UnselectEnvironmentNode(AbstractAssistantNode):
    """Deactivate the current execution environment.

    Use Case:
        - Clean up environment after use
        - Reset execution context
    """

    @classmethod
    def name(cls) -> str:
        return "UnselectEnvironment1"

    @classmethod
    def version(cls) -> str:
        return "1.0.0"

    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = "Deactivate execution environment"
        info.instructions = "Cleans up and deactivates the current environment"
        info.metadata = {}
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
        env_deactivator = ctx.node_metadata.get("_environment_deactivator")
        if env_deactivator is not None:
            env_deactivator(ctx)
