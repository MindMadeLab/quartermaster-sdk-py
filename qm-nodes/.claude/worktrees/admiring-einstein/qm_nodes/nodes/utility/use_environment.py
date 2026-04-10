"""UseEnvironment node — activate an execution environment."""

from qm_nodes.base import AbstractAssistantNode
from qm_nodes.config import AssistantInfo, FlowNodeConf
from qm_nodes.enums import (
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)


class UseEnvironmentNode(AbstractAssistantNode):
    """Activate an execution environment.

    Use Case:
        - Set up a specific runtime environment for subsequent nodes
        - Configure tool execution context
    """

    metadata_environment_id_key = "environment_id"
    metadata_environment_id_default = None

    @classmethod
    def name(cls) -> str:
        return "UseEnvironment1"

    @classmethod
    def version(cls) -> str:
        return "1.0.0"

    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = "Activate an execution environment"
        info.instructions = "Sets up environment for subsequent node execution"
        info.metadata = {
            cls.metadata_environment_id_key: cls.metadata_environment_id_default,
        }
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
        env_activator = ctx.node_metadata.get("_environment_activator")
        env_id = cls.get_metadata_key_value(
            ctx, cls.metadata_environment_id_key, cls.metadata_environment_id_default
        )
        if env_activator is not None and env_id is not None:
            env_activator(env_id, ctx)
