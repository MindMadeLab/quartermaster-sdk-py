"""UserMemory node — user-scoped persistent memory."""

from qm_nodes.base import AbstractAssistantNode
from qm_nodes.config import AssistantInfo, FlowNodeConf
from qm_nodes.enums import (
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)


class UserMemoryNode(AbstractAssistantNode):
    """User-scoped persistent memory that persists across flow executions.

    Use Case:
        - Store user preferences across sessions
        - Maintain user-specific state
    """

    metadata_memory_name_key = "memory_name"
    metadata_memory_name_default = "default"
    metadata_initial_data_key = "initial_data"
    metadata_initial_data_default = []

    @classmethod
    def name(cls) -> str:
        return "UserMemory1"

    @classmethod
    def version(cls) -> str:
        return "1.0"

    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = "User-scoped persistent memory"
        info.instructions = "Stores data that persists across flow executions per user"
        info.metadata = {
            cls.metadata_memory_name_key: cls.metadata_memory_name_default,
            cls.metadata_initial_data_key: cls.metadata_initial_data_default,
        }
        return info

    @classmethod
    def flow_config(cls) -> FlowNodeConf:
        return FlowNodeConf(
            traverse_in=AvailableTraversingIn.AwaitFirst,
            traverse_out=AvailableTraversingOut.SpawnAll,
            thought_type=AvailableThoughtTypes.SkipThought1,
            message_type=AvailableMessageTypes.Variable,
            accepts_incoming_edges=False,
            accepts_outgoing_edges=False,
        )

    @classmethod
    def think(cls, ctx) -> None:
        pass  # User memory initialization is handled by StartNode
