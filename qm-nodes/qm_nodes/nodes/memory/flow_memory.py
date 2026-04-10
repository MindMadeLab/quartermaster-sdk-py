"""FlowMemory node — flow-scoped persistent memory."""

from qm_nodes.base import AbstractAssistantNode
from qm_nodes.config import AssistantInfo, FlowNodeConf
from qm_nodes.enums import (
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)


class FlowMemoryNode(AbstractAssistantNode):
    """Flow-scoped persistent memory for storing data across nodes.

    Data is scoped to a single flow execution and accessible by all
    nodes within that flow.

    Use Case:
        - Store data that needs to be shared between nodes in a flow
        - Initialize persistent variables at flow start
    """

    metadata_memory_name_key = "memory_name"
    metadata_memory_name_default = "default"
    metadata_initial_data_key = "initial_data"
    metadata_initial_data_default = []

    @classmethod
    def name(cls) -> str:
        return "FlowMemory"

    @classmethod
    def version(cls) -> str:
        return "1.0"

    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = "Flow-scoped persistent memory"
        info.instructions = "Stores data accessible by all nodes in the flow"
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
        pass  # Memory initialization is handled by StartNode
