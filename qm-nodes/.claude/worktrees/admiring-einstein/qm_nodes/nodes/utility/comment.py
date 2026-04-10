"""Comment node — documentation in the graph."""

from qm_nodes.base import AbstractAssistantNode
from qm_nodes.config import AssistantInfo, FlowNodeConf
from qm_nodes.enums import (
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)


class CommentNode(AbstractAssistantNode):
    """Documentation node for adding comments to the graph.

    Use Case:
        - Add notes or documentation to the agent graph
        - Document purpose of specific flow sections
    """

    metadata_comment_key = "comment"
    metadata_comment_default = ""

    @classmethod
    def name(cls) -> str:
        return "Comment1"

    @classmethod
    def version(cls) -> str:
        return "1.0.0"

    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = "Add comments to the agent graph"
        info.instructions = "Documentation-only node, not connected to the flow"
        info.metadata = {
            cls.metadata_comment_key: cls.metadata_comment_default,
        }
        return info

    @classmethod
    def flow_config(cls) -> FlowNodeConf:
        return FlowNodeConf(
            thought_type=AvailableThoughtTypes.SkipThought1,
            traverse_in=AvailableTraversingIn.AwaitFirst,
            traverse_out=AvailableTraversingOut.SpawnAll,
            message_type=AvailableMessageTypes.Variable,
            accepts_incoming_edges=False,
            accepts_outgoing_edges=False,
        )

    @classmethod
    def think(cls, ctx) -> None:
        pass  # Comment nodes have no runtime logic
