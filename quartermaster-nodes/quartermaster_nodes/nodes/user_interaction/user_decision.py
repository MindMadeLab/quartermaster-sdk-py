"""UserDecision node — present choices to user."""

from quartermaster_nodes.base import AbstractAssistantNode
from quartermaster_nodes.config import AssistantInfo, FlowNodeConf
from quartermaster_nodes.enums import (
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)
from quartermaster_nodes.exceptions import MissingMemoryIDException, ProcessStopException


class UserDecisionV1(AbstractAssistantNode):
    """Present choices to the user and await their selection.

    Use Case:
        - Let users choose between different flow paths
        - Create interactive decision points
    """

    @classmethod
    def name(cls) -> str:
        return "UserDecision1"

    @classmethod
    def version(cls) -> str:
        return "1.0.0"

    @classmethod
    def flow_config(cls) -> FlowNodeConf:
        return FlowNodeConf(
            traverse_in=AvailableTraversingIn.AwaitAll,
            traverse_out=AvailableTraversingOut.SpawnPickedNode,
            thought_type=AvailableThoughtTypes.UsePreviousThought1,
            message_type=AvailableMessageTypes.Variable,
            available_thought_types={
                AvailableThoughtTypes.NewThought1,
                AvailableThoughtTypes.NewHiddenThought1,
                AvailableThoughtTypes.NewCollapsedThought1,
                AvailableThoughtTypes.UsePreviousThought1,
            },
        )

    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = "Present choices to user and await selection"
        info.instructions = "Shows options and waits for user decision"
        return info

    @classmethod
    def think(cls, ctx) -> None:
        if ctx.thought_id is None:
            raise MissingMemoryIDException()

        status_updater = ctx.node_metadata.get("_status_updater")
        if status_updater is not None:
            status_updater(ctx.flow_node_id, "AwaitingUserResponse")

        raise ProcessStopException("Awaiting user decision")
