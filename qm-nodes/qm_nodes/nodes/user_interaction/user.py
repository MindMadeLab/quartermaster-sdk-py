"""User node — pause flow and await user input."""

from qm_nodes.base import AbstractAssistantNode
from qm_nodes.config import AssistantInfo, FlowNodeConf
from qm_nodes.enums import (
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)
from qm_nodes.exceptions import MissingMemoryIDException, ProcessStopException


class UserNode1(AbstractAssistantNode):
    """Pause flow execution and wait for user input.

    Use Case:
        - Collect user messages during a flow
        - Create interactive conversation points
    """

    metadata_text_snippets_key = "text_snippets"
    metadata_text_snippets_default = []

    @classmethod
    def version(cls) -> str:
        return "1.0.0"

    @classmethod
    def name(cls) -> str:
        return "UserAssistant1"

    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = "Pause flow and await user input"
        info.instructions = "Stops flow execution until the user provides input"
        info.metadata = {
            cls.metadata_text_snippets_key: cls.metadata_text_snippets_default,
        }
        return info

    @classmethod
    def flow_config(cls) -> FlowNodeConf:
        return FlowNodeConf(
            traverse_in=AvailableTraversingIn.AwaitFirst,
            traverse_out=AvailableTraversingOut.SpawnAll,
            thought_type=AvailableThoughtTypes.NewThought1,
            available_thought_types={AvailableThoughtTypes.EditSameOrAddNew1},
            available_traversing_out={AvailableTraversingOut.SpawnNone},
            message_type=AvailableMessageTypes.User,
        )

    @classmethod
    def think(cls, ctx) -> None:
        if ctx.thought_id is None:
            raise MissingMemoryIDException()

        # Notify the runtime that we're awaiting user input
        status_updater = ctx.node_metadata.get("_status_updater")
        if status_updater is not None:
            status_updater(ctx.flow_node_id, "AwaitingUserResponse")

        raise ProcessStopException("Awaiting user input")
