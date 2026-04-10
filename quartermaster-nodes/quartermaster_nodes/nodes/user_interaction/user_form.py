"""UserForm node — collect structured user input."""

from quartermaster_nodes.base import AbstractAssistantNode
from quartermaster_nodes.config import AssistantInfo, FlowNodeConf
from quartermaster_nodes.enums import (
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)
from quartermaster_nodes.exceptions import MissingMemoryIDException, ProcessStopException


class UserFormV1(AbstractAssistantNode):
    """Collect structured input from the user via a form.

    Use Case:
        - Collect multiple structured inputs at once
        - Define parameters for user to fill in
    """

    metadata_parameters_key = "parameters"
    metadata_parameters_default_value = []

    @classmethod
    def name(cls) -> str:
        return "UserForm1"

    @classmethod
    def version(cls) -> str:
        return "1.0.0"

    @classmethod
    def flow_config(cls) -> FlowNodeConf:
        return FlowNodeConf(
            traverse_in=AvailableTraversingIn.AwaitFirst,
            traverse_out=AvailableTraversingOut.SpawnAll,
            thought_type=AvailableThoughtTypes.NewThought1,
            available_thought_types={
                AvailableThoughtTypes.EditSameOrAddNew1,
                AvailableThoughtTypes.NewHiddenThought1,
                AvailableThoughtTypes.NewCollapsedThought1,
                AvailableThoughtTypes.UsePreviousThought1,
            },
            message_type=AvailableMessageTypes.User,
        )

    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = "Collect structured user input via form"
        info.instructions = "Presents a form and waits for user to fill it"
        info.metadata = {
            cls.metadata_parameters_key: cls.metadata_parameters_default_value,
        }
        return info

    @classmethod
    def think(cls, ctx) -> None:
        if ctx.thought_id is None:
            raise MissingMemoryIDException()

        status_updater = ctx.node_metadata.get("_status_updater")
        if status_updater is not None:
            status_updater(ctx.flow_node_id, "AwaitingUserResponse")

        raise ProcessStopException("Awaiting user form input")
