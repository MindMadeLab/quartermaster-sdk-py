"""UserProgramForm node — user selects and configures a tool."""

from quartermaster_nodes.base import AbstractAssistantNode
from quartermaster_nodes.config import AssistantInfo, FlowNodeConf
from quartermaster_nodes.enums import (
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)
from quartermaster_nodes.exceptions import MissingMemoryIDException, ProcessStopException


class UserProgramFormV1(AbstractAssistantNode):
    """User selects a tool and configures its parameters.

    Use Case:
        - Let users choose which tool to run
        - Allow users to configure tool parameters
    """

    metadata_program_version_ids_key = "program_version_ids"
    metadata_program_version_ids_default = []

    @classmethod
    def name(cls) -> str:
        return "UserProgramForm1"

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
                AvailableThoughtTypes.UsePreviousThought1,
            },
            message_type=AvailableMessageTypes.User,
        )

    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = "User selects and configures a tool"
        info.instructions = "Presents tool selection and parameter configuration"
        info.metadata = {
            cls.metadata_program_version_ids_key: cls.metadata_program_version_ids_default,
        }
        return info

    @classmethod
    def think(cls, ctx) -> None:
        if ctx.thought_id is None:
            raise MissingMemoryIDException()

        status_updater = ctx.node_metadata.get("_status_updater")
        if status_updater is not None:
            status_updater(ctx.flow_node_id, "AwaitingUserResponse")

        raise ProcessStopException("Awaiting user tool selection")
