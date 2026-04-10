"""ProgramRunner node — execute a registered tool/program."""

from quartermaster_nodes.base import AbstractAssistantNode
from quartermaster_nodes.config import AssistantInfo, FlowNodeConf
from quartermaster_nodes.enums import (
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)


class ProgramRunner1(AbstractAssistantNode):
    """Execute a registered tool or program.

    Use Case:
        - Run specific tools with predefined or dynamic parameters
        - Integrate external services into the flow
    """

    metadata_program_version_id_key = "program_version_id"
    metadata_program_version_id_default = None
    metadata_parameters_key = "parameters"
    metadata_parameters_default = {}

    @classmethod
    def name(cls) -> str:
        return "ProgramRunner1"

    @classmethod
    def version(cls) -> str:
        return "1.0"

    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = "Execute a registered tool or program"
        info.instructions = "Runs a tool with the given parameters"
        info.metadata = {
            cls.metadata_program_version_id_key: cls.metadata_program_version_id_default,
            cls.metadata_parameters_key: cls.metadata_parameters_default,
        }
        return info

    @classmethod
    def flow_config(cls) -> FlowNodeConf:
        return FlowNodeConf(
            traverse_in=AvailableTraversingIn.AwaitFirst,
            traverse_out=AvailableTraversingOut.SpawnAll,
            thought_type=AvailableThoughtTypes.NewThought1,
            message_type=AvailableMessageTypes.Tool,
            available_thought_types={
                AvailableThoughtTypes.EditSameOrAddNew1,
                AvailableThoughtTypes.UsePreviousThought1,
                AvailableThoughtTypes.NewHiddenThought1,
            },
        )

    @classmethod
    def think(cls, ctx) -> None:
        program_executor = ctx.node_metadata.get("_program_executor")
        program_id = cls.get_metadata_key_value(
            ctx, cls.metadata_program_version_id_key, cls.metadata_program_version_id_default
        )
        parameters = cls.get_metadata_key_value(
            ctx, cls.metadata_parameters_key, cls.metadata_parameters_default
        )

        if program_executor is not None and program_id is not None:
            result = program_executor(program_id, parameters, ctx)
            if result is not None and ctx.handle is not None:
                ctx.handle.append_text(str(result))
