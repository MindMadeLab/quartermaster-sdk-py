"""Static program parameters — static tool parameter values."""

from qm_nodes.base import AbstractAssistantNode
from qm_nodes.config import AssistantInfo, FlowNodeConf
from qm_nodes.enums import (
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)


class StaticProgramParameters1(AbstractAssistantNode):
    """Provide static tool parameters without LLM.

    Use Case:
        - Pre-configure tool parameters for subsequent program execution
        - Pass static configuration to tools
    """

    metadata_parameters_key = "parameters"
    metadata_parameters_default = {}
    metadata_program_name_key = "program_name"
    metadata_program_name_default = ""

    @classmethod
    def name(cls) -> str:
        return "StaticProgramParameters1"

    @classmethod
    def version(cls) -> str:
        return "1.0"

    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = "Provide static tool parameters"
        info.instructions = "Pre-configure tool parameters without LLM"
        info.metadata = {
            cls.metadata_parameters_key: cls.metadata_parameters_default,
            cls.metadata_program_name_key: cls.metadata_program_name_default,
        }
        return info

    @classmethod
    def flow_config(cls) -> FlowNodeConf:
        return FlowNodeConf(
            traverse_in=AvailableTraversingIn.AwaitFirst,
            traverse_out=AvailableTraversingOut.SpawnAll,
            thought_type=AvailableThoughtTypes.NewThought1,
            message_type=AvailableMessageTypes.Tool,
        )

    @classmethod
    def think(cls, ctx) -> None:
        parameters = cls.get_metadata_key_value(
            ctx, cls.metadata_parameters_key, cls.metadata_parameters_default
        )
        assert ctx.handle is not None, "handle not set"
        ctx.handle.update_metadata(parameters)
