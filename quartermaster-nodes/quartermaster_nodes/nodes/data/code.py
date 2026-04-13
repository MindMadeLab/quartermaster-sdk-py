"""Code node — execute custom Python code."""

from quartermaster_nodes.base import AbstractAssistantNode
from quartermaster_nodes.config import AssistantInfo, FlowNodeConf
from quartermaster_nodes.enums import (
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)


class CodeNode(AbstractAssistantNode):
    """Execute custom Python code within the flow.

    Use Case:
        - Custom data processing or manipulation
        - Define new variables or functions for subsequent nodes
    """

    metadata_code_key = "code"
    metadata_code_default = ""
    metadata_filename_key = "filename"
    metadata_filename_default = ""

    @classmethod
    def name(cls) -> str:
        return "Code1"

    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = "Execute custom Python code"
        info.instructions = "Runs Python code as part of the agent workflow"
        info.metadata = {
            cls.metadata_code_key: cls.metadata_code_default,
            cls.metadata_filename_key: cls.metadata_filename_default,
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
        pass  # Code execution is handled by the runtime environment
