"""SubAssistant node — invoke a sub-flow."""

from quartermaster_nodes.base import AbstractAssistantNode
from quartermaster_nodes.config import AssistantInfo, FlowNodeConf
from quartermaster_nodes.enums import (
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)


class SubAssistant1(AbstractAssistantNode):
    """Invoke a sub-flow (nested flow execution).

    Executes another assistant flow as a sub-routine and returns
    control when the sub-flow completes.

    Use Case:
        - Reuse common flow patterns as sub-routines
        - Modularize complex flows into smaller components
    """

    metadata_sub_assistant_id_key = "sub_assistant_id"
    metadata_sub_assistant_id_default = None

    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = "Invoke a sub-flow as a nested execution"
        info.instructions = "Executes another assistant flow as a sub-routine"
        info.metadata = {
            cls.metadata_sub_assistant_id_key: cls.metadata_sub_assistant_id_default,
        }
        return info

    @classmethod
    def name(cls) -> str:
        return "SubAssistant1"

    @classmethod
    def version(cls) -> str:
        return "1.0"

    @classmethod
    def flow_config(cls) -> FlowNodeConf:
        return FlowNodeConf(
            traverse_in=AvailableTraversingIn.AwaitFirst,
            traverse_out=AvailableTraversingOut.SpawnAll,
            thought_type=AvailableThoughtTypes.NewThought1,
            message_type=AvailableMessageTypes.Assistant,
            available_thought_types={
                AvailableThoughtTypes.EditSameOrAddNew1,
                AvailableThoughtTypes.UsePreviousThought1,
                AvailableThoughtTypes.NewHiddenThought1,
            },
        )

    @classmethod
    def think(cls, ctx) -> None:
        sub_flow_runner = ctx.node_metadata.get("_sub_flow_runner")
        sub_assistant_id = cls.get_metadata_key_value(
            ctx, cls.metadata_sub_assistant_id_key, cls.metadata_sub_assistant_id_default
        )

        if sub_flow_runner is not None and sub_assistant_id is not None:
            sub_flow_runner(sub_assistant_id, ctx)
