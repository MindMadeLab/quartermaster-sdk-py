"""Text node — Jinja2 template rendering."""

from jinja2 import Template

from quartermaster_nodes.base import AbstractAssistantNode
from quartermaster_nodes.config import AssistantInfo, FlowNodeConf
from quartermaster_nodes.enums import (
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)


class TextNode(AbstractAssistantNode):
    """Render a Jinja2 text template using thought metadata.

    Use Case:
        - Generate dynamic text from templates
        - Format output using variables from previous nodes
    """

    metadata_text_key = "text"
    metadata_text_default = "This is a default text."

    @classmethod
    def name(cls) -> str:
        return "TextNode1"

    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = "Render Jinja2 text template"
        info.instructions = "Renders a template using thought metadata as context"
        info.metadata = {
            cls.metadata_text_key: cls.metadata_text_default,
        }
        return info

    @classmethod
    def flow_config(cls) -> FlowNodeConf:
        return FlowNodeConf(
            traverse_in=AvailableTraversingIn.AwaitFirst,
            traverse_out=AvailableTraversingOut.SpawnAll,
            thought_type=AvailableThoughtTypes.NewThought1,
            message_type=AvailableMessageTypes.Automatic,
            available_thought_types={
                AvailableThoughtTypes.EditSameOrAddNew1,
                AvailableThoughtTypes.UsePreviousThought1,
                AvailableThoughtTypes.NewHiddenThought1,
                AvailableThoughtTypes.NewCollapsedThought1,
                AvailableThoughtTypes.NewThought1,
            },
            available_message_types={
                AvailableMessageTypes.Assistant,
                AvailableMessageTypes.User,
            },
        )

    @classmethod
    def think(cls, ctx) -> None:
        value = cls.get_metadata_key_value(
            ctx, cls.metadata_text_key, cls.metadata_text_default
        )

        if ctx.thought is None:
            raise ValueError("Memory ID cannot be None")

        template = Template(value)
        metadata = ctx.thought.metadata
        rendered_text = template.render(metadata)

        assert ctx.handle is not None, "handle not set"
        ctx.handle.append_text(rendered_text)
