"""Merge node — combine multiple parallel conversation branches."""

from quartermaster_nodes.base import AbstractLLMAssistantNode
from quartermaster_nodes.config import AssistantInfo, FlowNodeConf
from quartermaster_nodes.enums import (
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)
from quartermaster_nodes.chain import Chain
from quartermaster_nodes.chain.handlers import (
    ContextManager,
    GenerateStreamResponse,
    PrepareMessages,
    ProcessStreamResponse,
    TransformToProvider,
)


class Merge1(AbstractLLMAssistantNode):
    """Combine multiple parallel conversation branches into a single response.

    Waits for ALL incoming branches, collects their text, and uses LLM
    to merge them into a coherent single message.

    Use Case:
        - When parallel branches need to be combined
        - When you need a unified summary of multiple conversation paths
    """

    metadata_prefix_message_key = "prefix_message"
    metadata_prefix_message_default_value = "Compress following conversations into one"
    metadata_suffix_message_key = "suffix_message"
    metadata_suffix_message_default_value = ""
    metadata_model_default_value = "gpt-4o-mini"
    metadata_provider_default_value = "openai"
    metadata_system_instruction_default_value = (
        "You are helpful assistant. Your task is to combine given messages "
        "from different conversations into one."
    )

    @classmethod
    def name(cls) -> str:
        return "Merge1"

    @classmethod
    def version(cls) -> str:
        return "1.0.0"

    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = "Merge multiple conversation branches into one"
        info.instructions = "Combines messages from parallel branches using LLM"
        info.metadata = {
            cls.metadata_prefix_message_key: cls.metadata_prefix_message_default_value,
            cls.metadata_suffix_message_key: cls.metadata_suffix_message_default_value,
            cls.metadata_system_instruction_key: cls.metadata_system_instruction_default_value,
            cls.metadata_model_key: cls.metadata_model_default_value,
            cls.metadata_provider_key: cls.metadata_provider_default_value,
            cls.metadata_temperature_key: cls.metadata_temperature_default_value,
            cls.metadata_max_input_tokens_key: cls.metadata_max_input_tokens_default_value,
            cls.metadata_max_output_tokens_key: cls.metadata_max_output_tokens_default_value,
            cls.metadata_max_messages_key: cls.metadata_max_messages_default_value,
            cls.metadata_stream_key: cls.metadata_stream_default_value,
        }
        return info

    @classmethod
    def flow_config(cls) -> FlowNodeConf:
        return FlowNodeConf(
            thought_type=AvailableThoughtTypes.NewThought1,
            traverse_in=AvailableTraversingIn.AwaitAll,
            traverse_out=AvailableTraversingOut.SpawnAll,
            message_type=AvailableMessageTypes.Assistant,
            available_thought_types={
                AvailableThoughtTypes.NewHiddenThought1,
                AvailableThoughtTypes.NewCollapsedThought1,
            },
            available_message_types={
                AvailableMessageTypes.User,
                AvailableMessageTypes.Automatic,
            },
        )

    @classmethod
    def prepare_message(cls, ctx) -> str:
        """Build the merge message from previous child thoughts."""
        prefix = cls.get_metadata_key_value(
            ctx, cls.metadata_prefix_message_key, cls.metadata_prefix_message_default_value
        )
        suffix = cls.get_metadata_key_value(
            ctx, cls.metadata_suffix_message_key, cls.metadata_suffix_message_default_value
        )

        thought = ctx.thought
        previous_thoughts = thought.get_previous_child_thoughts()
        content = "\n\n".join(prev.text for prev in previous_thoughts)

        return f"{prefix}\n\n{content}\n\n{suffix}".strip()

    @classmethod
    def think(cls, ctx) -> None:
        if not ctx.thought:
            raise ValueError("Memory ID is required for merge node")

        llm_config = cls.llm_config(ctx)
        context_config = cls.context_manager_config(ctx, llm_config)

        transformer = ctx.node_metadata.get("_transformer")
        client = ctx.node_metadata.get("_client")

        initial_data = {
            "flow_node_id": ctx.flow_node_id,
            "memory_id": ctx.thought_id,
            "to_memory_id": ctx.thought_id,
            "ctx": ctx,
        }

        content = cls.prepare_message(ctx)

        Chain() \
            .add_handler(
                PrepareMessages(
                    client, llm_config,
                    additional_message=content,
                    additional_message_role="assistant",
                )
            ) \
            .add_handler(ContextManager(client, llm_config, context_config)) \
            .add_handler(TransformToProvider(transformer)) \
            .add_handler(GenerateStreamResponse(client, llm_config)) \
            .add_handler(ProcessStreamResponse("to_memory_id")) \
            .run(initial_data)
