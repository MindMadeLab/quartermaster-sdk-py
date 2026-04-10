"""Reasoning node — extended thinking with chain-of-thought."""

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
    ValidateMemoryID,
)


class ReasoningV1(AbstractLLMAssistantNode):
    """Extended thinking / chain-of-thought reasoning without explicit instructions.

    Specialized for reasoning models (o1-mini, o1) that handle
    system instructions internally.

    Use Case:
        - When you need deep reasoning about a complex problem
        - When chain-of-thought analysis is required
    """

    metadata_system_instruction_default_value = None
    metadata_model_default_value = "o1-mini"
    metadata_provider_default_value = "openai"
    metadata_temperature_default_value = None
    metadata_max_input_tokens_default_value = 32768
    metadata_max_output_tokens_default_value = None
    metadata_max_messages_default_value = None
    metadata_stream_default_value = True

    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = "Provides reasoning based on previous thoughts without explicit instructions"
        info.instructions = "Uses reasoning models for chain-of-thought analysis"
        info.metadata = {
            cls.metadata_model_key: cls.metadata_model_default_value,
            cls.metadata_temperature_key: cls.metadata_temperature_default_value,
            cls.metadata_max_input_tokens_key: cls.metadata_max_input_tokens_default_value,
            cls.metadata_max_output_tokens_key: cls.metadata_max_output_tokens_default_value,
            cls.metadata_max_messages_key: cls.metadata_max_messages_default_value,
        }
        return info

    @classmethod
    def name(cls) -> str:
        return "ReasoningNode"

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
                AvailableThoughtTypes.NewCollapsedThought1,
            },
            available_message_types={
                AvailableMessageTypes.User,
                AvailableMessageTypes.Automatic,
            },
        )

    @classmethod
    def think(cls, ctx) -> None:
        llm_config = cls.llm_config(ctx)
        context_config = cls.context_manager_config(ctx, llm_config)

        transformer = ctx.node_metadata.get("_transformer")
        client = ctx.node_metadata.get("_client")

        initial_data = {
            "memory_id": ctx.thought_id,
            "flow_node_id": ctx.flow_node_id,
            "ctx": ctx,
        }

        Chain() \
            .add_handler(ValidateMemoryID()) \
            .add_handler(PrepareMessages(client, llm_config)) \
            .add_handler(ContextManager(client, llm_config, context_config)) \
            .add_handler(TransformToProvider(transformer)) \
            .add_handler(GenerateStreamResponse(client, llm_config)) \
            .add_handler(ProcessStreamResponse()) \
            .run(initial_data)
