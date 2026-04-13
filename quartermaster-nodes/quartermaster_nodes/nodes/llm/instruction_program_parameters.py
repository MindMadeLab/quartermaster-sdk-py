"""Instruction with tools and structured output."""

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
    GenerateToolCall,
    PrepareMessages,
    ProcessStreamResponse,
    TransformToProvider,
    ValidateMemoryID,
)


class InstructionProgramParameters1(AbstractLLMAssistantNode):
    """LLM instruction with tools and structured parameter output.

    Combines tool execution with structured output extraction.

    Use Case:
        - When you need both tool execution and structured output
        - Complex workflows requiring tool use and data extraction
    """

    metadata_program_version_ids_key = "program_version_ids"
    metadata_program_version_ids_default = []
    metadata_stream_default_value = False

    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = "LLM with tools and structured output"
        info.instructions = "Combines tool execution with structured parameter extraction"
        info.metadata = {
            cls.metadata_system_instruction_key: cls.metadata_system_instruction_default_value,
            cls.metadata_model_key: cls.metadata_model_default_value,
            cls.metadata_provider_key: cls.metadata_provider_default_value,
            cls.metadata_temperature_key: cls.metadata_temperature_default_value,
            cls.metadata_max_input_tokens_key: cls.metadata_max_input_tokens_default_value,
            cls.metadata_max_output_tokens_key: cls.metadata_max_output_tokens_default_value,
            cls.metadata_stream_key: cls.metadata_stream_default_value,
            cls.metadata_program_version_ids_key: cls.metadata_program_version_ids_default,
        }
        return info

    @classmethod
    def name(cls) -> str:
        return "InstructionProgramParameters1"

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
            },
        )

    @classmethod
    def think(cls, ctx) -> None:
        llm_config = cls.llm_config(ctx)
        llm_config.stream = False
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
            .add_handler(GenerateToolCall(client, [], llm_config)) \
            .add_handler(ProcessStreamResponse()) \
            .run(initial_data)
