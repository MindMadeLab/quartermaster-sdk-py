"""Instruction with tool execution — LLM + tools."""

from qm_nodes.base import AbstractLLMAssistantNode
from qm_nodes.config import AssistantInfo, FlowNodeConf
from qm_nodes.enums import (
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)
from qm_nodes.chain import Chain
from qm_nodes.chain.handlers import (
    ContextManager,
    GenerateStreamResponse,
    PrepareMessages,
    ProcessStreamResponse,
    TransformToProvider,
    ValidateMemoryID,
)


class InstructionProgram1(AbstractLLMAssistantNode):
    """LLM instruction with tool/program execution.

    Generates a response and can execute tools as part of the response.

    Use Case:
        - When the LLM needs to call tools while generating a response
        - When tool results should be incorporated into the output
    """

    metadata_program_version_ids_key = "program_version_ids"
    metadata_program_version_ids_default = []

    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = "LLM instruction with tool execution"
        info.instructions = "Generates a response and executes tools as needed"
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
        return "InstructionProgram1"

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
