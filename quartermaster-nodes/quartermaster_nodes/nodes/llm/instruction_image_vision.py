"""Instruction with image/vision input."""

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


class InstructionImageVision1(AbstractLLMAssistantNode):
    """LLM instruction with image/vision input support.

    Like InstructionNodeV1 but with vision enabled for processing images.

    Use Case:
        - When you need LLM to analyze images
        - When visual context is needed for the response
    """

    metadata_vision_default_value = True

    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = "LLM instruction with image/vision input"
        info.instructions = "Processes images alongside text using a vision-capable LLM"
        info.metadata = {
            cls.metadata_system_instruction_key: cls.metadata_system_instruction_default_value,
            cls.metadata_model_key: cls.metadata_model_default_value,
            cls.metadata_provider_key: cls.metadata_provider_default_value,
            cls.metadata_temperature_key: cls.metadata_temperature_default_value,
            cls.metadata_max_input_tokens_key: cls.metadata_max_input_tokens_default_value,
            cls.metadata_max_output_tokens_key: cls.metadata_max_output_tokens_default_value,
            cls.metadata_stream_key: cls.metadata_stream_default_value,
            cls.metadata_vision_key: cls.metadata_vision_default_value,
        }
        return info

    @classmethod
    def name(cls) -> str:
        return "InstructionImageVision1"

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
