"""Instruction node — the most important LLM node.

Generates responses based on system instructions and conversation history.
"""

import logging

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

logger = logging.getLogger(__name__)


class InstructionNodeV1(AbstractLLMAssistantNode):
    """Generate responses based on system instructions and previous thoughts.

    This is the core LLM node. It sends a prompt with conversation history
    to an LLM and streams the response back.

    Use Case:
        - Generate responses based on specific instructions
        - Create dynamic responses based on user input or context
    """

    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = (
            "Generate a response based on system instructions and conversation history"
        )
        info.instructions = (
            "Send conversation history to an LLM with system instructions and stream the response"
        )
        info.metadata = {
            cls.metadata_system_instruction_key: cls.metadata_system_instruction_default_value,
            cls.metadata_model_key: cls.metadata_model_default_value,
            cls.metadata_provider_key: cls.metadata_provider_default_value,
            cls.metadata_temperature_key: cls.metadata_temperature_default_value,
            cls.metadata_max_input_tokens_key: cls.metadata_max_input_tokens_default_value,
            cls.metadata_max_output_tokens_key: cls.metadata_max_output_tokens_default_value,
            cls.metadata_max_messages_key: cls.metadata_max_messages_default_value,
            cls.metadata_stream_key: cls.metadata_stream_default_value,
            cls.metadata_thinking_level_key: cls.metadata_thinking_level_default_value,
        }
        return info

    @classmethod
    def name(cls) -> str:
        return "InstructionNode"

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
                AvailableMessageTypes.Automatic,
                AvailableMessageTypes.User,
            },
        )

    @classmethod
    def think(cls, ctx) -> None:
        llm_config = cls.llm_config(ctx)
        context_config = cls.context_manager_config(ctx, llm_config)

        # Get transformer and client from context if available
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
