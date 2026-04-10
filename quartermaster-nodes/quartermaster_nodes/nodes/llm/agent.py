"""Agent node — autonomous sub-agent with tool orchestration."""

import logging
from typing import List

from quartermaster_nodes.base import AbstractLLMAssistantNode
from quartermaster_nodes.config import AssistantInfo, FlowNodeConf
from quartermaster_nodes.enums import (
    AvailableErrorHandlingStrategies,
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)
from quartermaster_nodes.chain import Chain
from quartermaster_nodes.chain.handlers import (
    ContextManager,
    GenerateNativeResponse,
    PrepareMessages,
    ProcessStreamResponse,
    TransformToProvider,
    ValidateMemoryID,
)
from quartermaster_nodes.protocols import ProgramContainer

logger = logging.getLogger(__name__)


class AgentNodeV1(AbstractLLMAssistantNode):
    """Autonomous agent with native response generation and automatic tool orchestration.

    Runs an agentic loop where the LLM decides whether to generate text
    or call tools, continuing until the task is complete or the max
    iteration limit is reached.

    Use Case:
        - When you need an autonomous agent that can use tools
        - When the LLM should decide when to stop based on task completion
        - When multi-step tool usage is required
    """

    metadata_program_version_ids_key = "program_version_ids"
    metadata_program_version_ids_default = []
    metadata_max_iterations_key = "max_iterations"
    metadata_max_iterations_default = 25

    metadata_tool_clearing_trigger_default_value = 10000
    metadata_tool_clearing_keep_default_value = 3
    metadata_max_tool_result_tokens_default_value = 2000

    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = "Autonomous agent with tool orchestration"
        info.instructions = (
            "Runs an agentic loop where the LLM decides whether to generate text "
            "or call tools, continuing until task completion"
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
            cls.metadata_program_version_ids_key: cls.metadata_program_version_ids_default,
            cls.metadata_max_iterations_key: cls.metadata_max_iterations_default,
            cls.metadata_tool_clearing_trigger_key: cls.metadata_tool_clearing_trigger_default_value,
            cls.metadata_tool_clearing_keep_key: cls.metadata_tool_clearing_keep_default_value,
            cls.metadata_max_tool_result_tokens_key: cls.metadata_max_tool_result_tokens_default_value,
        }
        return info

    @classmethod
    def name(cls) -> str:
        return "AgentNode"

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
            error_handling_strategy=AvailableErrorHandlingStrategies.Retry,
            available_thought_types={
                AvailableThoughtTypes.EditSameOrAddNew1,
                AvailableThoughtTypes.UsePreviousThought1,
                AvailableThoughtTypes.NewCollapsedThought1,
                AvailableThoughtTypes.NewHiddenThought1,
            },
            available_error_handling_strategies={
                AvailableErrorHandlingStrategies.Stop,
                AvailableErrorHandlingStrategies.Continue,
            },
        )

    @classmethod
    def think(cls, ctx) -> None:
        llm_config = cls.llm_config(ctx)
        context_config = cls.context_manager_config(ctx, llm_config)

        max_iterations = cls.get_metadata_key_value(
            ctx, cls.metadata_max_iterations_key, cls.metadata_max_iterations_default
        )

        # Load tools from context
        tool_loader = ctx.node_metadata.get("_tool_loader")
        tools: List[ProgramContainer] = []
        if tool_loader is not None:
            program_ids = cls.get_metadata_key_value(
                ctx,
                cls.metadata_program_version_ids_key,
                cls.metadata_program_version_ids_default,
            )
            tools = tool_loader(program_ids)

        transformer = ctx.node_metadata.get("_transformer")
        client = ctx.node_metadata.get("_client")
        stop_checker = ctx.node_metadata.get("_stop_checker")
        tool_executor = ctx.node_metadata.get("_tool_executor")

        initial_data = {
            "memory_id": ctx.thought_id,
            "flow_node_id": ctx.flow_node_id,
            "ctx": ctx,
        }

        iteration = 0
        while iteration < max_iterations:
            # Check stop signal
            if stop_checker and stop_checker():
                logger.info("Agent stopped by signal at iteration %d", iteration)
                break

            chain = Chain() \
                .add_handler(ValidateMemoryID()) \
                .add_handler(PrepareMessages(client, llm_config)) \
                .add_handler(ContextManager(client, llm_config, context_config)) \
                .add_handler(TransformToProvider(transformer)) \
                .add_handler(GenerateNativeResponse(client, tools, llm_config)) \
                .add_handler(ProcessStreamResponse())

            result = chain.run(initial_data)

            # Check if another call is needed (tool calls were made)
            processed = result.get("processed_response", {})
            tool_calls = processed.get("tool_calls", [])

            if not tool_calls:
                break  # No tool calls = generation complete

            # Execute tools if executor available
            if tool_executor:
                tool_results = tool_executor(tool_calls, ctx)
                initial_data["tool_results"] = tool_results

            iteration += 1

        if iteration >= max_iterations:
            logger.warning(
                "Agent reached max iterations (%d) for flow_node %s",
                max_iterations,
                ctx.flow_node_id,
            )
