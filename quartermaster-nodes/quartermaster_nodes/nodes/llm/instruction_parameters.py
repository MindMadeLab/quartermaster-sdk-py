"""Instruction with structured parameter output."""

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


class InstructionParameters1(AbstractLLMAssistantNode):
    """LLM instruction with structured parameter output.

    Uses tool calling to extract structured data from the LLM response.

    Use Case:
        - When you need the LLM to output structured data
        - When you need to extract specific parameters from the conversation
    """

    metadata_parameters_key = "parameters"
    metadata_parameters_default_value = []
    metadata_function_name_key = "function_name"
    metadata_function_name_default_value = "extract_parameters"
    metadata_function_description_key = "function_description"
    metadata_function_description_default_value = "Extract structured parameters"
    metadata_stream_default_value = False

    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = "LLM instruction with structured parameter output"
        info.instructions = "Uses tool calling to extract structured data from LLM"
        info.metadata = {
            cls.metadata_system_instruction_key: cls.metadata_system_instruction_default_value,
            cls.metadata_model_key: cls.metadata_model_default_value,
            cls.metadata_provider_key: cls.metadata_provider_default_value,
            cls.metadata_temperature_key: cls.metadata_temperature_default_value,
            cls.metadata_max_input_tokens_key: cls.metadata_max_input_tokens_default_value,
            cls.metadata_max_output_tokens_key: cls.metadata_max_output_tokens_default_value,
            cls.metadata_stream_key: cls.metadata_stream_default_value,
            cls.metadata_parameters_key: cls.metadata_parameters_default_value,
            cls.metadata_function_name_key: cls.metadata_function_name_default_value,
            cls.metadata_function_description_key: cls.metadata_function_description_default_value,
        }
        return info

    @classmethod
    def name(cls) -> str:
        return "InstructionParameters1"

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

        # Build tool from parameters metadata
        from quartermaster_nodes.protocols import ParameterContainer, ProgramContainer

        parameters = cls.get_metadata_key_value(
            ctx, cls.metadata_parameters_key, cls.metadata_parameters_default_value
        )
        func_name = cls.get_metadata_key_value(
            ctx, cls.metadata_function_name_key, cls.metadata_function_name_default_value
        )
        func_desc = cls.get_metadata_key_value(
            ctx,
            cls.metadata_function_description_key,
            cls.metadata_function_description_default_value,
        )

        tool = ProgramContainer(func_name, func_desc)
        for param in parameters:
            tool.add_parameter(
                ParameterContainer(
                    name=param.get("name", ""),
                    type=param.get("type", "string"),
                    description=param.get("description", ""),
                    is_required=param.get("required", False),
                )
            )

        initial_data = {
            "memory_id": ctx.thought_id,
            "flow_node_id": ctx.flow_node_id,
            "ctx": ctx,
        }

        Chain().add_handler(ValidateMemoryID()).add_handler(
            PrepareMessages(client, llm_config)
        ).add_handler(ContextManager(client, llm_config, context_config)).add_handler(
            TransformToProvider(transformer)
        ).add_handler(GenerateToolCall(client, [tool], llm_config)).add_handler(
            ProcessStreamResponse()
        ).run(initial_data)
