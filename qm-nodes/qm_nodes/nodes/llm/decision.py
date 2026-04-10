"""Decision node — LLM picks which path to follow."""

from qm_nodes.base import AbstractLLMAssistantNode
from qm_nodes.config import AssistantInfo, FlowNodeConf
from qm_nodes.enums import (
    NEXT_ASSISTANT_NODE_ID,
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)
from qm_nodes.chain import Chain
from qm_nodes.chain.handlers import (
    ContextManager,
    GenerateToolCall,
    PrepareMessages,
    ProcessStreamResponse,
    TransformToProvider,
    ValidateMemoryID,
)
from qm_nodes.protocols import ParameterContainer, ProgramContainer


class Decision1(AbstractLLMAssistantNode):
    """LLM-driven path selection based on available connections.

    The LLM evaluates the conversation context and picks the best path
    to follow by calling a tool with the selected edge ID.

    Use Case:
        - When you want LLM to decide which path to take based on previous messages
        - When you want to create dynamic flows based on user input or context
    """

    metadata_prefix_message_key = "prefix_message"
    metadata_prefix_message_default_value = (
        "Based on previous messages pick the best path to take. "
        "Here are available options:"
    )
    metadata_suffix_message_key = "suffix_message"
    metadata_suffix_message_default_value = ""
    metadata_model_default_value = "gpt-4o-mini"
    metadata_provider_default_value = "openai"
    metadata_system_instruction_default_value = (
        "You are helpful assistant. Your task is to decide which path "
        "to take based on previous messages and call function pick_path."
    )
    metadata_max_input_tokens_default_value = 32768
    metadata_stream_default_value = False

    @classmethod
    def name(cls) -> str:
        return "Decision1"

    @classmethod
    def version(cls) -> str:
        return "1.0.0"

    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = "Use LLM to decide which path to take"
        info.instructions = "Use LLM to decide which path to take based on conversation context"
        info.metadata = {
            cls.metadata_system_instruction_key: cls.metadata_system_instruction_default_value,
            cls.metadata_model_key: cls.metadata_model_default_value,
            cls.metadata_provider_key: cls.metadata_provider_default_value,
            cls.metadata_temperature_key: cls.metadata_temperature_default_value,
            cls.metadata_max_input_tokens_key: cls.metadata_max_input_tokens_default_value,
            cls.metadata_max_output_tokens_key: cls.metadata_max_output_tokens_default_value,
            cls.metadata_max_messages_key: cls.metadata_max_messages_default_value,
            cls.metadata_prefix_message_key: cls.metadata_prefix_message_default_value,
            cls.metadata_suffix_message_key: cls.metadata_suffix_message_default_value,
            cls.metadata_stream_key: cls.metadata_stream_default_value,
        }
        return info

    @classmethod
    def flow_config(cls) -> FlowNodeConf:
        return FlowNodeConf(
            thought_type=AvailableThoughtTypes.UsePreviousThought1,
            traverse_in=AvailableTraversingIn.AwaitFirst,
            traverse_out=AvailableTraversingOut.SpawnPickedNode,
            message_type=AvailableMessageTypes.Variable,
        )

    @classmethod
    def create_decision_tool(cls) -> ProgramContainer:
        """Create the pick_path tool for LLM decision making."""
        parameter = ParameterContainer(
            NEXT_ASSISTANT_NODE_ID,
            "string",
            "The id of the edge to follow",
            is_required=True,
        )
        tool = ProgramContainer(
            "pick_path",
            "Choose the best path to continue the conversation",
        )
        tool.add_parameter(parameter)
        return tool

    @classmethod
    def prepare_decision_message(cls, ctx) -> str:
        """Build the decision message with available edge options."""
        prefix_message = cls.get_metadata_key_value(
            ctx,
            cls.metadata_prefix_message_key,
            cls.metadata_prefix_message_default_value,
        )

        edges = ctx.assistant_node.predecessor_edges.all()
        available_connections = "\n".join(
            f"- {e.tail_id}: {e.direction_text}" for e in edges
        )

        suffix_message = cls.get_metadata_key_value(
            ctx,
            cls.metadata_suffix_message_key,
            cls.metadata_suffix_message_default_value,
        )

        return f"{prefix_message}\n\n{available_connections}\n\n{suffix_message}".strip()

    @classmethod
    def think(cls, ctx) -> None:
        llm_config = cls.llm_config(ctx)
        llm_config.stream = False

        transformer = ctx.node_metadata.get("_transformer")
        client = ctx.node_metadata.get("_client")
        context_config = cls.context_manager_config(ctx, llm_config)

        initial_data = {
            "memory_id": ctx.thought_id,
            "flow_node_id": ctx.flow_node_id,
            "ctx": ctx,
        }

        decision_message = cls.prepare_decision_message(ctx)
        tool = cls.create_decision_tool()

        Chain() \
            .add_handler(ValidateMemoryID()) \
            .add_handler(PrepareMessages(client, llm_config, decision_message)) \
            .add_handler(ContextManager(client, llm_config, context_config)) \
            .add_handler(TransformToProvider(transformer)) \
            .add_handler(GenerateToolCall(client, [tool], llm_config)) \
            .add_handler(ProcessStreamResponse(only_first_tool=True)) \
            .run(initial_data)
