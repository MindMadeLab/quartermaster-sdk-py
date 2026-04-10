"""Abstract base classes for all nodes."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional
from uuid import UUID

from qm_nodes.config import AssistantInfo, FlowNodeConf
from qm_nodes.protocols import (
    ContextManagerConfig,
    LLMConfig,
    NodeContext,
)

logger = logging.getLogger(__name__)


class AbstractAssistantNode(ABC):
    """Base class for all assistant nodes.

    All nodes are stateless and use class methods. They receive a
    NodeContext, do work, and return results through the context's handle.
    """

    @classmethod
    @abstractmethod
    def info(cls) -> AssistantInfo:
        """Return metadata about this node type."""
        pass

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """Return the node's display name."""
        pass

    @classmethod
    @abstractmethod
    def version(cls) -> str:
        """Return the node's version string."""
        pass

    @classmethod
    def deprecated(cls) -> bool:
        """Whether this node type is deprecated."""
        return False

    @classmethod
    @abstractmethod
    def flow_config(cls) -> FlowNodeConf:
        """Define the node's flow traversal configuration."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def think(cls, ctx: NodeContext) -> None:
        """Execute the node's core logic.

        Args:
            ctx: The execution context providing access to thought state,
                 metadata, and the handle for producing output.
        """
        pass

    @classmethod
    def get_metadata_key_value(
        cls,
        ctx: NodeContext,
        key: str,
        default: Any = None,
    ) -> Any:
        """Retrieve a value from node metadata with a default."""
        try:
            return ctx.node_metadata.get(key, default)
        except Exception as e:
            logger.error(
                "Error retrieving metadata for flow_node %s, key %s: %s",
                ctx.flow_node_id,
                key,
                str(e),
            )
            return default

    @classmethod
    def store_metadata_key_value(
        cls,
        ctx: NodeContext,
        key: str,
        value: Any,
    ) -> None:
        """Store a value in node metadata.

        Override this method to integrate with your persistence layer.
        The default implementation updates the context's metadata dict.
        """
        ctx.node_metadata[key] = value


class AbstractLLMAssistantNode(AbstractAssistantNode):
    """Base class for nodes that call LLM providers.

    Adds LLM-specific configuration: model, provider, temperature,
    token limits, system instructions, and context management settings.
    """

    # Model configuration
    metadata_model_key = "llm_model"
    metadata_model_default_value = "gpt-4o-mini"

    metadata_provider_key = "llm_provider"
    metadata_provider_default_value = "openai"

    metadata_temperature_key = "llm_temperature"
    metadata_temperature_default_value = 0.5

    # Token limits
    metadata_max_input_tokens_key = "llm_max_input_tokens"
    metadata_max_input_tokens_default_value = 16385

    metadata_max_output_tokens_key = "llm_max_output_tokens"
    metadata_max_output_tokens_default_value = 2048

    metadata_max_messages_key = "llm_max_messages"
    metadata_max_messages_default_value = None

    # Behavior
    metadata_stream_key = "llm_stream"
    metadata_stream_default_value = True

    metadata_vision_key = "llm_vision"
    metadata_vision_default_value = False

    metadata_system_instruction_key = "llm_system_instruction"
    metadata_system_instruction_default_value = (
        "You are helpful agent, try being precise and helpful."
    )

    metadata_thinking_level_key = "llm_thinking_level"
    metadata_thinking_level_default_value = "off"

    # Context management
    metadata_tool_clearing_trigger_key = "context_tool_clearing_trigger"
    metadata_tool_clearing_trigger_default_value = None

    metadata_tool_clearing_keep_key = "context_tool_clearing_keep"
    metadata_tool_clearing_keep_default_value = None

    metadata_exclude_tools_key = "context_exclude_tools"
    metadata_exclude_tools_default_value: list = []

    metadata_max_tool_result_tokens_key = "context_max_tool_result_tokens"
    metadata_max_tool_result_tokens_default_value = None

    metadata_image_clearing_trigger_key = "context_image_clearing_trigger"
    metadata_image_clearing_trigger_default_value = None

    metadata_image_clearing_keep_key = "context_image_clearing_keep"
    metadata_image_clearing_keep_default_value = None

    THINKING_LEVELS = {
        "off": (False, None),
        "low": (True, 1024),
        "medium": (True, 4096),
        "high": (True, 16384),
    }

    @classmethod
    def llm_config(cls, ctx: NodeContext) -> LLMConfig:
        """Build LLM configuration from node metadata."""
        meta = ctx.node_metadata

        provider = meta.get(cls.metadata_provider_key, cls.metadata_provider_default_value)
        model = meta.get(cls.metadata_model_key, cls.metadata_model_default_value)
        temperature = meta.get(cls.metadata_temperature_key, cls.metadata_temperature_default_value)
        if temperature is not None and isinstance(temperature, str):
            temperature = float(temperature)

        max_input_tokens = meta.get(
            cls.metadata_max_input_tokens_key, cls.metadata_max_input_tokens_default_value
        )
        max_output_tokens = meta.get(
            cls.metadata_max_output_tokens_key, cls.metadata_max_output_tokens_default_value
        )
        max_messages = meta.get(
            cls.metadata_max_messages_key, cls.metadata_max_messages_default_value
        )
        system_message = meta.get(
            cls.metadata_system_instruction_key, cls.metadata_system_instruction_default_value
        )
        stream = meta.get(cls.metadata_stream_key, cls.metadata_stream_default_value)
        vision = meta.get(cls.metadata_vision_key, cls.metadata_vision_default_value)

        thinking_level = meta.get(
            cls.metadata_thinking_level_key, cls.metadata_thinking_level_default_value
        )
        thinking_enabled, thinking_budget = cls.THINKING_LEVELS.get(
            thinking_level, (False, None)
        )

        return LLMConfig(
            model=model,
            provider=provider,
            vision=vision,
            stream=stream,
            system_message=system_message,
            temperature=temperature,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            max_messages=max_messages,
            thinking_enabled=thinking_enabled,
            thinking_budget=thinking_budget,
        )

    @classmethod
    def context_manager_config(
        cls,
        ctx: NodeContext,
        llm_config: LLMConfig,
    ) -> ContextManagerConfig:
        """Build context management configuration from node metadata."""
        return ContextManagerConfig(
            tool_clearing_trigger=cls.get_metadata_key_value(
                ctx,
                cls.metadata_tool_clearing_trigger_key,
                cls.metadata_tool_clearing_trigger_default_value,
            ),
            tool_clearing_keep=cls.get_metadata_key_value(
                ctx,
                cls.metadata_tool_clearing_keep_key,
                cls.metadata_tool_clearing_keep_default_value,
            ),
            exclude_tools=cls.get_metadata_key_value(
                ctx,
                cls.metadata_exclude_tools_key,
                cls.metadata_exclude_tools_default_value,
            ),
            max_tool_result_tokens=cls.get_metadata_key_value(
                ctx,
                cls.metadata_max_tool_result_tokens_key,
                cls.metadata_max_tool_result_tokens_default_value,
            ),
            image_clearing_trigger=cls.get_metadata_key_value(
                ctx,
                cls.metadata_image_clearing_trigger_key,
                cls.metadata_image_clearing_trigger_default_value,
            ),
            image_clearing_keep=cls.get_metadata_key_value(
                ctx,
                cls.metadata_image_clearing_keep_key,
                cls.metadata_image_clearing_keep_default_value,
            ),
            max_messages=llm_config.max_messages or 0,
            max_input_tokens=llm_config.max_input_tokens or 0,
        )
